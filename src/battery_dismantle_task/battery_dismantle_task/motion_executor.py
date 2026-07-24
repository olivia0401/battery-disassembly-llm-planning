#!/usr/bin/python3
"""Motion planning and execution for robot manipulator"""

import time
import threading
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (Constraints, JointConstraint, MotionPlanRequest, RobotState,
                             PositionIKRequest, PlanningSceneComponents)
from moveit_msgs.srv import GetPositionIK, GetPlanningScene, GetPositionFK
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

ARM_JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]
from .motion_config import (
    DEFAULT_VELOCITY_SCALE,
    DEFAULT_ACCELERATION_SCALE,
    JOINT_POSITION_TOLERANCE,
    VISUALIZATION_SLOW_FACTOR,
    PLANNING_TIMEOUT,
    EXECUTION_TIMEOUT,
    PLANNING_ATTEMPTS,
    PLANNING_TIME
)


class MotionExecutor:
    """Handles motion planning and trajectory execution"""

    def __init__(self, node, move_group_client, manipulator_controller, gripper_controller):
        self.node = node
        self._move_group_action_client = move_group_client
        self._manipulator_controller_client = manipulator_controller
        self._gripper_controller_client = gripper_controller
        self.manipulator_group_name = "manipulator"
        self.vel_scale = DEFAULT_VELOCITY_SCALE
        self.acc_scale = DEFAULT_ACCELERATION_SCALE
        # Direct joint-space execution (linear interpolation current->target),
        # bypassing OMPL/RRTConnect. Enabled because RRTConnect produced wildly
        # indirect paths for near-target moves (measured ~70x redundant joint
        # travel: tiny net change but huge back-and-forth excursions), which is
        # what looked like the arm "moving back and forth" in RViz. The demo
        # scene is simple (approach from above, ACM allows gripper-object
        # contact), so direct moves are clean and safe here.
        self.use_direct_execution = True  # Method 6: direct trajectory execution
        # IK service client; injected by skill_server (must share the action
        # ReentrantCallbackGroup so its response can be processed while a
        # command-callback worker thread is blocked on the IK future).
        self.ik_client = None
        # Planning-scene query client (same ReentrantCallbackGroup rationale);
        # lets approach poses be derived from where an object ACTUALLY is.
        self.scene_query_client = None
        # FK client (injected by skill_server); lets release place a part at the
        # gripper's true commanded-joint pose so it doesn't teleport to a spot
        # the gripper never reached.
        self.fk_client = None
        # Latest arm joint positions, tracked from /joint_states. Used as the
        # DEFAULT IK seed so every pose solves for the configuration nearest the
        # arm's current one. Seeding from a fixed pose (HOME) instead made
        # consecutive IK calls land on unrelated branches, so the arm swung
        # wildly between steps (up to ~3.8 rad on a single joint) and the held
        # part appeared to fly around. Seeding from the current state keeps each
        # move minimal and the motion smooth.
        self._latest_arm_joints = None
        self.node.create_subscription(
            JointState, "/joint_states", self._on_joint_state, 10)

    def _on_joint_state(self, msg):
        pos = dict(zip(msg.name, msg.position))
        if all(j in pos for j in ARM_JOINT_NAMES):
            self._latest_arm_joints = [pos[j] for j in ARM_JOINT_NAMES]

    def plan_execute_arm(self, joints, where):
        """Plan and execute arm motion to target joint positions"""
        self.node.get_logger().info(f"Executing arm motion for: {where}")

        # Method 6: Use direct trajectory execution (bypass MoveIt planning)
        if self.use_direct_execution:
            self.node.get_logger().info("Method 6: Using direct trajectory execution (GitHub method)")
            return self._execute_direct_trajectory(joints, where)

        # REMOVED: rclpy.spin_once(self.node, timeout_sec=0.5) used to be here.
        # This node is already added to a MultiThreadedExecutor spinning in a
        # background daemon thread (skill_server.py main()), and this method
        # runs INSIDE one of that executor's worker callbacks. Manually
        # calling spin_once on the same node from inside an already-spinning
        # callback is an unsafe nested-spin pattern in rclpy: it can steal a
        # callback (e.g. the action client's goal-response callback) that the
        # background executor thread was waiting to deliver to
        # _wait_for_planning_result's on_goal_response, which is exactly what
        # was observed — move_group genuinely planned and executed the motion
        # ("Solution was found and executed"), but skill_server's own
        # on_goal_response callback never fired, so _wait_for_event always
        # hit its 20s timeout and reported failure despite real success.

        # Build MoveGroup goal with joint constraints
        move_goal = MoveGroup.Goal()
        move_goal.request = MotionPlanRequest()
        move_goal.request.group_name = self.manipulator_group_name
        move_goal.request.num_planning_attempts = PLANNING_ATTEMPTS
        move_goal.request.allowed_planning_time = PLANNING_TIME
        move_goal.request.max_velocity_scaling_factor = self.vel_scale
        move_goal.request.max_acceleration_scaling_factor = self.acc_scale

        # Leave start_state empty - MoveIt will use current state from CurrentStateMonitor
        # DO NOT set is_diff or populate start_state manually

        # Set joint constraints
        move_goal.request.goal_constraints.append(Constraints())
        joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]
        for jname, jval in zip(joint_names, joints):
            jc = JointConstraint()
            jc.joint_name = jname
            jc.position = jval
            jc.tolerance_above = JOINT_POSITION_TOLERANCE
            jc.tolerance_below = JOINT_POSITION_TOLERANCE
            jc.weight = 1.0
            move_goal.request.goal_constraints[0].joint_constraints.append(jc)

        move_goal.planning_options.plan_only = False  # EXECUTE the motion!
        move_goal.planning_options.replan = True
        move_goal.planning_options.replan_attempts = 5

        # Send goal and wait for planning + execution result
        if not self._wait_for_planning_result(move_goal):
            return False

        return True

    def compute_ik(self, position, orientation, seed_joints=None,
                   ik_link="end_effector_link", frame="world", timeout_s=2.0):
        """Compute arm joint values for an end-effector pose via MoveIt /compute_ik.

        position: (x, y, z) in `frame`. orientation: (x, y, z, w) quaternion.
        Returns a 7-element list [joint_1..joint_7] on success, or None on
        failure (service missing, unreachable pose, or timeout). Lets the
        approach pose be derived from an object's Cartesian position at runtime
        instead of a hardcoded joint array.
        """
        if self.ik_client is None:
            self.node.get_logger().warn("compute_ik: no IK client injected")
            return None
        if not self.ik_client.service_is_ready() and not self.ik_client.wait_for_service(timeout_sec=3.0):
            self.node.get_logger().warn("compute_ik: /compute_ik service unavailable")
            return None

        # Default to seeding from the arm's CURRENT configuration so IK returns
        # the branch NEAREST where the arm already is — keeps each move small
        # and smooth. Callers may still pass an explicit seed to override.
        if seed_joints is None:
            seed_joints = self._latest_arm_joints

        # Build the IK request once (reused across the best-of-N solves below).
        req = GetPositionIK.Request()
        ik_req = PositionIKRequest()
        ik_req.group_name = self.manipulator_group_name
        ik_req.ik_link_name = ik_link
        ik_req.avoid_collisions = False
        ik_req.timeout = Duration(sec=int(timeout_s), nanosec=int((timeout_s % 1) * 1e9))

        ps = PoseStamped()
        ps.header.frame_id = frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(v) for v in position)
        (ps.pose.orientation.x, ps.pose.orientation.y,
         ps.pose.orientation.z, ps.pose.orientation.w) = (float(v) for v in orientation)
        ik_req.pose_stamped = ps

        if seed_joints:
            rs = RobotState()
            js = JointState()
            js.name = list(ARM_JOINT_NAMES)
            js.position = [float(v) for v in seed_joints]
            rs.joint_state = js
            ik_req.robot_state = rs

        req.ik_request = ik_req

        # The IK solver uses random restarts, so a single call can return a
        # valid-but-far branch — the arm would swing across the workspace and a
        # held part would fly with it (seen as up to ~4.5 rad on one joint
        # between steps). Solve a few times and KEEP THE SOLUTION CLOSEST to the
        # seed (the arm's current configuration), so the move is the smallest
        # one that still reaches the target. Stop early once a small move shows
        # up.
        best = None
        best_dist = None
        for _ in range(6):
            sol = self._call_ik_once(req)
            if sol is None:
                continue
            if seed_joints:
                dist = sum(abs(sol[i] - float(seed_joints[i]))
                           for i in range(len(ARM_JOINT_NAMES)))
            else:
                dist = 0.0
            if best is None or dist < best_dist:
                best, best_dist = sol, dist
            if not seed_joints or best_dist < 0.6:
                break
        if best is None:
            self.node.get_logger().warn("compute_ik: no solution after retries")
        return best

    def _call_ik_once(self, req):
        """One /compute_ik call. Returns a 7-joint list [joint_1..joint_7], or
        None on timeout/failure. Split out so compute_ik can solve several times
        and pick the solution nearest the seed (random restarts make repeated
        calls return different branches)."""
        # Block on the future via a timed poll (NOT spin_once — see the
        # nested-spin note in plan_execute_arm). The IK client shares the action
        # ReentrantCallbackGroup, so the background MultiThreadedExecutor
        # completes this future on another thread while we wait here.
        future = self.ik_client.call_async(req)
        start = time.time()
        while not future.done():
            if time.time() - start > 6.0:
                self.node.get_logger().warn("compute_ik: timed out waiting for IK result")
                return None
            time.sleep(0.01)
        res = future.result()
        if res is None or res.error_code.val != 1:
            return None
        name_to_pos = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        return [name_to_pos.get(j, 0.0) for j in ARM_JOINT_NAMES]

    def fk(self, joints, ik_link="end_effector_link", frame="world"):
        """Forward kinematics: world pose of `ik_link` at `joints`, via
        /compute_fk. Returns (x, y, z) or None. The arm reliably reaches its
        commanded joints, so FK of those joints is where the gripper ACTUALLY
        ends up — release uses this to drop the part exactly under the gripper
        instead of at a coordinate the gripper may not have reached."""
        if self.fk_client is None:
            return None
        if (not self.fk_client.service_is_ready()
                and not self.fk_client.wait_for_service(timeout_sec=1.0)):
            return None
        req = GetPositionFK.Request()
        req.header = Header(frame_id=frame)
        req.fk_link_names = [ik_link]
        js = JointState()
        js.name = list(ARM_JOINT_NAMES)
        js.position = [float(v) for v in joints]
        req.robot_state = RobotState(joint_state=js)
        future = self.fk_client.call_async(req)
        start = time.time()
        while not future.done():
            if time.time() - start > 3.0:
                return None
            time.sleep(0.01)
        res = future.result()
        if res is None or not res.pose_stamped:
            return None
        p = res.pose_stamped[0].pose.position
        return (p.x, p.y, p.z)

    def get_object_pose(self, object_id):
        """Look up a world collision object's pose + box dimensions from the
        live planning scene.

        Returns ((x, y, z), [dim_x, dim_y, dim_z]) for the object's reference
        pose, or None if the scene is unavailable or the object/geometry isn't
        found. This is the ground truth for where an object actually is in
        RViz, so approach poses computed from it stay aligned with what's
        rendered instead of a hardcoded hint.
        """
        if self.scene_query_client is None:
            self.node.get_logger().warn("get_object_pose: no scene query client injected")
            return None
        if (not self.scene_query_client.service_is_ready()
                and not self.scene_query_client.wait_for_service(timeout_sec=3.0)):
            self.node.get_logger().warn("get_object_pose: /get_planning_scene unavailable")
            return None

        # The live planning scene occasionally answers a query for a
        # just-added/just-modified object with its NAME present but its
        # geometry (primitives) not yet populated — so dims comes back None.
        # That None then cascades through execute_grasp: the oversize check is
        # skipped and attach_object falls back to a placeholder box size, so
        # the object visibly SHRINKS the instant it's picked up (and the
        # approach can't be IK-solved, so the arm doesn't track the object).
        # Retry a few times until the object comes back WITH geometry; keep the
        # best pose-only hit as a fallback if geometry never shows up.
        best = None
        for _ in range(4):
            res = self._query_planning_scene()
            if res is not None:
                for co in res.scene.world.collision_objects:
                    if co.id == object_id:
                        p = co.pose.position
                        dims = list(co.primitives[0].dimensions) if co.primitives else None
                        result = ((p.x, p.y, p.z), dims)
                        if dims and len(dims) >= 3:
                            return result          # complete — pose + geometry
                        best = result              # found, geometry not ready yet
                        break
            time.sleep(0.15)
        if best is None:
            self.node.get_logger().warn(
                f"get_object_pose: '{object_id}' not found in live scene after retries")
        else:
            self.node.get_logger().warn(
                f"get_object_pose: '{object_id}' found but geometry unavailable after retries")
        return best

    def _query_planning_scene(self):
        """One /get_planning_scene call for world object names + geometry.
        Returns the response, or None on timeout/failure. Split out of
        get_object_pose so the latter can retry cleanly."""
        req = GetPlanningScene.Request()
        req.components.components = (
            PlanningSceneComponents.WORLD_OBJECT_GEOMETRY |
            PlanningSceneComponents.WORLD_OBJECT_NAMES
        )
        future = self.scene_query_client.call_async(req)
        start = time.time()
        while not future.done():
            if time.time() - start > 5.0:
                self.node.get_logger().warn("get_object_pose: timed out querying scene")
                return None
            time.sleep(0.01)
        return future.result()

    def _wait_for_planning_result(self, move_goal):
        """Send goal and wait for MoveGroup planning result"""
        self.node.get_logger().info("Sending goal to MoveGroup...")
        send_goal_future = self._move_group_action_client.send_goal_async(move_goal)

        # Wait for goal acceptance
        goal_event = threading.Event()
        goal_data = {"handle": None, "accepted": False}

        def on_goal_response(future):
            try:
                goal_data["handle"] = future.result()
                goal_data["accepted"] = goal_data["handle"].accepted
            except Exception as e:
                self.node.get_logger().error(f"Goal response error: {e}")
            finally:
                goal_event.set()

        send_goal_future.add_done_callback(on_goal_response)

        # Wait with timeout
        if not self._wait_for_event(goal_event, timeout=20.0, task="goal acceptance"):
            return False

        if not goal_data["accepted"]:
            self.node.get_logger().error("Goal was rejected!")
            return False

        goal_handle = goal_data["handle"]
        self.node.get_logger().info("Goal accepted, waiting for planning result...")

        # Wait for planning result
        result_event = threading.Event()
        result_data = {"result": None, "success": False}

        def on_result(future):
            try:
                res = future.result().result
                result_data["result"] = res
                result_data["success"] = (res.error_code.val == 1)
            except Exception as e:
                self.node.get_logger().error(f"Result error: {e}")
            finally:
                result_event.set()

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(on_result)

        if not self._wait_for_event(result_event, timeout=20.0, task="planning result"):
            return False

        if not result_data["success"]:
            self.node.get_logger().error(f"Motion failed with error code: {result_data['result'].error_code.val}")
            return False

        # Execute the planned trajectory
        return self._execute_trajectory(result_data["result"])

    def _execute_trajectory(self, result):
        """Execute planned trajectory on controller"""
        if result.error_code.val == 1:  # SUCCESS
            self.node.get_logger().info(f"[OK] Planning successful! Executing trajectory on controller...")

            trajectory = result.planned_trajectory.joint_trajectory
            execute_goal = FollowJointTrajectory.Goal()
            execute_goal.trajectory = trajectory

            duration = self._extract_trajectory_duration(result)
            self.node.get_logger().info(f"Sending trajectory to controller (duration: {duration:.2f}s)...")

            # Send goal to controller
            execute_future = self._manipulator_controller_client.send_goal_async(execute_goal)

            # Wait for goal acceptance
            exec_start_time = time.time()
            while not execute_future.done():
                if time.time() - exec_start_time > 5.0:
                    self.node.get_logger().error("Timeout waiting for controller to accept trajectory!")
                    return False
                time.sleep(0.001)

            exec_goal_handle = execute_future.result()
            if not exec_goal_handle.accepted:
                self.node.get_logger().error("Controller rejected trajectory!")
                return False

            self.node.get_logger().info("[OK] Controller accepted trajectory, executing...")

            # Wait for execution to complete
            exec_result_future = exec_goal_handle.get_result_async()
            exec_start = time.time()
            timeout = duration + 10.0

            while not exec_result_future.done():
                if time.time() - exec_start > timeout:
                    self.node.get_logger().error(f"Timeout waiting for execution to complete!")
                    return False
                time.sleep(0.01)

            exec_result = exec_result_future.result().result
            if exec_result.error_code == 0:  # SUCCESS
                self.node.get_logger().info(f"[OK] Trajectory executed successfully in {time.time() - exec_start:.2f}s!")
                return True
            else:
                self.node.get_logger().error(f"Execution failed with error code: {exec_result.error_code}")
                return False
        else:
            self.node.get_logger().error(f"Motion failed with error code: {result.error_code.val}")
            return False

    def _wait_for_event(self, event, timeout, task):
        """Wait for event with timeout and polling"""
        start_time = time.time()
        while not event.is_set():
            elapsed = time.time() - start_time
            if elapsed > timeout:
                self.node.get_logger().error(f"Timeout waiting for {task}!")
                return False
            time.sleep(0.001)
        return True

    def _extract_trajectory_duration(self, result):
        """Extract the total duration from MoveIt2 trajectory"""
        try:
            if result.planned_trajectory and result.planned_trajectory.joint_trajectory:
                traj = result.planned_trajectory.joint_trajectory
                if traj.points and len(traj.points) > 0:
                    last_point = traj.points[-1]
                    duration = last_point.time_from_start.sec + last_point.time_from_start.nanosec / 1e9
                    scaled_duration = duration * VISUALIZATION_SLOW_FACTOR
                    return max(scaled_duration, 2.0)
        except Exception as e:
            self.node.get_logger().warn(f"Could not extract trajectory duration: {e}")
        return 8.0  # Default duration

    def _execute_direct_trajectory(self, target_joints, where):
        """
        Method 6: Direct trajectory execution (GitHub method)
        Build trajectory directly and execute without MoveIt planning
        This bypasses collision checking and state validation
        """
        self.node.get_logger().info(f"Building direct trajectory to: {where}")

        # Build JointTrajectory message
        trajectory = JointTrajectory()
        trajectory.joint_names = [
            "joint_1", "joint_2", "joint_3", "joint_4",
            "joint_5", "joint_6", "joint_7"
        ]

        # Create trajectory point for target position
        point = JointTrajectoryPoint()
        point.positions = list(target_joints)
        point.velocities = [0.0] * 7
        point.accelerations = [0.0] * 7

        # Set time to reach the target, SCALED BY HOW FAR THE ARM TRAVELS.
        # A fixed duration made a big joint reconfiguration snap across at high
        # speed while a tiny nudge crawled — inconsistent speed reads as jerky.
        # Timing the move by its largest joint delta keeps the apparent velocity
        # roughly constant across steps, so the motion looks smooth. Falls back
        # to the base time when the current pose isn't known yet.
        base = 2.0 / self.vel_scale
        cur = self._latest_arm_joints
        if cur is not None and len(cur) == len(target_joints):
            max_delta = max(abs(float(target_joints[i]) - float(cur[i]))
                            for i in range(len(target_joints)))
            duration_sec = max(1.5, min(4.0, 1.2 + 1.6 * max_delta))
        else:
            duration_sec = base
        point.time_from_start = Duration(sec=int(duration_sec), nanosec=int((duration_sec % 1) * 1e9))

        trajectory.points.append(point)

        # Create FollowJointTrajectory goal
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory

        self.node.get_logger().info(f"Sending direct trajectory to controller (duration: {duration_sec:.2f}s)...")

        # Send goal to controller
        send_future = self._manipulator_controller_client.send_goal_async(goal)

        # Wait for goal acceptance
        start_time = time.time()
        while not send_future.done():
            if time.time() - start_time > 5.0:
                self.node.get_logger().error("[FAIL] Timeout waiting for controller to accept trajectory!")
                return False
            time.sleep(0.01)

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.node.get_logger().error("[FAIL] Controller rejected direct trajectory!")
            return False

        self.node.get_logger().info("[OK] Controller accepted direct trajectory, executing...")

        # Wait for execution to complete
        result_future = goal_handle.get_result_async()
        exec_start = time.time()
        timeout = duration_sec + 5.0

        while not result_future.done():
            if time.time() - exec_start > timeout:
                self.node.get_logger().error(f"[FAIL] Timeout waiting for direct execution to complete!")
                return False
            time.sleep(0.01)

        result = result_future.result().result
        if result.error_code == 0:  # SUCCESS
            self.node.get_logger().info(f"[OK] Direct trajectory executed successfully in {time.time() - exec_start:.2f}s!")
            self.node.get_logger().info(f"Reached target: {where}")
            return True
        else:
            self.node.get_logger().error(f"[FAIL] Direct execution failed with error code: {result.error_code}")
            return False

    def plan_execute_gripper(self, joints, where):
        """Actuate the gripper to a target opening via the fake gripper controller.

        `joints[0]` drives robotiq_85_left_knuckle_joint (the single actuated
        joint; the other finger joints mimic it in the URDF), e.g. OPEN=0.1,
        CLOSE=0.8 from config/waypoints.json. Previously this was a no-op sleep,
        so the fingers never visibly moved; now they actually open/close.
        """
        self.node.get_logger().info(f"Executing gripper motion for: {where}")
        if not joints:
            self.node.get_logger().warn("gripper: no target position given")
            return False
        target = float(joints[0])

        if self._gripper_controller_client is None:
            self.node.get_logger().warn("gripper: no controller client; skipping")
            time.sleep(1.0)
            return True

        traj = JointTrajectory()
        traj.joint_names = ["robotiq_85_left_knuckle_joint"]
        point = JointTrajectoryPoint()
        point.positions = [target]
        point.time_from_start = Duration(sec=1, nanosec=0)
        traj.points.append(point)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        send_future = self._gripper_controller_client.send_goal_async(goal)
        start = time.time()
        while not send_future.done():
            if time.time() - start > 5.0:
                self.node.get_logger().warn("gripper: controller did not accept goal in time")
                return False
            time.sleep(0.01)
        handle = send_future.result()
        if not handle.accepted:
            self.node.get_logger().warn("gripper: goal rejected by controller")
            return False

        result_future = handle.get_result_async()
        start = time.time()
        while not result_future.done():
            if time.time() - start > 4.0:
                break  # don't fail the skill on a slow gripper result
            time.sleep(0.01)
        self.node.get_logger().info(f"[OK] Gripper '{where}' -> {target:.2f}")
        return True
