#!/usr/bin/python3
"""Motion planning and execution for robot manipulator"""

import time
import threading
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MotionPlanRequest,  RobotState
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
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
        self.use_direct_execution = False  # Method 6: Use direct trajectory execution (GitHub method)

    def plan_execute_arm(self, joints, where):
        """Plan and execute arm motion to target joint positions"""
        self.node.get_logger().info(f"🚀 Executing arm motion for: {where}")

        # Method 6: Use direct trajectory execution (bypass MoveIt planning)
        if self.use_direct_execution:
            self.node.get_logger().info("🔧 Method 6: Using direct trajectory execution (GitHub method)")
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
            self.node.get_logger().info(f"✅ Planning successful! Executing trajectory on controller...")

            trajectory = result.planned_trajectory.joint_trajectory
            execute_goal = FollowJointTrajectory.Goal()
            execute_goal.trajectory = trajectory

            duration = self._extract_trajectory_duration(result)
            self.node.get_logger().info(f"📤 Sending trajectory to controller (duration: {duration:.2f}s)...")

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

            self.node.get_logger().info("✅ Controller accepted trajectory, executing...")

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
                self.node.get_logger().info(f"✅ Trajectory executed successfully in {time.time() - exec_start:.2f}s!")
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
        self.node.get_logger().info(f"📦 Building direct trajectory to: {where}")

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

        # Set time to reach target (scaled by velocity factor)
        duration_sec = 3.0 / self.vel_scale  # Base duration scaled by velocity
        point.time_from_start = Duration(sec=int(duration_sec), nanosec=int((duration_sec % 1) * 1e9))

        trajectory.points.append(point)

        # Create FollowJointTrajectory goal
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory

        self.node.get_logger().info(f"🚀 Sending direct trajectory to controller (duration: {duration_sec:.2f}s)...")

        # Send goal to controller
        send_future = self._manipulator_controller_client.send_goal_async(goal)

        # Wait for goal acceptance
        start_time = time.time()
        while not send_future.done():
            if time.time() - start_time > 5.0:
                self.node.get_logger().error("❌ Timeout waiting for controller to accept trajectory!")
                return False
            time.sleep(0.01)

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.node.get_logger().error("❌ Controller rejected direct trajectory!")
            return False

        self.node.get_logger().info("✅ Controller accepted direct trajectory, executing...")

        # Wait for execution to complete
        result_future = goal_handle.get_result_async()
        exec_start = time.time()
        timeout = duration_sec + 5.0

        while not result_future.done():
            if time.time() - exec_start > timeout:
                self.node.get_logger().error(f"❌ Timeout waiting for direct execution to complete!")
                return False
            time.sleep(0.01)

        result = result_future.result().result
        if result.error_code == 0:  # SUCCESS
            self.node.get_logger().info(f"✅ Direct trajectory executed successfully in {time.time() - exec_start:.2f}s!")
            self.node.get_logger().info(f"🎯 Reached target: {where}")
            return True
        else:
            self.node.get_logger().error(f"❌ Direct execution failed with error code: {result.error_code}")
            return False

    def plan_execute_gripper(self, joints, where):
        """Execute gripper motion"""
        self.node.get_logger().info(f"🚀 Executing gripper motion for: {where}")
        self.node.get_logger().info(f"🤏 Simulating gripper motion to: {joints[0]:.3f}")
        time.sleep(1.5)
        self.node.get_logger().info(f"✅ Gripper motion '{where}' completed.")
        return True
