#!/usr/bin/python3
"""Main skill server node for robot control"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionClient
from std_msgs.msg import String
from sensor_msgs.msg import JointState
import json
import time
import threading

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import DisplayTrajectory, PlanningScene, AllowedCollisionMatrix, AllowedCollisionEntry, PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from control_msgs.action import FollowJointTrajectory

# MoveItPy for planning scene manipulation
try:
    from moveit.planning import MoveItPy
    MOVEITPY_AVAILABLE = True
except ImportError:
    MOVEITPY_AVAILABLE = False

from .session import Session
from .motion_executor import MotionExecutor
from .scene_manager import SceneManager
from .skill_handlers import SkillHandlers

EXECUTION_LOCK = threading.Lock()


class SkillServer(Node):
    """Main skill server node"""

    def __init__(self):
        super().__init__("skill_server")
        self._setup_parameters()
        self._init_state()

    def _setup_parameters(self):
        """Declare and read ROS parameters"""
        self.declare_parameter("waypoints_path", "")
        self.declare_parameter("manipulator_group", "manipulator")
        self.declare_parameter("gripper_group", "gripper")
        self.declare_parameter("ee_attach_link", "robotiq_85_base_link")
        self.declare_parameter("vel_scale", 0.8)
        self.declare_parameter("acc_scale", 0.8)
        self.declare_parameter("scene_update_wait_s", 1.0)
        self.declare_parameter("open_gripper_pose_name", "OPEN")
        self.declare_parameter("close_gripper_pose_name", "CLOSE")

        self.waypoints_path = self.get_parameter("waypoints_path").get_parameter_value().string_value
        self.manipulator_group_name = self.get_parameter("manipulator_group").get_parameter_value().string_value
        self.gripper_group_name = self.get_parameter("gripper_group").get_parameter_value().string_value
        self.ee_attach_link = self.get_parameter("ee_attach_link").get_parameter_value().string_value
        self.vel_scale = self.get_parameter("vel_scale").get_parameter_value().double_value
        self.acc_scale = self.get_parameter("acc_scale").get_parameter_value().double_value
        self.open_gripper_pose = self.get_parameter("open_gripper_pose_name").get_parameter_value().string_value
        self.close_gripper_pose = self.get_parameter("close_gripper_pose_name").get_parameter_value().string_value

    def _init_state(self):
        """Initialize internal state"""
        self.waypoints_json_ = {}
        self.session_ = Session()
        self.last_joint_state_ = None
        self.motion_executor = None
        self.scene_manager = None
        self.skill_handlers = None
        self.moveit_py = None  # MoveItPy instance for planning scene access

    def init_ros(self):
        """Initialize ROS2 clients and load waypoints"""
        if not self.waypoints_path:
            self.get_logger().error("Parameter 'waypoints_path' is required!")
            raise RuntimeError("Missing waypoints_path parameter")

        self._init_action_clients()
        self._load_waypoints(self.waypoints_path)
        self._init_modules()
        self._init_topics()
        # _setup_collision_matrix_fallback() was rewritten to MERGE into the
        # current ACM (fetch via GetPlanningScene, edit just the needed
        # entries, re-apply the full matrix) instead of replacing it with a
        # partial 3-entry one — the old replace-based version was wiping out
        # all 132 of the SRDF's own disable_collisions pairs and was the
        # primary cause of "Skipping invalid start state" -> "Catastrophic
        # failure" on every MoveGroup planning request. Safe to call now;
        # verified via /check_state_validity (0 contacts at HOME) and a full
        # moveTo HOME round-trip (planned, executed, success feedback).
        self._setup_collision_matrix()

        self.get_logger().info(f"✅ Skill Server Ready! Listening on /llm_commands")

    def _init_action_clients(self):
        """Initialize action clients"""
        self.get_logger().info("Waiting for MoveGroup action server...")
        self._action_callback_group = ReentrantCallbackGroup()

        self._move_group_action_client = ActionClient(
            self, MoveGroup, "/move_action",
            callback_group=self._action_callback_group
        )
        if not self._move_group_action_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("MoveGroup action server not available!")
        self.get_logger().info("✅ MoveGroup action server is available.")

        self._manipulator_controller_client = ActionClient(
            self, FollowJointTrajectory,
            "/fake_manipulator_controller/follow_joint_trajectory",
            callback_group=self._action_callback_group
        )
        self._gripper_controller_client = ActionClient(
            self, FollowJointTrajectory,
            "/fake_gripper_controller/follow_joint_trajectory",
            callback_group=self._action_callback_group
        )
        self.get_logger().info("✅ Controller action clients created.")

        self._planning_scene_client = self.create_client(
            ApplyPlanningScene, '/apply_planning_scene'
        )
        self.get_logger().info("✅ Planning scene client created.")

    def _init_modules(self):
        """Initialize executor modules"""
        self.motion_executor = MotionExecutor(
            self,
            self._move_group_action_client,
            self._manipulator_controller_client,
            self._gripper_controller_client
        )
        self.motion_executor.manipulator_group_name = self.manipulator_group_name
        self.motion_executor.vel_scale = self.vel_scale
        self.motion_executor.acc_scale = self.acc_scale

        self.scene_manager = SceneManager(self, self._planning_scene_client)
        self.skill_handlers = SkillHandlers(
            self, self.waypoints_json_, self.motion_executor, self.scene_manager
        )
        self.skill_handlers.open_gripper_pose = self.open_gripper_pose
        self.skill_handlers.close_gripper_pose = self.close_gripper_pose

    def _init_topics(self):
        """Initialize publishers and subscribers"""
        self.command_sub_ = self.create_subscription(
            String, "/llm_commands", self.command_callback, 10
        )
        self.feedback_pub_ = self.create_publisher(String, "/llm_feedback", 10)
        self.joint_state_sub_ = self.create_subscription(
            JointState, "/joint_states", self._joint_state_callback, 10
        )
        self.display_trajectory_sub_ = self.create_subscription(
            DisplayTrajectory, "/display_planned_path",
            self._display_trajectory_callback, 10
        )

    def _setup_collision_matrix(self):
        """Disable self-collision between gripper base and wrist links using MoveItPy (Method 7)"""
        self.get_logger().info("🔧 Method 7: Setting up ACM using MoveItPy planning scene read_write...")

        if not MOVEITPY_AVAILABLE:
            self.get_logger().warn("⚠️ MoveItPy not available, falling back to service-based method...")
            self._setup_collision_matrix_fallback()
            return

        try:
            # Initialize MoveItPy
            self.get_logger().info("Initializing MoveItPy...")
            self.moveit_py = MoveItPy(node_name="moveit_py_planning_scene")

            # Get planning scene monitor
            planning_scene_monitor = self.moveit_py.get_planning_scene_monitor()

            # Use read_write context to modify ACM
            self.get_logger().info("Acquiring planning scene write lock...")
            with planning_scene_monitor.read_write() as scene:
                self.get_logger().info("Modifying AllowedCollisionMatrix...")

                # Get the ACM
                acm = scene.allowed_collision_matrix

                # Set entries to allow collisions between gripper base and wrist links
                acm.set_entry("robotiq_85_base_link", "spherical_wrist_1_link", True)
                acm.set_entry("robotiq_85_base_link", "spherical_wrist_2_link", True)

                # Update the scene state
                scene.current_state.update()

                self.get_logger().info("✅ ACM modified: allowed collisions between:")
                self.get_logger().info("   - robotiq_85_base_link <-> spherical_wrist_1_link")
                self.get_logger().info("   - robotiq_85_base_link <-> spherical_wrist_2_link")

            self.get_logger().info("✅ Method 7 ACM setup completed successfully!")

        except Exception as e:
            self.get_logger().error(f"❌ Method 7 failed: {e}")
            self.get_logger().warn("Falling back to service-based method...")
            self._setup_collision_matrix_fallback()

    def _setup_collision_matrix_fallback(self):
        """Fallback: Use ApplyPlanningScene service to modify ACM.

        FIXED (was the root cause of "Skipping invalid start state" on every
        MoveGroup planning request): the previous version sent a partial
        3-entry AllowedCollisionMatrix as part of a diff PlanningScene.
        MoveIt treats the `allowed_collision_matrix` field in a diff scene as
        a full REPLACEMENT of the current ACM, not a per-entry merge — so
        this was silently wiping out all 132 of the SRDF's own correctly
        -declared disable_collisions pairs (config/moveit/gen3_robotiq_2f_85
        .srdf), leaving only the 3 entries listed here. Confirmed via
        /check_state_validity: the robot's own HOME pose was reported
        invalid with 16 self-collision contacts that the SRDF explicitly
        disables, and the count dropped to 0 once this was fixed to merge
        (fetch the full current ACM via GetPlanningScene, edit just the
        entries we need, then re-apply the FULL matrix) instead of replace.
        """
        self.get_logger().info("Using ApplyPlanningScene service fallback (merge, not replace)...")

        get_scene_client = self.create_client(GetPlanningScene, '/get_planning_scene')
        apply_scene_client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        if not get_scene_client.wait_for_service(timeout_sec=5.0) or \
           not apply_scene_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("Planning scene services not available, skipping ACM setup")
            return

        import time

        def call_sync(client, request, timeout=5.0):
            future = client.call_async(request)
            start = time.time()
            while not future.done():
                if time.time() - start > timeout:
                    return None
                time.sleep(0.01)
            return future.result()

        # 1) fetch the CURRENT full ACM (do not start from an empty one)
        get_req = GetPlanningScene.Request()
        get_req.components = PlanningSceneComponents()
        get_req.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        scene_resp = call_sync(get_scene_client, get_req)
        if scene_resp is None:
            self.get_logger().warn("⏱️ Timeout fetching current ACM, skipping ACM setup")
            return
        acm = scene_resp.scene.allowed_collision_matrix

        # 2) add/overwrite just the pairs we need, preserving every existing entry
        pairs_to_disable = [
            ("robotiq_85_base_link", "spherical_wrist_1_link"),
            ("robotiq_85_base_link", "spherical_wrist_2_link"),
        ]
        names = list(acm.entry_names)
        for link in {l for pair in pairs_to_disable for l in pair} - set(names):
            names.append(link)
            for entry in acm.entry_values:
                entry.enabled.append(False)
            acm.entry_values.append(AllowedCollisionEntry(enabled=[False] * len(names)))
        acm.entry_names = names
        for a, b in pairs_to_disable:
            i, j = names.index(a), names.index(b)
            acm.entry_values[i].enabled[j] = True
            acm.entry_values[j].enabled[i] = True

        # 3) re-apply the FULL matrix (never a partial one — see docstring)
        scene = PlanningScene()
        scene.is_diff = True
        scene.allowed_collision_matrix = acm
        request = ApplyPlanningScene.Request()
        request.scene = scene
        result = call_sync(apply_scene_client, request, timeout=2.0)

        if result is None:
            self.get_logger().warn("⏱️ Timeout waiting for ACM update")
        elif result.success:
            self.get_logger().info("✅ ACM updated (merged): disabled gripper-wrist collisions "
                                    f"without touching the other {len(names) - 2} entries")
        else:
            self.get_logger().warn("❌ Failed to update ACM via fallback method")

    def _load_waypoints(self, path):
        """Load waypoints from JSON file"""
        self.get_logger().info(f"Loading waypoints from: {path}")
        try:
            with open(path, 'r') as f:
                self.waypoints_json_ = json.load(f)
            if "poses" not in self.waypoints_json_:
                raise RuntimeError("Waypoints JSON missing 'poses' object")
            self.get_logger().info("✅ Waypoints loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load waypoints: {e}")

    def _joint_state_callback(self, msg):
        """Store latest joint state"""
        self.last_joint_state_ = msg

    def _display_trajectory_callback(self, msg):
        """Log RViz planned trajectories"""
        try:
            if not msg.trajectory or len(msg.trajectory) == 0:
                return
            traj = msg.trajectory[0].joint_trajectory
            if not traj.points or len(traj.points) == 0:
                return

            final_point = traj.points[-1]
            target_positions = list(final_point.positions[:7])
            duration = final_point.time_from_start.sec + final_point.time_from_start.nanosec / 1e9

            self.get_logger().info(f"🎯 RViz planning detected! Duration: {duration:.2f}s")
        except Exception as e:
            self.get_logger().error(f"Error processing RViz trajectory: {e}")

    def publish_feedback(self, status, message, command_id="", stage="", code=0):
        """Publish feedback to LLM"""
        feedback = {
            "schema": "llm_fb/v1",
            "status": status,
            "message": message,
            "stage": stage,
            "code": code,
            "timestamp_ns": self.get_clock().now().nanoseconds,
        }
        if command_id:
            feedback["command_id"] = command_id

        msg = String()
        msg.data = json.dumps(feedback)
        self.feedback_pub_.publish(msg)

        feedback_msg = f"📤 Feedback({stage}/{status}): {message}"
        if status in ["failure", "rejected"]:
            self.get_logger().error(feedback_msg)
        else:
            self.get_logger().info(feedback_msg)

    def command_callback(self, msg):
        """Handle incoming skill commands"""
        if not EXECUTION_LOCK.acquire(blocking=False):
            self.get_logger().warn("Rejecting command: another skill in progress.")
            self.publish_feedback("rejected", "Skill execution in progress.")
            return

        try:
            self.get_logger().info(f"📨 Received command: {msg.data}")
            cmd = json.loads(msg.data)
            command_id = cmd.get("command_id", "")

            if cmd.get("schema") != "llm_cmd/v1":
                self.publish_feedback("rejected", "Unsupported schema", command_id)
                return

            skill = cmd.get("skill")
            if not skill:
                self.publish_feedback("rejected", "Missing 'skill'", command_id)
                return

            self._update_session_context(cmd)
            ok = self.dispatch_skill(skill, cmd, command_id)

            self.publish_feedback(
                "success" if ok else "failure",
                f"Skill '{skill}' {'completed' if ok else 'failed'}",
                command_id
            )
        except Exception as e:
            import traceback
            self.get_logger().error(f"FATAL ERROR: {e}")
            self.get_logger().error(f"Traceback: {traceback.format_exc()}")
            self.publish_feedback("failure", f"Unexpected error: {e}")
        finally:
            EXECUTION_LOCK.release()

    def _update_session_context(self, cmd):
        """Update session context from command"""
        if "context" in cmd and isinstance(cmd["context"], dict):
            ctx = cmd["context"]
            if "object" in ctx:
                self.session_.object = ctx["object"]
            if "plan_only" in ctx:
                self.session_.plan_only = ctx["plan_only"]
            if "vel_scale" in ctx:
                self.session_.vel_scale = ctx["vel_scale"]
            if "acc_scale" in ctx:
                self.session_.acc_scale = ctx["acc_scale"]
            if "place" in ctx and isinstance(ctx["place"], dict):
                self.session_.place = ctx["place"]

    def dispatch_skill(self, skill, cmd, command_id):
        """Dispatch skill to appropriate handler"""
        try:
            # Atomic skills
            if skill == "moveTo":
                return self._handle_move_to(cmd, command_id)
            elif skill == "selectObject":
                return self._handle_select_object(cmd, command_id)
            elif skill == "selectPlace":
                return self._handle_select_place(cmd, command_id)
            elif skill == "openGripper":
                return self._handle_open_gripper(cmd)
            elif skill == "closeGripper":
                return self._handle_close_gripper(cmd)
            elif skill == "approach":
                return self._handle_approach(cmd, command_id)
            elif skill == "place":
                return self._handle_place(cmd, command_id)
            elif skill == "retreat":
                return self._handle_retreat(cmd, command_id)
            # High-level skills
            elif skill == "grasp":
                return self._handle_grasp(cmd, command_id)
            elif skill == "release":
                return self._handle_release(cmd, command_id)
            elif skill == "dismantle":
                return self._handle_dismantle(cmd, command_id)
            elif skill == "sequence":
                return self._handle_sequence(cmd, command_id)
            # Skills the LLM side (src/llm_agent/config/skills.json) was
            # already declaring but this dispatcher had no handler for —
            # any plan using these used to be REJECTED here as "Unknown
            # skill" even though the LLM/evaluation pipeline treated them
            # as valid. See REVISION_MEMO.md for the audit that found this.
            elif skill == "inspect":
                return self._handle_inspect(cmd, command_id)
            elif skill == "waitForStabilization":
                return self._handle_wait_for_stabilization(cmd, command_id)
            elif skill == "rotateGripper":
                return self._handle_rotate_gripper(cmd, command_id)
            elif skill == "unscrew":
                return self._handle_unscrew(cmd, command_id)
            elif skill == "disconnect":
                return self._handle_disconnect(cmd, command_id)
            else:
                self.publish_feedback("rejected", f"Unknown skill: {skill}", command_id, "dispatch")
                return False

        except Exception as e:
            self.get_logger().error(f"Exception in dispatch_skill ({skill}): {e}", exc_info=True)
            self.publish_feedback("failure", f"Exception: {e}", command_id, skill)
            return False

    def _get_pose_joints(self, name):
        """Get joint positions for a named pose"""
        return self.waypoints_json_.get("poses", {}).get(name)

    def _handle_move_to(self, cmd, command_id):
        """Handle moveTo skill"""
        target = cmd.get("target")
        if not target:
            self.publish_feedback("rejected", "moveTo missing 'target'", command_id, "moveTo")
            return False
        joints = self._get_pose_joints(target)
        if not joints:
            return False
        ok = self.motion_executor.plan_execute_arm(joints, "moveTo")
        if ok:
            self.session_.last_arm_joints = joints
        return ok

    def _handle_select_object(self, cmd, command_id):
        """Handle selectObject skill"""
        obj = cmd.get("params", {}).get("object_id")
        if not obj:
            self.publish_feedback("rejected", "selectObject requires params.object_id", command_id, "selectObject")
            return False
        self.session_.object = obj
        self.publish_feedback("progress", f"current object = {obj}", command_id, "selectObject")
        return True

    def _handle_select_place(self, cmd, command_id):
        """Handle selectPlace skill"""
        place_params = cmd.get("params", {})
        if not any(k in place_params for k in ["bin", "pose_name", "joints"]):
            self.publish_feedback("rejected", "selectPlace requires bin/pose_name/joints", command_id, "selectPlace")
            return False
        self.session_.place = place_params
        self.publish_feedback("progress", "place selected", command_id, "selectPlace")
        return True

    def _handle_open_gripper(self, cmd):
        """Handle openGripper skill"""
        pose_name = cmd.get("params", {}).get("pose_name", self.open_gripper_pose)
        joints = self._get_pose_joints(pose_name)
        if not joints:
            return False
        return self.motion_executor.plan_execute_gripper(joints, "openGripper")

    def _handle_close_gripper(self, cmd):
        """Handle closeGripper skill"""
        pose_name = cmd.get("params", {}).get("pose_name", self.close_gripper_pose)
        joints = self._get_pose_joints(pose_name)
        if not joints:
            return False
        return self.motion_executor.plan_execute_gripper(joints, "closeGripper")

    def _handle_approach(self, cmd, command_id):
        """Handle approach skill"""
        obj = self._extract_object_from_cmd(cmd) or self.session_.object
        if obj not in self.waypoints_json_.get("objects", {}):
            self.publish_feedback("rejected", f"approach: unknown object '{obj}'", command_id, "approach")
            return False
        joints = self.waypoints_json_["objects"][obj].get("approach")
        if not joints:
            return False
        ok = self.motion_executor.plan_execute_arm(joints, "approach")
        if ok:
            self.session_.last_arm_joints = joints
        return ok

    def _handle_place(self, cmd, command_id):
        """Handle place skill"""
        obj = self._extract_object_from_cmd(cmd) or self.session_.object
        local_place = cmd.get("params", {}) or self.session_.place
        target_joints = self.skill_handlers._resolve_place_joints(obj, local_place)
        if not target_joints:
            self.publish_feedback("failure", "place: cannot resolve joints", command_id, "place")
            return False
        return self.motion_executor.plan_execute_arm(target_joints, "place")

    def _handle_retreat(self, cmd, command_id):
        """Handle retreat skill"""
        obj = self._extract_object_from_cmd(cmd) or self.session_.object
        if obj not in self.waypoints_json_.get("objects", {}):
            self.publish_feedback("rejected", f"retreat: unknown object '{obj}'", command_id, "retreat")
            return False
        joints = self.waypoints_json_["objects"][obj].get("retreat")
        if not joints:
            return False
        return self.motion_executor.plan_execute_arm(joints, "retreat")

    def _handle_grasp(self, cmd, command_id):
        """Handle grasp skill"""
        obj = self._extract_object_from_cmd(cmd) or self.session_.object
        if not obj:
            self.publish_feedback("rejected", "grasp needs an object", command_id, "grasp")
            return False
        return self.skill_handlers.execute_grasp(obj)

    def _handle_release(self, cmd, command_id):
        """Handle release skill"""
        obj = self._extract_object_from_cmd(cmd) or self.session_.object
        if not obj:
            self.publish_feedback("rejected", "release needs an object", command_id, "release")
            return False
        place_joints = self.skill_handlers._resolve_place_joints(
            obj, cmd.get("place", {}) or self.session_.place
        )
        return self.skill_handlers.execute_release(obj, place_joints)

    def _handle_inspect(self, cmd, command_id):
        """Handle inspect skill.

        No camera/perception pipeline exists (see RQ4 in eval/ — perception
        is currently a geometric simulation, not a real sensor). This moves
        the arm to the object's approach pose, where a wrist-mounted camera
        WOULD see it if one existed, and reports success. It performs no
        actual visual check.
        """
        obj = self._extract_object_from_cmd(cmd) or self.session_.object
        if obj not in self.waypoints_json_.get("objects", {}):
            self.publish_feedback("rejected", f"inspect: unknown object '{obj}'", command_id, "inspect")
            return False
        joints = self.waypoints_json_["objects"][obj].get("approach")
        if not joints:
            return False
        ok = self.motion_executor.plan_execute_arm(joints, "inspect")
        if ok:
            self.session_.last_arm_joints = joints
            self.publish_feedback("progress", "inspect: positioned for visual check "
                                   "(no perception pipeline wired up — see RQ4 caveat)",
                                   command_id, "inspect")
        return ok

    def _handle_wait_for_stabilization(self, cmd, command_id):
        """Handle waitForStabilization skill: settle time after a motion, no movement."""
        seconds = cmd.get("params", {}).get("seconds", 1.0)
        try:
            seconds = max(0.0, min(10.0, float(seconds)))  # clamp: no runaway sleeps
        except (TypeError, ValueError):
            seconds = 1.0
        time.sleep(seconds)
        return True

    def _handle_rotate_gripper(self, cmd, command_id):
        """Handle rotateGripper skill: rotate the wrist joint by params.angle (degrees).

        Rotates relative to the last successfully-commanded arm pose
        (session_.last_arm_joints), falling back to HOME if the arm hasn't
        moved yet this session.
        """
        angle = cmd.get("params", {}).get("angle", cmd.get("angle"))
        if angle is None:
            self.publish_feedback("rejected", "rotateGripper requires params.angle (degrees)",
                                   command_id, "rotateGripper")
            return False
        base = self.session_.last_arm_joints or self._get_pose_joints("HOME")
        if not base:
            self.publish_feedback("failure", "rotateGripper: no base pose available",
                                   command_id, "rotateGripper")
            return False
        joints = self.skill_handlers.rotated_joints(base, float(angle))
        ok = self.motion_executor.plan_execute_arm(joints, "rotateGripper")
        if ok:
            self.session_.last_arm_joints = joints
        return ok

    def _handle_unscrew(self, cmd, command_id):
        """Handle unscrew skill (scripted gripper-rotation cycles — see
        SkillHandlers.execute_unscrew docstring for the force-sensing caveat)."""
        obj = self._extract_object_from_cmd(cmd) or self.session_.object
        if not obj:
            self.publish_feedback("rejected", "unscrew needs an object", command_id, "unscrew")
            return False
        turns = cmd.get("params", {}).get("turns", 2)
        ok = self.skill_handlers.execute_unscrew(obj, turns=int(turns))
        if ok:
            self.session_.last_arm_joints = None  # sequence ends at retreat; force re-resolve next time
        return ok

    def _handle_disconnect(self, cmd, command_id):
        """Handle disconnect skill (grip-and-pull — see SkillHandlers.execute_disconnect
        docstring for the force-sensing caveat)."""
        obj = self._extract_object_from_cmd(cmd) or self.session_.object
        if not obj:
            self.publish_feedback("rejected", "disconnect needs an object", command_id, "disconnect")
            return False
        ok = self.skill_handlers.execute_disconnect(obj)
        if ok:
            self.session_.last_arm_joints = None
        return ok

    def _handle_dismantle(self, cmd, command_id):
        """Handle dismantle skill"""
        targets = cmd.get("targets", [])
        if not targets:
            self.publish_feedback("rejected", "dismantle: missing targets", command_id, "dismantle")
            return False
        place_default = cmd.get("place", {})
        return self.skill_handlers.execute_dismantle(targets, place_default or self.session_.place)

    def _handle_sequence(self, cmd, command_id):
        """Handle sequence skill"""
        steps = cmd.get("params", {}).get("steps", [])
        if not steps:
            self.publish_feedback("rejected", "sequence needs params.steps", command_id, "sequence")
            return False
        for step in steps:
            sub_skill = step.get("skill")
            if not sub_skill:
                continue
            if not self.dispatch_skill(sub_skill, step, command_id):
                return False
        return True

    def _extract_object_from_cmd(self, cmd):
        """Extract object name from command"""
        if "target" in cmd and isinstance(cmd["target"], str):
            return cmd["target"]
        if "params" in cmd and isinstance(cmd["params"], dict):
            # Check for 'object', 'object_id', or 'target' in params
            return (cmd["params"].get("object") or
                    cmd["params"].get("object_id") or
                    cmd["params"].get("target") or "")
        return ""


def main(args=None):
    rclpy.init(args=args)
    skill_server_node = SkillServer()

    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(skill_server_node)

    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    time.sleep(2.0)

    try:
        skill_server_node.init_ros()
        thread.join()
    except Exception as e:
        skill_server_node.get_logger().fatal(f"Failed to initialize SkillServer: {e}")
    finally:
        executor.shutdown()
        skill_server_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
