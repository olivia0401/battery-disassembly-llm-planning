#!/usr/bin/python3
"""Skill execution handlers for robot tasks"""

import math

# joint_7 is the wrist-roll joint on the Kinova Gen3 (confirmed by HOME pose
# in config/waypoints.json: index 6 = 1.5708 rad). rotateGripper and unscrew
# both rotate this joint relative to a base pose; there is no separate wrist
# DOF, so "rotating the gripper" and "rotating the wrist" are the same thing
# in this rig.
WRIST_JOINT_INDEX = 6
# Conservative joint-limit guard so a bad LLM-supplied angle can't command a
# wrap-around motion. Not derived from the URDF — a placeholder safety bound
# until real joint limits are wired in from src/battery_dismantle_task/urdf.
WRIST_JOINT_SOFT_LIMIT_RAD = 2 * math.pi


class SkillHandlers:
    """Handles execution of high-level robot skills"""

    def __init__(self, node, waypoints, motion_executor, scene_manager):
        self.node = node
        self.waypoints = waypoints
        self.motion = motion_executor
        self.scene = scene_manager
        self.open_gripper_pose = "OPEN"
        self.close_gripper_pose = "CLOSE"

    def execute_grasp(self, object_name):
        """Execute complex grasp sequence"""
        self.node.get_logger().info(f"🤏 Executing complex grasp: {object_name}")

        if object_name not in self.waypoints.get("objects", {}):
            self.node.get_logger().error(f"Unknown object: {object_name}")
            return False

        obj_data = self.waypoints["objects"][object_name]

        # 1) Gripper on_approach
        pose = obj_data.get("gripper_hooks", {}).get("on_approach")
        if pose:
            joints = self.waypoints.get("poses", {}).get(pose)
            if not joints or not self.motion.plan_execute_gripper(joints, "gripper-open"):
                return False

        # 2) Approach
        joints = obj_data.get("approach")
        if not joints or not self.motion.plan_execute_arm(joints, "approach"):
            return False

        # 3) Gripper after_approach
        pose = obj_data.get("gripper_hooks", {}).get("after_approach")
        if pose:
            joints = self.waypoints.get("poses", {}).get(pose)
            if not joints or not self.motion.plan_execute_gripper(joints, "gripper-close"):
                return False

        # 4) Attach object to gripper in planning scene
        # Re-enabled: visual_state_manager's "handles this based on feedback"
        # claim was false — its update_attached_object timer is disabled
        # (visual_state_manager.py line ~129), so attach_object_visual() was
        # a no-op that only set a flag, never actually removing the static
        # world collision object. Result: after any grasp, the gripper
        # collided with the still-present world copy of the object on every
        # subsequent MoveIt planning request ("Skipping invalid start state"
        # -> "Catastrophic failure", confirmed via /check_state_validity
        # showing robotiq_85_*_finger_tip_link penetrating TopCoverBolts/
        # BatteryBox_0 by up to 8.2cm). This call uses AttachedCollisionObject,
        # which MoveIt automatically (a) removes from world collision objects
        # when the ID matches, and (b) keeps glued to end_effector_link via TF
        # with no polling timer needed.
        if not self.scene.attach_object(object_name, "end_effector_link"):
            self.node.get_logger().warn(f"⚠️  attach_object failed for '{object_name}' — "
                                         f"planning scene may still show a stale world collision")

        self.node.get_logger().info(f"✅ Grasp '{object_name}' done")
        return True

    def execute_release(self, object_name, place_joints):
        """Execute release sequence"""
        self.node.get_logger().info(f"✋ Executing release: {object_name}")

        if object_name not in self.waypoints.get("objects", {}):
            self.node.get_logger().error(f"Unknown object: {object_name}")
            return False

        # 1) Move to place position
        if not place_joints:
            self.node.get_logger().error(f"Cannot resolve place for '{object_name}'")
            return False
        if not self.motion.plan_execute_arm(place_joints, "place"):
            return False

        # 2) Open gripper
        open_joints = self.waypoints.get("poses", {}).get(self.open_gripper_pose)
        if not open_joints or not self.motion.plan_execute_gripper(open_joints, "release-gripper"):
            return False

        # 3) Detach object from gripper
        # Re-enabled alongside execute_grasp's attach_object — see that
        # comment for why visual_state_manager's own attach/detach path
        # doesn't actually work.
        if not self.scene.detach_object(object_name, "end_effector_link"):
            self.node.get_logger().warn(f"⚠️  detach_object failed for '{object_name}'")

        # 4) Retreat
        retreat_joints = self.waypoints["objects"][object_name].get("retreat")
        if retreat_joints and not self.motion.plan_execute_arm(retreat_joints, "retreat"):
            self.node.get_logger().warn("⚠️  Retreat failed, but release completed")

        self.node.get_logger().info(f"✅ Release '{object_name}' complete")
        return True

    def execute_dismantle(self, targets, place_default):
        """Execute dismantle sequence for multiple objects"""
        for i, obj in enumerate(targets):
            self.node.get_logger().info(f"Dismantle step {i+1}/{len(targets)}: {obj}")

            if not self.execute_grasp(obj):
                return False

            # Resolve place position
            place_joints = self._resolve_place_joints(obj, place_default)
            if not place_joints:
                self.node.get_logger().error(f"Cannot resolve place for {obj}")
                return False

            if not self.motion.plan_execute_arm(place_joints, "place"):
                return False

            # Open gripper
            open_joints = self.waypoints.get("poses", {}).get(self.open_gripper_pose)
            if not self.motion.plan_execute_gripper(open_joints, "openAfterPlace"):
                return False

            # Retreat
            retreat_joints = self.waypoints["objects"][obj].get("retreat")
            if retreat_joints and not self.motion.plan_execute_arm(retreat_joints, "retreat"):
                return False

        return True

    def _resolve_place_joints(self, object_name, place_in):
        """Resolve place joints from object and place info"""
        def try_bin_to_pose(bin_name):
            try:
                pose_name = self.waypoints["scene"]["bins"][bin_name]["pose_name"]
                return self.waypoints.get("poses", {}).get(pose_name)
            except (KeyError, TypeError):
                return None

        # From explicit 'place' parameter
        if place_in:
            if "bin" in place_in:
                joints = try_bin_to_pose(place_in["bin"])
                if joints:
                    return joints
            if "pose_name" in place_in:
                joints = self.waypoints.get("poses", {}).get(place_in["pose_name"])
                if joints:
                    return joints
            if "joints" in place_in and place_in["joints"]:
                return place_in["joints"]

        # From object's default place
        if object_name in self.waypoints.get("objects", {}):
            obj_data = self.waypoints["objects"][object_name]
            if "place" in obj_data and isinstance(obj_data["place"], list):
                return obj_data["place"]

        return None

    def rotated_joints(self, base_joints, angle_deg):
        """Return a copy of base_joints with the wrist joint rotated by angle_deg.

        Shared by skill_server's rotateGripper handler and execute_unscrew
        below, so both go through one clamp/limit check.
        """
        joints = list(base_joints)
        delta = math.radians(angle_deg)
        new_wrist = joints[WRIST_JOINT_INDEX] + delta
        new_wrist = max(-WRIST_JOINT_SOFT_LIMIT_RAD, min(WRIST_JOINT_SOFT_LIMIT_RAD, new_wrist))
        joints[WRIST_JOINT_INDEX] = new_wrist
        return joints

    def execute_unscrew(self, object_name, turns=2, degrees_per_turn=-90):
        """Simplified unscrew: approach, grip, rotate the wrist N times, release, retreat.

        HONESTY NOTE: there is no torque/force sensing on this rig (see
        REVISION_MEMO.md — tier-2 runtime validation was designed but never
        wired up), so this cannot detect "bolt loosened" vs "gripper just
        spun in place." It is a scripted motion sequence, not a verified
        bolt-removal action. Treat results from this skill as "the arm
        executed the unscrew MOTION", not "the bolt came out."
        """
        if object_name not in self.waypoints.get("objects", {}):
            self.node.get_logger().error(f"Unknown object: {object_name}")
            return False
        self.node.get_logger().info(f"🔩 Executing simplified unscrew: {object_name}")

        obj_data = self.waypoints["objects"][object_name]
        approach = obj_data.get("approach")
        if not approach or not self.motion.plan_execute_arm(approach, "unscrew-approach"):
            return False

        close_joints = self.waypoints.get("poses", {}).get(self.close_gripper_pose)
        if not close_joints or not self.motion.plan_execute_gripper(close_joints, "unscrew-grip"):
            return False

        joints = approach
        for i in range(turns):
            joints = self.rotated_joints(joints, degrees_per_turn)
            if not self.motion.plan_execute_arm(joints, f"unscrew-turn-{i+1}"):
                self.node.get_logger().warn(f"⚠️  Unscrew turn {i+1}/{turns} failed")
                return False

        open_joints = self.waypoints.get("poses", {}).get(self.open_gripper_pose)
        if open_joints:
            self.motion.plan_execute_gripper(open_joints, "unscrew-release")

        retreat = obj_data.get("retreat")
        if retreat:
            self.motion.plan_execute_arm(retreat, "unscrew-retreat")

        self.node.get_logger().info(f"✅ Unscrew motion sequence for '{object_name}' complete "
                                     f"(scripted, not force-verified)")
        return True

    def execute_disconnect(self, object_name):
        """Grip a connector, pull back (retreat), then release.

        Same honesty caveat as execute_unscrew: this is grip-and-pull, no
        force feedback confirms the connector actually disengaged.
        """
        if object_name not in self.waypoints.get("objects", {}):
            self.node.get_logger().error(
                f"Unknown object for disconnect: '{object_name}'. "
                f"Known objects: {list(self.waypoints.get('objects', {}).keys())}. "
                f"Add a waypoints.json entry for this connector before retrying.")
            return False
        self.node.get_logger().info(f"🔌 Executing disconnect: {object_name}")

        obj_data = self.waypoints["objects"][object_name]
        approach = obj_data.get("approach")
        if not approach or not self.motion.plan_execute_arm(approach, "disconnect-approach"):
            return False

        close_joints = self.waypoints.get("poses", {}).get(self.close_gripper_pose)
        if not close_joints or not self.motion.plan_execute_gripper(close_joints, "disconnect-grip"):
            return False

        retreat = obj_data.get("retreat")
        if not retreat or not self.motion.plan_execute_arm(retreat, "disconnect-pull"):
            return False

        open_joints = self.waypoints.get("poses", {}).get(self.open_gripper_pose)
        if open_joints:
            self.motion.plan_execute_gripper(open_joints, "disconnect-release")

        self.node.get_logger().info(f"✅ Disconnect '{object_name}' complete (grip-and-pull, not force-verified)")
        return True
