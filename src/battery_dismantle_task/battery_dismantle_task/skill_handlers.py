#!/usr/bin/python3
"""Skill execution handlers for robot tasks"""

import math

from .scene_manager import TABLE_Z, PLACED_OBJECT_DIMS

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

# End-effector orientations (x, y, z, w) for each named approach strategy, in
# the `world` frame. These define the grasp DIRECTION; the approach POSITION is
# derived from the object's live planning-scene pose (see _resolve_approach), so
# approach joints are computed by IK at runtime and track wherever the object
# is actually rendered instead of a hardcoded hint.
#   top_down : tool points straight down (180deg about Y) -> grasp from above.
#   side_-Y  : top_down rotated +90deg about X -> tool points along +Y, i.e.
#              approach a target from its -Y side (used for the battery body).
APPROACH_ORIENTATIONS = {
    "top_down": (0.0, 1.0, 0.0, 0.0),
    "side_-Y": (0.0, 0.7071067811865476, 0.7071067811865476, 0.0),
}

# Approach geometry (m). For a top-down grasp the end-effector is placed this
# far above the object centre along the gripper approach axis. Matches
# scene_manager.ATTACH_OFFSET_Z (the measured fingertip reach) so the grasped
# object stays in place (doesn't jump) when attached.
GRIPPER_GRASP_REACH = 0.04
SIDE_CLEARANCE = 0.10       # stand off this far beyond the object's -Y face


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

        # 2) Approach (IK-computed from object pose when available)
        joints = self._resolve_approach(object_name, obj_data)
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
        # Look up the object's real box size so the attached copy matches its
        # actual mesh (attach_object used to hardcode 5x5x2cm for every
        # object, so e.g. TopCoverBolts — 30x20x2cm — visibly shrank/clipped
        # the moment it was grasped).
        live_info = self.motion.get_object_pose(object_name)
        real_dims = live_info[1] if live_info and live_info[1] else None
        if not self.scene.attach_object(object_name, "end_effector_link", dimensions=real_dims):
            self.node.get_logger().error(f"❌ attach_object failed for '{object_name}' — "
                                          f"aborting grasp so the arm doesn't move an unattached object")
            return False

        self.node.get_logger().info(f"✅ Grasp '{object_name}' done")
        return True

    def _lower_onto_table(self, seed_joints, where="place-lower"):
        """Gently lower the held object straight down until it rests on the
        table, keeping the current gripper orientation, so the part is SET DOWN
        instead of dropped the moment the gripper opens.

        Reads the gripper's live world pose, solves IK for the same XY/orientation
        at a Z that puts the box on the table, and moves there. Returns True (a
        no-op) and lets the place continue if the pose/IK can't be resolved or
        the gripper is already at table height — the gentle lower is a polish
        step, not a hard gate.
        """
        ee = self.scene.get_link_world_pose("end_effector_link")
        if ee is None:
            self.node.get_logger().warn("place: no gripper pose; skipping gentle lower")
            return True
        (ex, ey, ez), quat = ee
        # Use the real held object's dims/offset (scene.held_z_offset), not a
        # generic constant — that used to assume every object sat at
        # ATTACH_OFFSET_Z, which stopped being true once oversized objects
        # (e.g. BatteryBox_0) got pushed forward to clear the gripper body.
        dims = self.scene.attached_dims or PLACED_OBJECT_DIMS
        target_z = TABLE_Z + dims[2] / 2.0 + self.scene.held_z_offset(dims)
        if ez - target_z <= 0.005:
            return True  # already at/below table height — nothing to lower
        joints = self.motion.compute_ik((ex, ey, target_z), quat, seed_joints=seed_joints)
        if not joints:
            self.node.get_logger().warn("place: IK for lower failed; skipping gentle lower")
            return True
        self.node.get_logger().info(
            f"⬇️  Lowering held object {ez - target_z:.3f} m onto the table before release")
        return self.motion.plan_execute_arm(joints, where)

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

        # 1.5) Gently lower so the object is set down, not dropped, on release.
        self._lower_onto_table(place_joints, "release-lower")

        # 2) Open gripper
        open_joints = self.waypoints.get("poses", {}).get(self.open_gripper_pose)
        if not open_joints or not self.motion.plan_execute_gripper(open_joints, "release-gripper"):
            return False

        # 3) Detach object from gripper
        # Re-enabled alongside execute_grasp's attach_object — see that
        # comment for why visual_state_manager's own attach/detach path
        # doesn't actually work.
        if not self.scene.detach_object(object_name, "end_effector_link"):
            self.node.get_logger().error(f"❌ detach_object failed for '{object_name}'")
            return False

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

            # Gently lower so the part is set down, not dropped, on release.
            self._lower_onto_table(place_joints, "place-lower")

            # Open gripper
            open_joints = self.waypoints.get("poses", {}).get(self.open_gripper_pose)
            if not self.motion.plan_execute_gripper(open_joints, "openAfterPlace"):
                return False

            # Leave the part on the table where it was released (was missing
            # here, so dismantle never actually let go of the object).
            if not self.scene.detach_object(obj, "end_effector_link"):
                self.node.get_logger().error(f"❌ detach_object failed for '{obj}'")
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

        # IK-computed from cartesian_hints.place_position, mirroring
        # _resolve_approach: the static 'place' joint array below was hand-
        # tuned once and isn't where cartesian_hints.place_position actually
        # is, so going there in joint space (direct execution, no Cartesian
        # path planning) can swing the arm through a big, unrelated arc.
        # Solving IK for the real place position keeps the motion comparable
        # in size/shape to the approach move.
        if object_name in self.waypoints.get("objects", {}):
            obj_data = self.waypoints["objects"][object_name]
            hint = obj_data.get("cartesian_hints", {}).get("place_position")
            strategy = obj_data.get("approach_strategy")
            orientation = APPROACH_ORIENTATIONS.get(strategy)
            if hint and orientation:
                target = (hint["x"], hint["y"], hint["z"] + GRIPPER_GRASP_REACH)
                seed = self.waypoints.get("poses", {}).get("HOME") or obj_data.get("approach")
                joints = self.motion.compute_ik(target, orientation, seed_joints=seed)
                if joints:
                    self.node.get_logger().info(
                        f"🧮 IK place for '{object_name}': target@"
                        f"({target[0]:.3f},{target[1]:.3f},{target[2]:.3f})")
                    return joints
                self.node.get_logger().warn(
                    f"⚠️  IK failed for '{object_name}' place; falling back to static joints")

            # From object's default place
            if "place" in obj_data and isinstance(obj_data["place"], list):
                return obj_data["place"]

        return None

    def _resolve_approach(self, object_name, obj_data):
        """Resolve the approach joint target for an object.

        Preferred path: read the object's live pose + box dimensions from the
        planning scene, offset it by a stand-off clearance in the direction the
        approach_strategy specifies, and solve IK for that Cartesian pose — so
        the approach point is always aligned with where the object is actually
        rendered, and auto-tracks it if the object moves. Falls back to the
        static 'approach' joint array when the object has no strategy, the
        strategy is unknown, the scene lookup fails, or IK fails.
        """
        static = obj_data.get("approach")
        strategy = obj_data.get("approach_strategy")
        if not strategy:
            return static  # object opted out of IK resolution

        orientation = APPROACH_ORIENTATIONS.get(strategy)
        if orientation is None:
            self.node.get_logger().warn(
                f"Unknown approach_strategy '{strategy}' for '{object_name}'; "
                f"using static approach joints")
            return static

        info = self.motion.get_object_pose(object_name)
        if not info or not info[1] or len(info[1]) < 3:
            self.node.get_logger().warn(
                f"⚠️  Could not read live pose for '{object_name}'; "
                f"using static approach joints")
            return static

        (ox, oy, oz), (dx, dy, dz) = info
        if strategy == "top_down":
            target = (ox, oy, oz + GRIPPER_GRASP_REACH)
        elif strategy == "side_-Y":
            target = (ox, oy - dy / 2.0 - SIDE_CLEARANCE, oz)
        else:
            return static

        seed = self.waypoints.get("poses", {}).get("HOME") or static
        joints = self.motion.compute_ik(target, orientation, seed_joints=seed)
        if joints:
            self.node.get_logger().info(
                f"🧮 IK approach for '{object_name}' [{strategy}]: object@"
                f"({ox:.3f},{oy:.3f},{oz:.3f}) -> approach@"
                f"({target[0]:.3f},{target[1]:.3f},{target[2]:.3f})")
            return joints

        self.node.get_logger().warn(
            f"⚠️  IK failed for '{object_name}' approach; falling back to static joints")
        return static

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
        approach = self._resolve_approach(object_name, obj_data)
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
        approach = self._resolve_approach(object_name, obj_data)
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
