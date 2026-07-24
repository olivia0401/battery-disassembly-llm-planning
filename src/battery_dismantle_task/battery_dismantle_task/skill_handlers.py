#!/usr/bin/python3
"""Skill execution handlers for robot tasks"""

import math
import time

from .scene_manager import TABLE_Z, PLACED_OBJECT_DIMS

# After a direct-trajectory arm move, the fake controller reports the goal
# 'done' a beat before robot_state_publisher republishes the settled TF. Any
# code that reads the gripper's world pose immediately after a move can catch a
# stale / mid-swing transform; feeding that into an IK target then sends the arm
# (and the held object) somewhere wrong — e.g. the held battery ending up
# hovering in mid-air instead of set down on the table. Let TF catch up first.
TF_SETTLE_SEC = 0.4

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

# Robotiq 2F-85 maximum finger opening (~8.5 cm). An object whose footprint is
# wider than this in BOTH horizontal axes can never be enclosed by the gripper.
# Grasping it anyway drives the fingers into the box, which corrupts MoveIt's
# start state and jams move_group for every subsequent plan (the "box frozen in
# mid-air, nothing works until restart" failure). execute_grasp checks this up
# front and refuses before any motion.
GRIPPER_MAX_OPENING = 0.085


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
        self.node.get_logger().info(f"Executing complex grasp: {object_name}")

        if object_name not in self.waypoints.get("objects", {}):
            self.node.get_logger().error(f"Unknown object: {object_name}")
            return False

        obj_data = self.waypoints["objects"][object_name]
        # One authoritative box size (live scene -> waypoints-declared ->
        # default), resolved ONCE and reused for the oversize check AND the
        # attach below, so a flaky live-scene read can't make the two disagree
        # (which is what let the attached copy shrink to the placeholder size).
        obj_dims = self._object_dims(object_name)

        # 0) Refuse ungraspable objects BEFORE any motion. Rejecting oversized
        # targets here is what stops the failed grasp from jamming move_group
        # (see GRIPPER_MAX_OPENING).
        if min(obj_dims[0], obj_dims[1]) > GRIPPER_MAX_OPENING:
            self.node.get_logger().error(
                f"[FAIL] Cannot grasp '{object_name}': footprint "
                f"{obj_dims[0]*100:.0f}x{obj_dims[1]*100:.0f} cm exceeds the gripper's "
                f"{GRIPPER_MAX_OPENING*100:.1f} cm opening in both axes. "
                f"Refusing before motion so move_group is not left in a bad state.")
            return False

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
        # Attach at the object's real box size (resolved above) so the attached
        # copy matches its mesh instead of shrinking to attach_object's
        # placeholder. Logged so a wrong size is visible in the skill_server log.
        self.node.get_logger().info(f"Attaching '{object_name}' at dims={obj_dims}")
        if not self.scene.attach_object(object_name, "end_effector_link", dimensions=obj_dims):
            self.node.get_logger().error(f"[FAIL] attach_object failed for '{object_name}' — "
                                          f"aborting grasp so the arm doesn't move an unattached object")
            return False

        self.node.get_logger().info(f"[OK] Grasp '{object_name}' done")
        return True

    def _object_dims(self, object_name):
        """Resolve an object's [x, y, z] box size from a single, reliable order:
        live planning-scene geometry -> waypoints-declared `dimensions` ->
        generic default. Shared by grasp (attach) and release (set-down height +
        placement) so they can never disagree about how big the object is. The
        live query can transiently return no geometry (see get_object_pose), so
        the declared size is the safety net that stops the attached/placed copy
        from collapsing to the placeholder box."""
        info = self.motion.get_object_pose(object_name)
        live = info[1] if info and info[1] and len(info[1]) >= 3 else None
        declared = (self.waypoints.get("objects", {})
                    .get(object_name, {}).get("dimensions"))
        return list(live or declared or PLACED_OBJECT_DIMS)

    def _place_world_xy(self, object_name):
        """(x, y) world location to set an object down at, from its
        cartesian_hints.place_position, or None if it has none. Lets release put
        the object at a KNOWN spot instead of wherever a racy post-move gripper
        TF read lands."""
        hint = (self.waypoints.get("objects", {}).get(object_name, {})
                .get("cartesian_hints", {}).get("place_position"))
        if hint and "x" in hint and "y" in hint:
            return (float(hint["x"]), float(hint["y"]))
        return None

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
        # Let the preceding move's TF settle before trusting the gripper pose
        # (see TF_SETTLE_SEC) — a stale read here is what sent the held object
        # to a mid-air "place" instead of down onto the table.
        time.sleep(TF_SETTLE_SEC)
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
            f" Lowering held object {ez - target_z:.3f} m onto the table before release")
        return self.motion.plan_execute_arm(joints, where)

    def execute_release(self, object_name, place_joints):
        """Execute release sequence"""
        self.node.get_logger().info(f"Executing release: {object_name}")

        if object_name not in self.waypoints.get("objects", {}):
            self.node.get_logger().error(f"Unknown object: {object_name}")
            return False

        # 1) Move to the set-down pose. _resolve_place_joints now targets the
        # table set-down height directly (gripper just above the table at the
        # place location), so the held object rests on the table the moment the
        # gripper opens — no separate "lower" step reading a racy live gripper
        # TF, which is what used to strand the object hovering in mid-air.
        if not place_joints:
            self.node.get_logger().error(f"Cannot resolve place for '{object_name}'")
            return False
        if not self.motion.plan_execute_arm(place_joints, "place"):
            return False

        # 2) Open gripper
        open_joints = self.waypoints.get("poses", {}).get(self.open_gripper_pose)
        if not open_joints or not self.motion.plan_execute_gripper(open_joints, "release-gripper"):
            return False

        # 3) Detach and drop the part at the gripper's ACTUAL pose — the forward
        # kinematics of the joints it was just commanded to. The arm reliably
        # reaches its commanded joints, so FK(place_joints) is exactly where the
        # gripper is; the held part hangs one grasp-reach below it, so it lands
        # right under the fingers with NO teleport. (Placing at a separate
        # target the IK may only have approximately reached is what made the
        # second part fly across to the drop point.)
        dims = self._object_dims(object_name)
        gripper_fk = self.motion.fk(place_joints)
        if gripper_fk is not None:
            place_at = (gripper_fk[0], gripper_fk[1],
                        max(TABLE_Z + dims[2] / 2.0, gripper_fk[2] - GRIPPER_GRASP_REACH))
            self.node.get_logger().info(
                f"release '{object_name}': gripper@({gripper_fk[0]:.3f},"
                f"{gripper_fk[1]:.3f},{gripper_fk[2]:.3f}) -> part@("
                f"{place_at[0]:.3f},{place_at[1]:.3f},{place_at[2]:.3f})")
        else:
            xy = self._place_world_xy(object_name)
            place_at = (xy[0], xy[1], TABLE_Z + dims[2] / 2.0) if xy else None
        if not self.scene.detach_object(object_name, "end_effector_link", place_at=place_at):
            self.node.get_logger().error(f"[FAIL] detach_object failed for '{object_name}'")
            return False

        # 4) Retreat
        retreat_joints = self.waypoints["objects"][object_name].get("retreat")
        if retreat_joints and not self.motion.plan_execute_arm(retreat_joints, "retreat"):
            self.node.get_logger().warn("[WARN] Retreat failed, but release completed")

        self.node.get_logger().info(f"[OK] Release '{object_name}' complete")
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
                self.node.get_logger().error(f"[FAIL] detach_object failed for '{obj}'")
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
                # Target the table SET-DOWN height directly: put the gripper one
                # fingertip-reach above where the object's centre will rest on
                # the table (TABLE_Z + half height). Opening the gripper here
                # sets the object straight down — no separate lower step, and
                # release places it deterministically at this same spot.
                dims = self._object_dims(object_name)
                setdown_z = TABLE_Z + dims[2] / 2.0 + GRIPPER_GRASP_REACH
                target = (hint["x"], hint["y"], setdown_z)
                # seed_joints=None -> seed from the arm's current configuration
                # (minimal move) instead of a fixed HOME branch.
                joints = self.motion.compute_ik(target, orientation, seed_joints=None)
                if joints:
                    self.node.get_logger().info(
                        f"IK place for '{object_name}': target@"
                        f"({target[0]:.3f},{target[1]:.3f},{target[2]:.3f})")
                    return joints
                self.node.get_logger().warn(
                    f"[WARN] IK failed for '{object_name}' place; falling back to static joints")

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
                f"[WARN] Could not read live pose for '{object_name}'; "
                f"using static approach joints")
            return static

        (ox, oy, oz), (dx, dy, dz) = info
        if strategy == "top_down":
            target = (ox, oy, oz + GRIPPER_GRASP_REACH)
        elif strategy == "side_-Y":
            target = (ox, oy - dy / 2.0 - SIDE_CLEARANCE, oz)
        else:
            return static

        # seed_joints=None -> motion_executor seeds IK from the arm's CURRENT
        # configuration, so the approach is the minimal move from where the arm
        # is now (not a jump to some HOME-seeded branch, which made the arm
        # swing wildly between steps).
        joints = self.motion.compute_ik(target, orientation, seed_joints=None)
        if joints:
            self.node.get_logger().info(
                f"IK approach for '{object_name}' [{strategy}]: object@"
                f"({ox:.3f},{oy:.3f},{oz:.3f}) -> approach@"
                f"({target[0]:.3f},{target[1]:.3f},{target[2]:.3f})")
            return joints

        self.node.get_logger().warn(
            f"[WARN] IK failed for '{object_name}' approach; falling back to static joints")
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
        self.node.get_logger().info(f"Executing simplified unscrew: {object_name}")

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
                self.node.get_logger().warn(f"[WARN] Unscrew turn {i+1}/{turns} failed")
                return False

        open_joints = self.waypoints.get("poses", {}).get(self.open_gripper_pose)
        if open_joints:
            self.motion.plan_execute_gripper(open_joints, "unscrew-release")

        retreat = obj_data.get("retreat")
        if retreat:
            self.motion.plan_execute_arm(retreat, "unscrew-retreat")

        self.node.get_logger().info(f"[OK] Unscrew motion sequence for '{object_name}' complete "
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
        self.node.get_logger().info(f"Executing disconnect: {object_name}")

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

        self.node.get_logger().info(f"[OK] Disconnect '{object_name}' complete (grip-and-pull, not force-verified)")
        return True
