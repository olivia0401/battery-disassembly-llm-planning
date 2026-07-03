#!/usr/bin/python3
"""Planning Scene Manager for object attach/detach operations"""

from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from tf2_ros import Buffer, TransformListener
import rclpy
import time

# Size of the placed-down part (matches the attached box in attach_object).
PLACED_OBJECT_DIMS = [0.05, 0.05, 0.03]
# Table/ground height (base_link at z=0 in this rig).
TABLE_Z = 0.0
# How long to wait for /apply_planning_scene to actually confirm an
# attach/detach before giving up. Must not be infinite (a wedged move_group
# would hang the command worker thread forever).
PLANNING_SCENE_TIMEOUT_SEC = 5.0
# Measured (via tf2_echo) translation from end_effector_link to
# robotiq_85_left/right_finger_tip_link: both fingertips sit at z=+0.040 in
# the gripper's local frame (their X offsets are mirrored and cancel at the
# grasp centerline). This is the gripper's actual reach.
FINGER_REACH_Z = 0.04
# How far below end_effector_link a held object that fits within the gripper
# sits (must match GRIPPER_GRASP_REACH in skill_handlers, so the object sets
# down where the gripper actually is). Tied to the measured FINGER_REACH_Z
# instead of being its own guessed constant, so the two can't drift apart.
ATTACH_OFFSET_Z = FINGER_REACH_Z
# Robotiq 2F-85's max opening (~8.5cm). Objects wider than this in X or Y
# (e.g. BatteryBox_0, 16x12cm) physically can't be enclosed by the gripper —
# see the z_offset branch in attach_object.
MAX_GRASP_VISUAL_WIDTH = 0.08
GRASP_CLEARANCE = 0.005  # small buffer past the fingertip plane


class SceneManager:
    """Manages Planning Scene for object attachment/detachment"""

    def __init__(self, node, planning_scene_client):
        self.node = node
        self._planning_scene_client = planning_scene_client
        self._attached_object = None
        # Real dims of whatever's currently attached (set by attach_object),
        # so detach_object re-adds the world copy at the correct size instead
        # of a generic placeholder.
        self._attached_dims = None
        # TF so detach can leave the part where the gripper released it.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)

    def held_z_offset(self, dims):
        """How far below end_effector_link an object with box `dims` sits
        while held. Single source of truth for attach_object, detach_object,
        AND skill_handlers._lower_onto_table — they used to each hardcode
        their own constant, which drifted out of sync (e.g. the gentle-lower
        step assumed everything used ATTACH_OFFSET_Z, which stopped being
        true once oversized objects got pushed forward to FINGER_REACH_Z)."""
        if dims[0] <= MAX_GRASP_VISUAL_WIDTH and dims[1] <= MAX_GRASP_VISUAL_WIDTH:
            return ATTACH_OFFSET_Z
        return FINGER_REACH_Z + GRASP_CLEARANCE + dims[2] / 2.0

    def _call_and_wait(self, request, timeout_sec=PLANNING_SCENE_TIMEOUT_SEC):
        """Send an ApplyPlanningScene request and block until move_group has
        actually applied it (or we time out). The planning scene client runs
        on the ReentrantCallbackGroup, so its response is processed on
        another executor thread while we poll here. Returns True only if
        move_group confirmed success — callers must not assume a sent
        request means the scene was actually updated."""
        future = self._planning_scene_client.call_async(request)
        start = time.monotonic()
        while not future.done():
            if time.monotonic() - start > timeout_sec:
                self.node.get_logger().error(
                    "⏱️  /apply_planning_scene timed out waiting for a response")
                return False
            time.sleep(0.01)
        exc = future.exception()
        if exc is not None:
            self.node.get_logger().error(f"/apply_planning_scene call failed: {exc}")
            return False
        response = future.result()
        if not response.success:
            self.node.get_logger().error("/apply_planning_scene reported failure")
        return response.success

    def attach_object(self, object_name, link_name="end_effector_link", dimensions=None):
        """Attach an object to the robot gripper in the planning scene.

        `dimensions` should be the object's REAL [x, y, z] box size (e.g. from
        the live world collision object via MotionExecutor.get_object_pose).
        Without it we used to hardcode a 5x5x2cm box for every object, so a
        30x20cm part like TopCoverBolts visibly shrank to a tiny box the
        instant it was grasped — a size mismatch that looks like clipping/
        bad collision geometry against the gripper and the rest of the scene.
        """
        try:
            self.node.get_logger().info(f"🔗 Attaching '{object_name}' to '{link_name}'...")

            if not self._planning_scene_client.wait_for_service(timeout_sec=5.0):
                self.node.get_logger().warn(f"⚠️  Planning scene service not available")
                return False

            # Create an AttachedCollisionObject message
            aco = AttachedCollisionObject()
            aco.link_name = link_name
            aco.object.id = object_name
            aco.object.operation = CollisionObject.ADD
            aco.object.header.frame_id = link_name

            # Use the object's real size if known; fall back to the old
            # placeholder only if the caller couldn't look it up. Always the
            # real size — shrinking it to fit visually just trades one bug
            # (object clips the gripper) for another (object visibly shrinks
            # the instant it's picked up).
            box_dims = list(dimensions) if dimensions and len(dimensions) == 3 else [0.05, 0.05, 0.02]
            self._attached_dims = box_dims
            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = box_dims
            aco.object.primitives.append(primitive)

            # Position relative to the gripper link. Objects that fit within
            # the gripper's opening nestle at the measured fingertip depth
            # (ATTACH_OFFSET_Z) like before. Objects WIDER than the gripper
            # can ever close around (e.g. BatteryBox_0, 16x12cm vs an 8.5cm
            # max opening) will clip the gripper body at ANY X/Y if they
            # share the same Z depth — shrinking width doesn't fix that, only
            # moving them far enough along Z that they no longer overlap the
            # gripper's own geometry at all does. So oversized objects hang
            # forward, clear of FINGER_REACH_Z, instead of nestled inside it.
            z_offset = self.held_z_offset(box_dims)

            pose = Pose()
            pose.position.x = 0.0
            pose.position.y = 0.0
            pose.position.z = z_offset  # below gripper along approach axis
            pose.orientation.w = 1.0
            aco.object.primitive_poses.append(pose)

            # Create planning scene message. Both the top-level scene.is_diff
            # AND the nested robot_state.is_diff must be set — move_group
            # silently ignores attached_collision_objects edits otherwise
            # ("RobotState is not marked as is_diff... Object is ignored"),
            # while still returning success=True, so the object never
            # actually attaches/detaches even though we think it did.
            planning_scene = PlanningScene()
            planning_scene.robot_state.attached_collision_objects.append(aco)
            planning_scene.robot_state.is_diff = True
            planning_scene.is_diff = True

            # Send to planning scene and wait for move_group to confirm it
            # before reporting success — otherwise a caller (e.g. the place
            # skill) can start moving the arm before the object is actually
            # attached, which looks like the gripper suddenly dropping it.
            request = ApplyPlanningScene.Request()
            request.scene = planning_scene
            if not self._call_and_wait(request):
                self.node.get_logger().error(f"❌ Attach of '{object_name}' was not confirmed")
                return False

            self._attached_object = object_name
            self.node.get_logger().info(f"✅ Attached '{object_name}' to '{link_name}'")
            return True

        except Exception as e:
            self.node.get_logger().error(f"Exception in attach_object: {e}", exc_info=True)
            return False

    def get_link_world_pose(self, link_name="end_effector_link"):
        """Return ((x, y, z), (qx, qy, qz, qw)) of `link_name` in the world
        frame, or None if TF isn't available yet. Lets the place sequence lower
        the held object straight down (keeping the current orientation) and lets
        detach leave it exactly where the gripper released it."""
        try:
            tf = self.tf_buffer.lookup_transform(
                "world", link_name, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
            t = tf.transform.translation
            r = tf.transform.rotation
            return (t.x, t.y, t.z), (r.x, r.y, r.z, r.w)
        except Exception as e:
            self.node.get_logger().warn(f"get_link_world_pose: TF lookup failed ({e})")
            return None

    def detach_object(self, object_name, link_name="end_effector_link"):
        """Release an object and leave it resting where the gripper let go —
        instead of removing it (which made a separate visual copy teleport to a
        fixed spot) or snapping it down to table height (which made it "drop"
        the instant the gripper opened). Looks up the gripper's current world
        pose and re-adds the part at the object's ACTUAL released height
        (gripper Z minus the attach offset), clamped so it can't sink through
        the table, while removing the attached copy.
        """
        try:
            self.node.get_logger().info(f"🔓 Releasing '{object_name}' where the gripper let go...")

            if not self._planning_scene_client.wait_for_service(timeout_sec=5.0):
                self.node.get_logger().warn(f"⚠️  Planning scene service not available")
                return False

            # Use the real attached dims (set by attach_object) so the
            # re-added world copy is the object's actual size, not a generic
            # placeholder that doesn't match its mesh/visual.
            dims = self._attached_dims if self._attached_dims else PLACED_OBJECT_DIMS
            # Must match attach_object's offset, or the object visibly jumps
            # by the difference the instant it's released.
            z_offset = self.held_z_offset(dims)

            place_pose = None
            ee = self.get_link_world_pose(link_name)
            if ee is not None:
                (ex, ey, ez), _ = ee
                # Object centre sits z_offset below the gripper; clamp so it
                # rests on (never below) the table if released right at it.
                rest_z = max(TABLE_Z + dims[2] / 2.0, ez - z_offset)
                place_pose = (ex, ey, rest_z)
            else:
                self.node.get_logger().warn("detach: TF lookup failed; removing without re-place")

            # Step 1: remove the attached copy from the gripper. Must be its own
            # planning-scene diff — combining detach + world-add of the same id
            # in one message leaves the object BOTH attached and in the world.
            ps_detach = PlanningScene()
            ps_detach.is_diff = True
            aco = AttachedCollisionObject()
            aco.link_name = link_name
            aco.object.id = object_name
            aco.object.operation = CollisionObject.REMOVE
            aco.object.header.frame_id = link_name
            ps_detach.robot_state.attached_collision_objects.append(aco)
            # Same nested-is_diff requirement as attach_object — without this
            # move_group silently ignores the REMOVE and the object stays
            # attached to the gripper forever even though we get success=True.
            ps_detach.robot_state.is_diff = True
            req1 = ApplyPlanningScene.Request()
            req1.scene = ps_detach
            if not self._call_and_wait(req1):
                self.node.get_logger().error(f"❌ Detach of '{object_name}' was not confirmed")
                return False

            # Step 2: re-add it to the world at the gripper's released height.
            if place_pose is not None:
                ps_place = PlanningScene()
                ps_place.is_diff = True
                world_obj = CollisionObject()
                world_obj.header.frame_id = "world"
                world_obj.id = object_name
                world_obj.operation = CollisionObject.ADD
                prim = SolidPrimitive()
                prim.type = SolidPrimitive.BOX
                prim.dimensions = dims
                pose = Pose()
                pose.position.x = place_pose[0]
                pose.position.y = place_pose[1]
                pose.position.z = place_pose[2]  # where the gripper let go
                pose.orientation.w = 1.0
                world_obj.primitives.append(prim)
                world_obj.primitive_poses.append(pose)
                ps_place.world.collision_objects.append(world_obj)
                req2 = ApplyPlanningScene.Request()
                req2.scene = ps_place
                if not self._call_and_wait(req2):
                    self.node.get_logger().error(
                        f"❌ Re-add of released '{object_name}' to the world was not confirmed")
                    return False

            self._attached_object = None
            self._attached_dims = None
            where = tuple(round(v, 3) for v in place_pose) if place_pose else "(removed)"
            self.node.get_logger().info(f"✅ Released '{object_name}' at {where}")
            return True

        except Exception as e:
            self.node.get_logger().error(f"Exception in detach_object: {e}", exc_info=True)
            return False

    @property
    def attached_object(self):
        return self._attached_object

    @property
    def attached_dims(self):
        return self._attached_dims
