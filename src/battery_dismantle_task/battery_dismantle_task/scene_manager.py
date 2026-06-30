#!/usr/bin/python3
"""Planning Scene Manager for object attach/detach operations"""

from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose


class SceneManager:
    """Manages Planning Scene for object attachment/detachment"""

    def __init__(self, node, planning_scene_client):
        self.node = node
        self._planning_scene_client = planning_scene_client
        self._attached_object = None

    def attach_object(self, object_name, link_name="end_effector_link"):
        """Attach an object to the robot gripper in the planning scene"""
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

            # Define a simple box shape for the object
            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = [0.05, 0.05, 0.02]  # 5cm x 5cm x 2cm
            aco.object.primitives.append(primitive)

            # Position relative to the gripper link
            pose = Pose()
            pose.position.x = 0.0
            pose.position.y = 0.0
            pose.position.z = 0.05  # 5cm in front of gripper
            pose.orientation.w = 1.0
            aco.object.primitive_poses.append(pose)

            # Create planning scene message
            planning_scene = PlanningScene()
            planning_scene.robot_state.attached_collision_objects.append(aco)
            planning_scene.is_diff = True

            # Send to planning scene (fire and forget)
            request = ApplyPlanningScene.Request()
            request.scene = planning_scene
            self._planning_scene_client.call_async(request)

            self._attached_object = object_name
            self.node.get_logger().info(f"✅ Sent attach request for '{object_name}' to '{link_name}'")
            return True

        except Exception as e:
            self.node.get_logger().error(f"Exception in attach_object: {e}", exc_info=True)
            return False

    def detach_object(self, object_name, link_name="end_effector_link"):
        """Detach an object from the robot gripper in the planning scene"""
        try:
            self.node.get_logger().info(f"🔓 Detaching '{object_name}' from '{link_name}'...")

            if not self._planning_scene_client.wait_for_service(timeout_sec=5.0):
                self.node.get_logger().warn(f"⚠️  Planning scene service not available")
                return False

            # Create an AttachedCollisionObject message for removal
            aco = AttachedCollisionObject()
            aco.link_name = link_name
            aco.object.id = object_name
            aco.object.operation = CollisionObject.REMOVE
            aco.object.header.frame_id = link_name

            # Create planning scene message
            planning_scene = PlanningScene()
            planning_scene.robot_state.attached_collision_objects.append(aco)
            planning_scene.is_diff = True

            # Send to planning scene (fire and forget)
            request = ApplyPlanningScene.Request()
            request.scene = planning_scene
            self._planning_scene_client.call_async(request)

            self._attached_object = None
            self.node.get_logger().info(f"✅ Sent detach request for '{object_name}' from '{link_name}'")
            return True

        except Exception as e:
            self.node.get_logger().error(f"Exception in detach_object: {e}", exc_info=True)
            return False

    @property
    def attached_object(self):
        return self._attached_object
