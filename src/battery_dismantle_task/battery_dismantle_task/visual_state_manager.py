#!/usr/bin/env python3
"""
Visual state manager - owns creation and movement of scene objects.
Shows objects being grasped and moved in RViz.

How it works:
1. On start-up, create the battery and top-cover collision objects
2. Listen to skill feedback
3. grasp succeeded -> remove the object from the world and attach it to the gripper
4. release succeeded -> the object is re-created where it was released
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from tf2_ros import TransformListener, Buffer
import json


class VisualStateManager(Node):
    """
    Visual state manager for RViz scene visualization and object tracking.

    This node manages collision objects in the MoveIt planning scene, creating
    visual representations of battery components and updating their states based
    on robot manipulation feedback.

    Responsibilities:
        - Create initial scene objects (battery, top cover, bolts)
        - Listen to LLM command feedback to track object states
        - Attach/detach objects to/from gripper during grasp/release
        - Update object positions in the planning scene

    Subscribed Topics:
        /llm_feedback (std_msgs/String): Skill execution feedback
        /llm_commands (std_msgs/String): Current manipulation commands

    Service Clients:
        /apply_planning_scene: Apply scene updates to MoveIt
    """
    def __init__(self):
        super().__init__('visual_state_manager')

        # Subscribe to skill feedback
        self.feedback_sub = self.create_subscription(
            String,
            '/llm_feedback',
            self.feedback_callback,
            10
        )

        # Subscribe to LLM commands to track the current target
        self.command_sub = self.create_subscription(
            String,
            '/llm_commands',  # Fixed: subscribe to correct topic (plural)
            self.command_callback,
            10
        )

        # Planning-scene service client
        self.scene_client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')

        # Wait for the service
        self.get_logger().info('Waiting for /apply_planning_scene service...')
        while not self.scene_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /apply_planning_scene service...')

        # TF listener (for the gripper pose)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Object dimensions
        # Sized so a Robotiq-85 (max opening ~85mm) can actually grip them:
        # the battery pack is a fixed workpiece (not grasped); the small 5cm cover block on top is the grasp target.
        BATTERY_BASE_X = 0.45
        BATTERY_BASE_Y = 0.0
        BATTERY_BASE_Z = 0.0
        # Battery pack (workpiece). Width must be <= the gripper's max opening (~8.5cm) to be graspable,
        # otherwise the fingers penetrate the box -> invalid MoveIt start state -> move_group hangs.
        # See skill_handlers.GRIPPER_MAX_OPENING.
        BATTERY_W, BATTERY_D, BATTERY_H = 0.07, 0.07, 0.05
        COVER_W, COVER_D, COVER_H = 0.05, 0.05, 0.03         # graspable cover block

        self.object_definitions = {
            'TopCoverBolts': {
                'dimensions': [COVER_W, COVER_D, COVER_H],  # cover block (grasp target)
                'initial_pose': {
                    'x': BATTERY_BASE_X,
                    'y': BATTERY_BASE_Y,
                    'z': BATTERY_BASE_Z + BATTERY_H + COVER_H/2  # sits on top of the battery
                },
                'place_pose': {
                    'x': 0.3,
                    'y': -0.4,
                    'z': COVER_H/2  # tray position
                }
            },
            'BatteryBox_0': {
                'dimensions': [BATTERY_W, BATTERY_D, BATTERY_H],  # battery pack (workpiece)
                'initial_pose': {
                    'x': BATTERY_BASE_X,
                    'y': BATTERY_BASE_Y,
                    'z': BATTERY_BASE_Z + BATTERY_H/2  # centre
                },
                'place_pose': {
                    'x': 0.3,
                    'y': 0.4,
                    'z': BATTERY_H/2  # recycling-bin position
                }
            }
        }

        # currently grasped object
        self.grasped_object = None

        # current target object
        self.current_target = None

        # whether the original object was removed (so removal runs only once)
        self.original_removed = False

        # Give move_group time to start
        self.get_logger().info('Waiting 3s for move_group to start...')
        import time
        time.sleep(3)

        # Initialise the scene
        self.initialize_scene()

        # 10Hz attached-object pose update timer - disabled; the object is created once on grasp
        # self.update_timer = self.create_timer(0.1, self.update_attached_object)

        self.get_logger().info('[OK] Visual State Manager ready!')

    def command_callback(self, msg):
        """Handle an LLM command and record its target object."""
        try:
            command_json = json.loads(msg.data)
            # Extract target from params.object, params.target, params.object_id, or top-level target
            target = None
            if 'params' in command_json and isinstance(command_json['params'], dict):
                target = (command_json['params'].get('object') or
                         command_json['params'].get('target') or
                         command_json['params'].get('object_id'))
            if not target and 'target' in command_json:
                target = command_json['target']

            if target and target in self.object_definitions:
                self.current_target = target
                self.get_logger().info(f"Current target: {target}")
        except Exception as e:
            self.get_logger().debug(f"Command parse error: {e}")

    def clean_attached_objects(self):
        """Remove any leftover attached objects, one REMOVE request per object."""
        # Possible attached-object names (all naming formats)
        possible_attached = ['TopCoverBolts_attached', 'topcover_attached',
                           'BatteryBox_0_attached', 'BatteryBox_1_attached']

        for obj_name in possible_attached:
            req = ApplyPlanningScene.Request()
            req.scene = PlanningScene()
            req.scene.is_diff = True

            obj = CollisionObject()
            obj.id = obj_name
            obj.operation = CollisionObject.REMOVE
            req.scene.world.collision_objects.append(obj)

            # Async call; the short sleep gives MoveIt time to apply each removal
            self.scene_client.call_async(req)
            import time
            time.sleep(0.1)  # give MoveIt time to process

        self.get_logger().info('Cleared leftover attached objects')

    def initialize_scene(self):
        """Initialise the scene: create all collision objects."""
        # First remove any leftover attached objects
        self.clean_attached_objects()

        req = ApplyPlanningScene.Request()
        req.scene = PlanningScene()
        req.scene.is_diff = True

        # Create TopCoverBolts
        top_cover = self.create_collision_object(
            'TopCoverBolts',
            self.object_definitions['TopCoverBolts']['dimensions'],
            self.object_definitions['TopCoverBolts']['initial_pose']
        )
        req.scene.world.collision_objects.append(top_cover)

        # Create BatteryBox_0
        battery = self.create_collision_object(
            'BatteryBox_0',
            self.object_definitions['BatteryBox_0']['dimensions'],
            self.object_definitions['BatteryBox_0']['initial_pose']
        )
        req.scene.world.collision_objects.append(battery)

        # Allowed-collision matrix: TopCoverBolts may touch BatteryBox_0, plus gripper-internal contacts
        from moveit_msgs.msg import AllowedCollisionMatrix, AllowedCollisionEntry
        acm = AllowedCollisionMatrix()

        # Add scene objects + gripper links + arm links to the ACM
        gripper_and_arm_links = [
            'robotiq_85_left_inner_knuckle_link',
            'robotiq_85_left_finger_tip_link',
            'robotiq_85_right_inner_knuckle_link',
            'robotiq_85_right_finger_tip_link',
            'robotiq_85_left_knuckle_link',
            'robotiq_85_right_knuckle_link',
            'robotiq_85_left_finger_link',
            'robotiq_85_right_finger_link',
            'robotiq_85_base_link',
            'base_link',
            'shoulder_link',
            'bracelet_link',       # arm end link
            'end_effector_link'    # end-effector link
        ]
        acm.entry_names = ['BatteryBox_0', 'TopCoverBolts'] + gripper_and_arm_links

        # Build the ACM: gripper/arm links may touch each other, and scene objects may touch the gripper
        n = len(acm.entry_names)
        acm.entry_values = []

        for i in range(n):
            entry = AllowedCollisionEntry()
            enabled = []
            for j in range(n):
                # Collision rules between scene objects
                if i < 2 and j < 2:
                    enabled.append(i != j and (i == 1 or j == 1))  # TopCoverBolts may touch BatteryBox_0
                # All gripper/arm link pairs allowed
                elif i >= 2 and j >= 2:
                    enabled.append(True)
                # Scene objects vs gripper/arm allowed (needed for grasping)
                else:
                    enabled.append(True)  # allow scene object / gripper contact
            entry.enabled = enabled
            acm.entry_values.append(entry)

        req.scene.allowed_collision_matrix = acm

        # Call the service with retries. Right after move_group starts, /apply_planning_scene
        # may not confirm within a single 2s timeout, leaving the scene empty (the battery and
        # cover block 'disappear' in RViz). Retry and check success until the objects exist.
        import time
        created = False
        for attempt in range(1, 7):
            future = self.scene_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            if future.result() and future.result().success:
                self.get_logger().info('[OK] Created scene objects: TopCoverBolts, BatteryBox_0')
                created = True
                break
            self.get_logger().warn(f'[WARN] Scene-object creation not confirmed (attempt {attempt}/6), retrying...')
            time.sleep(1.0)
        if not created:
            self.get_logger().error('[FAIL] Could not create scene objects (6 attempts)')

    def create_collision_object(self, object_id, dimensions, pose_dict):
        """Create a collision object."""
        collision_obj = CollisionObject()
        collision_obj.header.frame_id = 'world'
        collision_obj.id = object_id
        collision_obj.operation = CollisionObject.ADD

        # Shape
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = dimensions

        # Pose
        pose = Pose()
        pose.position.x = pose_dict['x']
        pose.position.y = pose_dict['y']
        pose.position.z = pose_dict['z']
        pose.orientation.w = 1.0

        collision_obj.primitives.append(primitive)
        collision_obj.primitive_poses.append(pose)

        return collision_obj

    def feedback_callback(self, msg):
        """Handle skill feedback."""
        try:
            data = msg.data

            # Try to parse JSON feedback
            try:
                feedback_json = json.loads(data)
                message_lower = feedback_json.get('message', '').lower()

                # Detect a successful grasp - message format: "Skill ''grasp'' completed"
                if (feedback_json.get('status') == 'success' and
                    'grasp' in message_lower and
                    'completed' in message_lower):

                    # Use current_target (set in command_callback)
                    if self.current_target and self.current_target in self.object_definitions:
                        self.get_logger().info(f"Grasp succeeded: {self.current_target}")
                        self.attach_object_visual(self.current_target)
                    else:
                        self.get_logger().warn(f"[WARN] Grasp succeeded but no current target or target undefined: {self.current_target}")

                # Detect a successful release
                elif (feedback_json.get('status') == 'success' and
                      'release' in message_lower and
                      'completed' in message_lower):

                    if self.grasped_object:
                        self.get_logger().info(f"Release succeeded: {self.grasped_object}")
                        self.detach_object_visual(self.grasped_object)
                    else:
                        self.get_logger().warn("[WARN] Release succeeded but no object was grasped")

            except json.JSONDecodeError:
                # Legacy message format
                data_lower = data.lower()
                if 'grasp' in data_lower and 'completed' in data_lower:
                    if self.current_target and self.current_target in self.object_definitions:
                        self.get_logger().info(f"Grasp succeeded: {self.current_target}")
                        self.attach_object_visual(self.current_target)

                elif 'release' in data_lower and 'completed' in data_lower:
                    if self.grasped_object:
                        self.get_logger().info(f"Release succeeded: {self.grasped_object}")
                        self.detach_object_visual(self.grasped_object)

        except Exception as e:
            self.get_logger().error(f"Feedback parse error: {e}")

    def attach_object_visual(self, object_id):
        """Visualise a grasp (remove from the world, ready to follow the gripper)."""
        # Mark as grasped; update_attached_object creates the gripper-following copy
        self.grasped_object = object_id
        self.original_removed = False  # reset flag
        self.get_logger().info(f"  [OK] {object_id} marked as grasped (will follow the gripper)")

    def detach_object_visual(self, object_id):
        """Placement is handled by scene_manager.detach_object (the part stays where the
        gripper let go). Re-creating the object at a fixed point here was what made parts
        teleport to the floor, so this only clears state."""
        self.get_logger().info(f"  [OK] {object_id} placement handled by scene_manager (left where released)")
        self.grasped_object = None

    def update_attached_object(self):
        """Timer update: make the grasped object follow the gripper."""
        if self.grasped_object is None:
            return

        try:
            # Current gripper pose
            transform = self.tf_buffer.lookup_transform(
                'world',
                'robotiq_85_base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05)
            )

            # Create/update the object at the gripper
            req = ApplyPlanningScene.Request()
            req.scene = PlanningScene()
            req.scene.is_diff = True

            collision_obj = CollisionObject()
            collision_obj.header.frame_id = 'world'
            collision_obj.id = f"{self.grasped_object}_attached"
            collision_obj.operation = CollisionObject.ADD

            # Shape
            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = self.object_definitions[self.grasped_object]['dimensions']

            # Pose (object follows the gripper, offset by object size)
            pose = Pose()
            # Object centre sits below the gripper: fingertip length (~8cm) + half the object height
            object_height = self.object_definitions[self.grasped_object]['dimensions'][2]
            offset_z = -0.08 - object_height / 2.0  # below the gripper

            pose.position.x = transform.transform.translation.x
            pose.position.y = transform.transform.translation.y
            pose.position.z = transform.transform.translation.z + offset_z
            pose.orientation = transform.transform.rotation

            collision_obj.primitives.append(primitive)
            collision_obj.primitive_poses.append(pose)
            req.scene.world.collision_objects.append(collision_obj)

            # On the first run, also remove the original object
            if not self.original_removed:
                original_obj = CollisionObject()
                original_obj.id = self.grasped_object
                original_obj.operation = CollisionObject.REMOVE
                req.scene.world.collision_objects.append(original_obj)
                self.original_removed = True
                self.get_logger().info(f"  Removed original {self.grasped_object}; created following copy")

            # Async call (don't wait, to avoid blocking)
            self.scene_client.call_async(req)

        except Exception:
            # TF lookups can fail transiently; not logged
            pass


def main(args=None):
    rclpy.init(args=args)
    node = VisualStateManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
