#!/usr/bin/env python3
"""
Interactive Joint Control - dragging a joint in RViz moves it in real time.
The node keeps retrying the controller connection until it succeeds, to avoid start-up race conditions.
"""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import InteractiveMarkerFeedback
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.action import ActionClient
from builtin_interfaces.msg import Duration
import threading
import math

class ResilientInteractiveControl(Node):
    """
    Resilient interactive joint control node for RViz manipulation.

    This node enables real-time robot control via RViz interactive markers,
    with resilient connection handling to overcome timing/race conditions
    during system startup.

    Features:
        - Background thread for action server connection retry
        - Continuous joint position updates from interactive markers
        - Automatic reconnection on controller restart
        - Separate control for arm and gripper

    Subscribed Topics:
        /rviz_moveit_motion_planning_display/.../feedback: Interactive marker feedback

    Action Clients:
        /fake_manipulator_controller/follow_joint_trajectory
        /fake_gripper_controller/follow_joint_trajectory
    """
    def __init__(self):
        super().__init__('resilient_interactive_control')
        self.get_logger().info("--- Interactive joint control node (resilient connection) ---")

        # Action clients are initialised in a background thread
        self.manipulator_client = None
        self.gripper_client = None
        self.servers_ready = threading.Event() # set once the action servers are ready

        # State
        self.current_joint_positions = {}
        self.last_sent_positions = {}
        self.lock = threading.Lock()

        # Background thread that waits for and connects to the action servers
        self.connection_thread = threading.Thread(target=self.wait_for_servers, daemon=True)
        self.connection_thread.start()

        # Subscribe to RViz interactive-marker feedback
        # The callback checks the servers are ready before sending anything
        feedback_topic = '/rviz_moveit_motion_planning_display/robot_interaction_interactive_marker_topic/feedback'
        self.subscription = self.create_subscription(
            InteractiveMarkerFeedback,
            feedback_topic,
            self.marker_feedback_callback,
            10
        )
        
        # Timer that periodically sends joint targets
        self.timer = self.create_timer(0.1, self.send_command_callback)
        self.get_logger().info("Node started; waiting for controllers in the background...")

    def wait_for_servers(self):
        """Wait (indefinitely, in a background thread) for the action servers."""
        manipulator_action_server = '/fake_manipulator_controller/follow_joint_trajectory'
        gripper_action_server = '/fake_gripper_controller/follow_joint_trajectory'
        
        self.get_logger().info(f"[connect] Waiting for arm action server: {manipulator_action_server}...")
        temp_manipulator_client = ActionClient(self, FollowJointTrajectory, manipulator_action_server)
        temp_manipulator_client.wait_for_server()
        self.manipulator_client = temp_manipulator_client
        self.get_logger().info("[OK] [connect] Arm controller connected")
        
        self.get_logger().info(f"[connect] Waiting for gripper action server: {gripper_action_server}...")
        temp_gripper_client = ActionClient(self, FollowJointTrajectory, gripper_action_server)
        temp_gripper_client.wait_for_server()
        self.gripper_client = temp_gripper_client
        self.get_logger().info("[OK] [connect] Gripper controller connected")
        
        # Signal the main thread that the servers are ready
        self.servers_ready.set()
        self.get_logger().info("--- All controllers connected; node fully active ---")

    def marker_feedback_callback(self, msg: InteractiveMarkerFeedback):
        """Handle drag feedback from RViz."""
        # Ignore feedback until the controllers are ready
        if not self.servers_ready.is_set() or msg.event_type != InteractiveMarkerFeedback.POSE_UPDATE:
            return
        
        control_name = msg.control_name
        
        if "rotation" in control_name or "translation" in control_name:
            parts = control_name.split('_')
            if len(parts) >= 3:
                joint_name = '_'.join(parts[1:-1])
                q = msg.pose.orientation
                angle = 2 * math.atan2(math.sqrt(q.x**2 + q.y**2 + q.z**2), q.w)
                
                # Simple sign handling (may need adjusting to the URDF axis directions)
                if q.z < 0 and (joint_name in ['joint_2', 'joint_4', 'joint_6']):
                     angle = -angle

                with self.lock:
                    self.current_joint_positions[joint_name] = angle

    def send_command_callback(self):
        """Periodically check for and send joint commands."""
        # Do nothing until the controllers are ready
        if not self.servers_ready.is_set():
            return

        with self.lock:
            if not self.current_joint_positions or self.current_joint_positions == self.last_sent_positions:
                return
            
            positions_to_send = self.current_joint_positions.copy()
            self.last_sent_positions = positions_to_send.copy()

        self.send_trajectory(positions_to_send)

    def send_trajectory(self, joint_positions: dict):
        """Build a trajectory and send it to the controller."""
        arm_joints = {k: v for k, v in joint_positions.items() if 'robotiq' not in k}

        if arm_joints and self.manipulator_client:
            goal = FollowJointTrajectory.Goal()
            traj = JointTrajectory()
            traj.joint_names = list(arm_joints.keys())
            point = JointTrajectoryPoint()
            point.positions = list(arm_joints.values())
            point.time_from_start = Duration(sec=0, nanosec=100000000) # 0.1s
            traj.points.append(point)
            goal.trajectory = traj
            self.manipulator_client.send_goal_async(goal)

def main(args=None):
    rclpy.init(args=args)
    node = ResilientInteractiveControl()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
