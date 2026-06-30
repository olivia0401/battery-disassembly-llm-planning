#!/usr/bin/env python3
"""
Publish initial joint states for fake hardware mode.
This gives MoveIt an initial state to start from.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Time


class InitialJointStatePublisher(Node):
    def __init__(self):
        super().__init__('initial_joint_state_publisher')

        self.publisher = self.create_publisher(JointState, 'joint_states', 10)

        # Publish initial state at 10Hz for better state tracking
        # Higher frequency ensures MoveIt and RViz always have recent joint states
        self.timer = self.create_timer(0.1, self.publish_joint_states)

        # Safe initial joint positions (non-colliding)
        # Using SRDF HOME position (collision-free)
        self.joint_state = JointState()
        self.joint_state.name = [
            # Arm joints
            'joint_1', 'joint_2', 'joint_3', 'joint_4',
            'joint_5', 'joint_6', 'joint_7',
            # Gripper main joint (mimic joints follow automatically)
            'robotiq_85_left_knuckle_joint'
        ]
        # Safe home position - matches waypoints.json HOME (collision-free)
        # Gripper: slightly open to avoid internal geometry collision
        self.joint_state.position = [
            # Arm joints - exact values from waypoints.json HOME (collision-free)
            0.0,        # joint_1
            0.2618,     # joint_2
            3.14159,    # joint_3
            -2.2689,    # joint_4
            0.0,        # joint_5
            0.9599,     # joint_6
            1.5708,     # joint_7
            # Gripper main joint - OPEN position from waypoints.json
            0.1         # robotiq_85_left_knuckle_joint (OPEN)
        ]

        self.get_logger().info('Publishing joint states at 10Hz (7 arm + 1 gripper joint, OPEN)...')
        self.get_logger().info('Running continuously to provide joint_states for fake hardware mode')

    def publish_joint_states(self):
        # Publish joint states continuously
        self.joint_state.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.joint_state)


def main(args=None):
    rclpy.init(args=args)
    node = InitialJointStatePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
