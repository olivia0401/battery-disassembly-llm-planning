#!/usr/bin/env python3
"""
Interactive Joint Control - 拖动关节即实时移动 (最终弹性连接版)
本节点会持续尝试连接控制器，直到成功为止，以解决时序/竞速问题。
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
        self.get_logger().info("--- 最终弹性连接版交互控制节点 ---")

        # Action clients 会在后台线程中初始化
        self.manipulator_client = None
        self.gripper_client = None
        self.servers_ready = threading.Event() # 用于发出“服务器已就绪”信号的事件

        # 初始化变量
        self.current_joint_positions = {}
        self.last_sent_positions = {}
        self.lock = threading.Lock()

        # 启动一个后台线程，专门负责等待并连接到 Action Servers
        self.connection_thread = threading.Thread(target=self.wait_for_servers, daemon=True)
        self.connection_thread.start()

        # 订阅 RViz 的 feedback 话题
        # 回调函数会在发送指令前检查服务器是否就绪
        feedback_topic = '/rviz_moveit_motion_planning_display/robot_interaction_interactive_marker_topic/feedback'
        self.subscription = self.create_subscription(
            InteractiveMarkerFeedback,
            feedback_topic,
            self.marker_feedback_callback,
            10
        )
        
        # 启动定时器，定期发送关节目标
        self.timer = self.create_timer(0.1, self.send_command_callback)
        self.get_logger().info("节点已启动，正在后台等待控制器连接...")

    def wait_for_servers(self):
        """在后台线程中，无限期地等待 Action Servers"""
        manipulator_action_server = '/fake_manipulator_controller/follow_joint_trajectory'
        gripper_action_server = '/fake_gripper_controller/follow_joint_trajectory'
        
        self.get_logger().info(f"[连接线程] 正在等待机械臂 Action Server: {manipulator_action_server}...")
        temp_manipulator_client = ActionClient(self, FollowJointTrajectory, manipulator_action_server)
        temp_manipulator_client.wait_for_server()
        self.manipulator_client = temp_manipulator_client
        self.get_logger().info("✅ [连接线程] 机械臂控制器已连接！")
        
        self.get_logger().info(f"[连接线程] 正在等待夹爪 Action Server: {gripper_action_server}...")
        temp_gripper_client = ActionClient(self, FollowJointTrajectory, gripper_action_server)
        temp_gripper_client.wait_for_server()
        self.gripper_client = temp_gripper_client
        self.get_logger().info("✅ [连接线程] 夹爪控制器已连接！")
        
        # 发出信号，告诉主线程服务器已准备就绪
        self.servers_ready.set()
        self.get_logger().info("--- 所有控制器已连接，节点功能已完全激活 ---")

    def marker_feedback_callback(self, msg: InteractiveMarkerFeedback):
        """处理来自 RViz 的拖动反馈"""
        # 如果控制器还未就绪，则忽略所有反馈
        if not self.servers_ready.is_set() or msg.event_type != InteractiveMarkerFeedback.POSE_UPDATE:
            return
        
        control_name = msg.control_name
        
        if "rotation" in control_name or "translation" in control_name:
            parts = control_name.split('_')
            if len(parts) >= 3:
                joint_name = '_'.join(parts[1:-1])
                q = msg.pose.orientation
                angle = 2 * math.atan2(math.sqrt(q.x**2 + q.y**2 + q.z**2), q.w)
                
                # 简单的符号处理（可能需要根据 URDF 轴向调整）
                if q.z < 0 and (joint_name in ['joint_2', 'joint_4', 'joint_6']):
                     angle = -angle

                with self.lock:
                    self.current_joint_positions[joint_name] = angle

    def send_command_callback(self):
        """定时检查并发送关节命令"""
        # 如果控制器还未就绪，则什么都不做
        if not self.servers_ready.is_set():
            return

        with self.lock:
            if not self.current_joint_positions or self.current_joint_positions == self.last_sent_positions:
                return
            
            positions_to_send = self.current_joint_positions.copy()
            self.last_sent_positions = positions_to_send.copy()

        self.send_trajectory(positions_to_send)

    def send_trajectory(self, joint_positions: dict):
        """构建并发送轨迹给控制器"""
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
