#!/usr/bin/env python3
"""
Complete working launch file for fake execution with INTERACTIVE CONTROL support.
This combines all necessary components in one file including ros2_control.

Key components:
1. Robot State Publisher
2. ROS2 Control Node (controller_manager) - NEW!
3. Joint State Broadcaster - NEW!
4. MoveIt move_group
5. RViz2
6. Interactive Joint Control Node - NEW!
7. Skill Server
8. Visual State Manager
"""

import os
import yaml
from pathlib import Path
import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, ExecuteProcess
from launch.substitutions import Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import xacro

def load_yaml(package_share, file_path):
    """Helper function to load a YAML file."""
    full_path = os.path.join(package_share, file_path)
    with open(full_path, 'r') as f:
        return yaml.safe_load(f)

def generate_launch_description():
    # --- Get Package Paths Using ament_index ---
    battery_dismantle_pkg_share = Path(get_package_share_directory("battery_dismantle_task"))
    kortex_description_pkg_share = Path(get_package_share_directory("kortex_description"))

    # --- File Paths ---
    # Use our custom xacro that includes fixed gripper macro
    xacro_file = str(battery_dismantle_pkg_share / "urdf" / "gen3_robotiq_2f_85.xacro")
    srdf_file = str(battery_dismantle_pkg_share / "config" / "moveit" / "gen3_robotiq_2f_85.srdf")
    rviz_file = str(battery_dismantle_pkg_share / "config" / "rviz" / "moveit.rviz")
    ros2_controllers_file = str(battery_dismantle_pkg_share / "config" / "moveit" / "ros2_controllers.yaml")

    # --- Load Configuration Files Manually ---
    # Using custom xacro with fixed gripper macro (no isaac_joint_commands compatibility issues)
    robot_description_config = xacro.process_file(
        xacro_file, mappings={
            "use_fake_hardware": "true",
        }
    )
    robot_description = {"robot_description": robot_description_config.toxml()}

    with open(srdf_file, "r") as f:
        robot_description_semantic_config = f.read()
    robot_description_semantic = {"robot_description_semantic": robot_description_semantic_config}

    kinematics_config = load_yaml(str(battery_dismantle_pkg_share), "config/moveit/kinematics.yaml")
    ompl_planning_config = load_yaml(str(battery_dismantle_pkg_share), "config/moveit/ompl_planning.yaml")
    moveit_controllers_config = load_yaml(str(battery_dismantle_pkg_share), "config/moveit/moveit_controllers.yaml")

    planning_pipelines = {
        "planning_pipelines": ["ompl"],
        "ompl": ompl_planning_config,
    }

    # CRITICAL: Add trajectory execution parameters for fake execution
    trajectory_execution = {
        "moveit_manage_controllers": True,
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.01,
    }

    # ========================================================================
    # NODES
    # ========================================================================

    # 1. Robot State Publisher - publishes robot TF tree
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )

    # 2. ROS2 Control Node (controller_manager) - CRITICAL FOR INTERACTIVE CONTROL!
    # This manages all ros2_control controllers and provides joint state feedback
    controller_manager_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, ros2_controllers_file],
        output="screen",
    )

    # 3. MoveIt move_group - motion planning node
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_config,
            planning_pipelines,
            trajectory_execution,
            moveit_controllers_config,
        ],
    )

    # 4. RViz2 - visualization
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_file],
        parameters=[
            robot_description,
            robot_description_semantic,
            planning_pipelines,
            kinematics_config,
        ],
    )

    # ========================================================================
    # TIMED ACTIONS - These start after delays to ensure proper initialization
    # ========================================================================

    # 5. Spawn joint_state_broadcaster - wait 1.5s for controller_manager
    # This broadcasts joint states from ros2_control to /joint_states topic
    spawn_joint_state_broadcaster = TimerAction(
        period=1.5,
        actions=[
            ExecuteProcess(
                cmd=["ros2", "control", "load_controller", "--set-state", "active", "joint_state_broadcaster"],
                output="screen"
            )
        ],
    )

    # 6. Spawn fake_manipulator_controller - wait 2.5s
    # This controller accepts joint trajectory commands for the arm
    spawn_manipulator_controller = TimerAction(
        period=2.5,
        actions=[
            ExecuteProcess(
                cmd=["ros2", "control", "load_controller", "--set-state", "active", "fake_manipulator_controller"],
                output="screen"
            )
        ],
    )

    # 7. Spawn fake_gripper_controller - wait 3.0s
    # This controller accepts joint commands for the gripper
    spawn_gripper_controller = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=["ros2", "control", "load_controller", "--set-state", "active", "fake_gripper_controller"],
                output="screen"
            )
        ],
    )

    # 8. Visual State Manager - wait 4.0s for move_group to be ready
    visual_state_manager_node = TimerAction(
        period=4.0,
        actions=[
            Node(
                package="battery_dismantle_task",
                executable="visual_state_manager_node",
                name="visual_state_manager",
                output="screen",
                parameters=[{"use_sim_time": False}]
            )
        ]
    )

    # 9. Interactive Joint Control Node - wait 5.0s for all controllers to be active
    # THIS IS THE KEY NODE FOR INTERACTIVE CONTROL IN RVIZ!
    # It listens to RViz interactive markers and sends commands to ros2_control
    interactive_control_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package="battery_dismantle_task",
                executable="interactive_joint_control_node",
                name="interactive_joint_control",
                output="screen",
                parameters=[{"use_sim_time": False}]
            )
        ]
    )

    # 10. Skill Server - wait 10.0s for everything to be ready
    waypoints_file = str(battery_dismantle_pkg_share / "config" / "waypoints.json")
    skill_server_params = {
        "waypoints_path": waypoints_file,
        "manipulator_group": "manipulator",
        "gripper_group": "gripper",
        "ee_attach_link": "robotiq_85_base_link",
        "vel_scale": 0.8,
        "acc_scale": 0.8,
        "scene_update_wait_s": 1.0,
        "open_gripper_pose_name": "OPEN",
        "close_gripper_pose_name": "CLOSE",
        "use_sim_time": False,
    }

    skill_server_node = TimerAction(
        period=10.0,
        actions=[
            Node(
                package="battery_dismantle_task",
                executable="skill_server_node",
                name="skill_server",
                output="screen",
                parameters=[
                    robot_description,
                    robot_description_semantic,
                    kinematics_config,
                    skill_server_params,
                ],
            )
        ],
    )

    # ========================================================================
    # LAUNCH DESCRIPTION
    # ========================================================================
    return LaunchDescription([
        # Core nodes (start immediately)
        robot_state_publisher_node,
        controller_manager_node,  # NEW!
        move_group_node,
        rviz_node,

        # Timed actions (start with delays)
        spawn_joint_state_broadcaster,  # NEW! (1.5s)
        spawn_manipulator_controller,    # NEW! (2.5s)
        spawn_gripper_controller,        # NEW! (3.0s)
        visual_state_manager_node,       # (4.0s)
        interactive_control_node,        # NEW! (5.0s) - CRITICAL FOR RVIZ INTERACTION
        skill_server_node,               # (10.0s)
    ])
