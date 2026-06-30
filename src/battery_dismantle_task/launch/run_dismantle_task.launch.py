#!/usr/bin/env python3
"""
Final, definitive, and correct launch file for fake execution.
This version manually loads all parameters to bypass the MoveItConfigsBuilder bug.
This file ONLY launches MoveIt, RViz, and core scene components.
Controllers and interactive nodes are launched SEPARATELY.
"""

import os
import yaml
from pathlib import Path
import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
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
    # --- Dynamic Package Paths (Fixed from hardcoded) ---
    battery_dismantle_pkg_share = Path(get_package_share_directory('battery_dismantle_task'))
    kortex_description_pkg_share = Path(get_package_share_directory('kortex_description'))

    # --- File Paths ---
    xacro_file = str(kortex_description_pkg_share / "robots" / "gen3.xacro")
    srdf_file = str(battery_dismantle_pkg_share / "config" / "moveit" / "gen3_robotiq_2f_85.srdf")
    rviz_file = str(battery_dismantle_pkg_share / "config" / "rviz" / "moveit.rviz")
    
    # --- Load Configuration Files Manually ---
    robot_description_config = xacro.process_file(
        xacro_file, mappings={"dof": "7", "gripper": "robotiq_2f_85", "use_fake_hardware": "true"}
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

    # --- Nodes ---
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_config,
            planning_pipelines,
            moveit_controllers_config,
            {"moveit_manage_controllers": False},
        ],
    )
    
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
    
    waypoints_file = str(battery_dismantle_pkg_share / "config" / "waypoints.json")
    skill_server_params = {
        "waypoints_path": waypoints_file, "manipulator_group": "manipulator", "gripper_group": "gripper",
        "ee_attach_link": "robotiq_85_base_link", "vel_scale": 0.8, "acc_scale": 0.8,
        "scene_update_wait_s": 1.0, "open_gripper_pose_name": "OPEN", "close_gripper_pose_name": "CLOSE",
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

    return LaunchDescription([
        robot_state_publisher_node,
        move_group_node,
        rviz_node,
        skill_server_node,
        visual_state_manager_node,
    ])