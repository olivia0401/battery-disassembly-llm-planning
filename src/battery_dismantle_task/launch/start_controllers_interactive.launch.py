#!/usr/bin/env python3
"""
Secondary launch file to start controllers and interactive node.
This should be launched *after* the main run_dismantle_task.launch.py is fully started.
"""

import os
import yaml
from pathlib import Path
from launch import LaunchDescription
from launch.actions import TimerAction, ExecuteProcess
from launch.substitutions import Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # --- Dynamic Package Paths (Fixed from hardcoded) ---
    battery_dismantle_share = Path(get_package_share_directory('battery_dismantle_task'))
    kortex_description_share = Path(get_package_share_directory('kortex_description'))

    # --- File Paths ---
    ros2_controllers_file = str(battery_dismantle_share / "config" / "moveit" / "ros2_controllers.yaml")

    # --- We need robot_description to start ros2_control_node ---
    # This is a simplified version, assuming the main launch file has already set this up.
    # In a real robust scenario, you might read this from a shared config or topic.
    robot_description_content = {"robot_description": Command(["xacro ", str(kortex_description_share / "robots" / "gen3.xacro"), " gripper:=robotiq_2f_85", " dof:=7", " use_fake_hardware:=true"])}

    # --- Nodes ---
    
    # Controller Manager for fake hardware
    controller_manager_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description_content, ros2_controllers_file],
        output="screen",
    )

    # Spawn controllers after controller_manager starts
    # Joint state broadcaster MUST be loaded first
    spawn_joint_state_broadcaster = TimerAction(
        period=1.5,
        actions=[
            ExecuteProcess(
                cmd=["ros2", "control", "load_controller", "--set-state", "active", "joint_state_broadcaster"],
                output="screen"
            )
        ],
    )

    spawn_manipulator_controller = TimerAction(
        period=2.5,
        actions=[
            ExecuteProcess(
                cmd=["ros2", "control", "load_controller", "--set-state", "active", "fake_manipulator_controller"],
                output="screen"
            )
        ],
    )

    spawn_gripper_controller = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=["ros2", "control", "load_controller", "--set-state", "active", "fake_gripper_controller"],
                output="screen"
            )
        ],
    )

    # Interactive joint control
    interactive_control_node = TimerAction(
        period=5.0, # Give controllers time to be fully active
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

    return LaunchDescription([
        controller_manager_node,
        spawn_joint_state_broadcaster,
        spawn_manipulator_controller,
        spawn_gripper_controller,
        interactive_control_node,
    ])

# Helper to get Command since we are in a different file
from launch.substitutions import Command
