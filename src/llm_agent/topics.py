#!/usr/bin/env python3
"""
ROS2 topic name constants.

This module centralizes all topic names used across the LLM-ROS2 system
to avoid hardcoded strings and make refactoring easier.
"""

# LLM Communication Topics
LLM_COMMANDS = '/llm_commands'
LLM_FEEDBACK = '/llm_feedback'

# Robot State Topics
JOINT_STATES = '/joint_states'
ROBOT_DESCRIPTION = '/robot_description'

# MoveIt Topics
DISPLAY_PLANNED_PATH = '/display_planned_path'
PLANNING_SCENE = '/planning_scene'
PLANNING_SCENE_WORLD = '/planning_scene_world'

# Interactive Marker Topics
INTERACTIVE_MARKER_FEEDBACK = '/rviz_moveit_motion_planning_display/robot_interaction_interactive_marker_topic/feedback'

# Service Names
APPLY_PLANNING_SCENE = '/apply_planning_scene'
GET_PLANNING_SCENE = '/get_planning_scene'

# Action Names
MOVE_GROUP_ACTION = '/move_action'
MANIPULATOR_FOLLOW_JOINT_TRAJECTORY = '/fake_manipulator_controller/follow_joint_trajectory'
GRIPPER_FOLLOW_JOINT_TRAJECTORY = '/fake_gripper_controller/follow_joint_trajectory'
