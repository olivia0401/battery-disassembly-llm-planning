#!/usr/bin/env python3
"""
Motion execution configuration constants.

This module centralizes all magic numbers and configuration parameters
related to robot motion planning and execution.
"""

# Velocity and Acceleration Scaling
DEFAULT_VELOCITY_SCALE = 0.8  # 80% of maximum velocity for safety
DEFAULT_ACCELERATION_SCALE = 0.8  # 80% of maximum acceleration

# Joint Tolerances
JOINT_POSITION_TOLERANCE = 0.01  # radians (~0.57 degrees)
JOINT_VELOCITY_TOLERANCE = 0.0  # rad/s (zero for position-only control)

# Trajectory Timing
VISUALIZATION_SLOW_FACTOR = 3.0  # Slow down 3x for better RViz visualization
MIN_TRAJECTORY_DURATION = 0.5  # Minimum duration for any trajectory (seconds)
DEFAULT_TRAJECTORY_DURATION = 3.0  # Default duration when not computed

# Planning Timeouts
PLANNING_TIMEOUT = 10.0  # Maximum time to wait for motion planning (seconds)
EXECUTION_TIMEOUT = 30.0  # Maximum time to wait for trajectory execution (seconds)
STATE_UPDATE_TIMEOUT = 5.0  # Timeout for waiting for current robot state (seconds)

# MoveGroup Configuration
PLANNING_ATTEMPTS = 5  # Number of planning attempts before giving up
PLANNING_TIME = 5.0  # Time allowed for each planning attempt (seconds)
GOAL_TOLERANCE = 0.01  # Goal position tolerance (radians)

# Controller Names
MANIPULATOR_CONTROLLER = 'fake_manipulator_controller'
GRIPPER_CONTROLLER = 'fake_gripper_controller'

# Planning Group Names
ARM_PLANNING_GROUP = 'arm'
GRIPPER_PLANNING_GROUP = 'gripper'

# End Effector Link
END_EFFECTOR_LINK = 'end_effector_link'
