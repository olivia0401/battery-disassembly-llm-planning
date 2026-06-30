#!/usr/bin/python3
"""Session data for robot skill execution context"""

class Session:
    """Stores context information for skill execution"""
    def __init__(self):
        self.object = ""
        self.place = {}  # dict with bin, pose_name, or joints
        self.plan_only = False
        self.vel_scale = 1.0
        self.acc_scale = 1.0
        # Last 7 joint positions the arm was successfully commanded to.
        # Used by rotateGripper (and unscrew, which calls it) as the base
        # pose to apply a wrist-joint delta to, since this project has no
        # direct joint-state feedback loop wired into skill dispatch.
        # None until the first successful moveTo/approach.
        self.last_arm_joints = None
