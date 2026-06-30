#!/usr/bin/env python3
"""
Scene object definitions and dimensions.

This module defines the physical properties and positions of all objects
in the battery disassembly workspace.
"""
from dataclasses import dataclass
from typing import List


@dataclass
class ObjectDimensions:
    """Physical dimensions of an object in meters."""
    width: float
    depth: float
    height: float

    def as_list(self) -> List[float]:
        """Return dimensions as [x, y, z] list."""
        return [self.width, self.depth, self.height]


@dataclass
class ObjectPose:
    """Position and orientation of an object."""
    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


# Battery Base Position
BATTERY_BASE_POSITION = ObjectPose(
    x=0.45,  # meters in front of robot
    y=0.0,   # centered
    z=0.0    # on table surface
)

# Object Dimensions
BATTERY_DIMENSIONS = ObjectDimensions(
    width=0.36,   # x-axis
    depth=0.26,   # y-axis
    height=0.08   # z-axis (vertical)
)

TOP_COVER_DIMENSIONS = ObjectDimensions(
    width=0.36,
    depth=0.26,
    height=0.01  # thin cover plate
)

BOLT_DIMENSIONS = ObjectDimensions(
    width=0.02,   # small bolt
    depth=0.02,
    height=0.015
)

# Derived Positions
BATTERY_HEIGHT = BATTERY_DIMENSIONS.height

# Top cover sits on top of battery
TOP_COVER_POSITION = ObjectPose(
    x=BATTERY_BASE_POSITION.x,
    y=BATTERY_BASE_POSITION.y,
    z=BATTERY_BASE_POSITION.z + BATTERY_HEIGHT + TOP_COVER_DIMENSIONS.height / 2
)

# Bolt positions (4 corners of top cover)
BOLT_OFFSET_X = 0.15  # Distance from center to bolt in X direction
BOLT_OFFSET_Y = 0.10  # Distance from center to bolt in Y direction

BOLT_POSITIONS = [
    ObjectPose(  # Front-left bolt
        x=TOP_COVER_POSITION.x + BOLT_OFFSET_X,
        y=TOP_COVER_POSITION.y + BOLT_OFFSET_Y,
        z=TOP_COVER_POSITION.z + TOP_COVER_DIMENSIONS.height / 2 + BOLT_DIMENSIONS.height / 2
    ),
    ObjectPose(  # Front-right bolt
        x=TOP_COVER_POSITION.x + BOLT_OFFSET_X,
        y=TOP_COVER_POSITION.y - BOLT_OFFSET_Y,
        z=TOP_COVER_POSITION.z + TOP_COVER_DIMENSIONS.height / 2 + BOLT_DIMENSIONS.height / 2
    ),
    ObjectPose(  # Back-left bolt
        x=TOP_COVER_POSITION.x - BOLT_OFFSET_X,
        y=TOP_COVER_POSITION.y + BOLT_OFFSET_Y,
        z=TOP_COVER_POSITION.z + TOP_COVER_DIMENSIONS.height / 2 + BOLT_DIMENSIONS.height / 2
    ),
    ObjectPose(  # Back-right bolt
        x=TOP_COVER_POSITION.x - BOLT_OFFSET_X,
        y=TOP_COVER_POSITION.y - BOLT_OFFSET_Y,
        z=TOP_COVER_POSITION.z + TOP_COVER_DIMENSIONS.height / 2 + BOLT_DIMENSIONS.height / 2
    ),
]

# Collection Bin Position (for placing removed components)
COLLECTION_BIN_POSITION = ObjectPose(
    x=0.3,
    y=0.4,
    z=0.1
)
