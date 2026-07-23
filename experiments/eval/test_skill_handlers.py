"""Unit tests for src/battery_dismantle_task/.../skill_handlers.py.

These are the pure-logic parts of the ROS2 package: waypoint/place
resolution and the rotateGripper/unscrew wrist-rotation math.

REQUIRES A ROS 2 ENVIRONMENT. skill_handlers.py does `from .scene_manager
import TABLE_Z, PLACED_OBJECT_DIMS`, and scene_manager.py imports rclpy,
moveit_msgs and tf2_ros at module level — so importing skill_handlers pulls
the whole ROS stack in transitively, even though the logic under test is
pure math. (An earlier version of this docstring claimed the module was
importable on a plain Python install; that stopped being true when the
scene_manager import was added, and the claim is corrected here.)

Two consequences, both handled below:
  1. It must be imported as `battery_dismantle_task.skill_handlers`, not as a
     top-level `skill_handlers` — the relative import needs package context.
  2. Without ROS 2 installed the whole module is skipped, not errored. A bare
     ImportError at collection time aborts the entire pytest session and takes
     the 37 pure-function tests in test_metrics.py / test_stats.py down with
     it; `importorskip` keeps `python -m pytest` green on a plain install.

CI runs test_metrics.py and test_stats.py explicitly (see .github/workflows/
ci.yml) and never reaches this file.

Run (needs ROS 2):  python -m pytest experiments/eval/test_skill_handlers.py -v
"""
import math
import sys
from pathlib import Path

import pytest

# scene_manager (imported transitively by skill_handlers) needs these.
pytest.importorskip("rclpy", reason="needs a ROS 2 install")
pytest.importorskip("moveit_msgs", reason="needs a ROS 2 install")

# Parent of the package dir, so `battery_dismantle_task` is importable AS a
# package and skill_handlers' `from .scene_manager import ...` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent /
                       "src" / "battery_dismantle_task"))

from battery_dismantle_task.skill_handlers import (  # noqa: E402
    SkillHandlers, WRIST_JOINT_INDEX, WRIST_JOINT_SOFT_LIMIT_RAD,
)

HOME_JOINTS = [0.0, 0.2618, 3.14159, -2.2689, 0.0, 0.9599, 1.5708]

WAYPOINTS = {
    "poses": {"HOME": HOME_JOINTS, "OPEN": [0.1], "CLOSE": [0.8]},
    "objects": {
        "TopCoverBolts": {
            "approach": [-0.0524, 0.7505, 0.0, 1.4486, -0.0175, 0.9948, -0.0524],
            "retreat": HOME_JOINTS,
            "place": [0.8203, 1.2392, 3.0369, -1.1345, 0.1396, -0.7679, 2.2515],
        },
    },
    "scene": {"bins": {"main": {"pose_name": "HOME"}}},
}


class _NullLogger:
    def info(self, *a, **k): pass
    def warn(self, *a, **k): pass
    def error(self, *a, **k): pass


class _DummyNode:
    """Stand-in for rclpy.node.Node — every handler method that logs calls
    self.node.get_logger(), so a real ROS2 node would be needed for a
    pure-None node. This swallows logging calls so the pure decision logic
    (which object is known, what the rotated angle is, ...) can be tested
    without a ROS2 runtime."""
    def get_logger(self):
        return _NullLogger()


def make_handlers():
    # motion_executor/scene_manager are unused by the pure-logic methods
    # under test (rotated_joints, _resolve_place_joints, the unknown-object
    # guards) — None is fine for those, the constructor just stores them.
    return SkillHandlers(node=_DummyNode(), waypoints=WAYPOINTS, motion_executor=None, scene_manager=None)


# ---------------------------------------------------------------- rotated_joints
def test_rotated_joints_only_changes_wrist():
    sh = make_handlers()
    out = sh.rotated_joints(HOME_JOINTS, 90)
    for i in range(7):
        if i == WRIST_JOINT_INDEX:
            assert out[i] != HOME_JOINTS[i]
        else:
            assert out[i] == HOME_JOINTS[i]


def test_rotated_joints_positive_angle_matches_radians():
    sh = make_handlers()
    out = sh.rotated_joints(HOME_JOINTS, 90)
    expected = HOME_JOINTS[WRIST_JOINT_INDEX] + math.radians(90)
    assert abs(out[WRIST_JOINT_INDEX] - expected) < 1e-9


def test_rotated_joints_does_not_mutate_input():
    sh = make_handlers()
    original = list(HOME_JOINTS)
    sh.rotated_joints(HOME_JOINTS, 45)
    assert HOME_JOINTS == original


def test_rotated_joints_clamped_to_soft_limit():
    """A bad/extreme angle from a hallucinating LLM must not command an
    unbounded wrist rotation — this is the safety guard added alongside the
    5 missing skill handlers (see REVISION_MEMO.md)."""
    sh = make_handlers()
    out = sh.rotated_joints(HOME_JOINTS, 100000)  # absurd angle
    assert out[WRIST_JOINT_INDEX] <= WRIST_JOINT_SOFT_LIMIT_RAD
    out2 = sh.rotated_joints(HOME_JOINTS, -100000)
    assert out2[WRIST_JOINT_INDEX] >= -WRIST_JOINT_SOFT_LIMIT_RAD


def test_rotated_joints_repeated_calls_compose():
    """execute_unscrew applies rotated_joints in a loop — confirm repeated
    application accumulates rather than resetting each time."""
    sh = make_handlers()
    j1 = sh.rotated_joints(HOME_JOINTS, -90)
    j2 = sh.rotated_joints(j1, -90)
    expected = HOME_JOINTS[WRIST_JOINT_INDEX] + math.radians(-180)
    assert abs(j2[WRIST_JOINT_INDEX] - expected) < 1e-9


# ---------------------------------------------------------------- _resolve_place_joints
def test_resolve_place_joints_from_explicit_pose_name():
    sh = make_handlers()
    out = sh._resolve_place_joints("TopCoverBolts", {"pose_name": "HOME"})
    assert out == HOME_JOINTS


def test_resolve_place_joints_from_bin():
    sh = make_handlers()
    out = sh._resolve_place_joints("TopCoverBolts", {"bin": "main"})
    assert out == HOME_JOINTS


def test_resolve_place_joints_falls_back_to_object_default():
    sh = make_handlers()
    out = sh._resolve_place_joints("TopCoverBolts", None)
    assert out == WAYPOINTS["objects"]["TopCoverBolts"]["place"]


def test_resolve_place_joints_unknown_object_and_no_place_is_none():
    sh = make_handlers()
    assert sh._resolve_place_joints("NoSuchObject", None) is None


# ---------------------------------------------------------------- unknown-object guards
# These mirror the dispatch_skill "Unknown skill" gap that was found and
# fixed: execute_unscrew/execute_disconnect must REJECT objects that have no
# waypoints.json entry rather than inventing coordinates for them (e.g. the
# "BMS connector" / "power connector" commands that were failing in practice).
def test_execute_unscrew_rejects_unknown_object():
    sh = make_handlers()
    assert sh.execute_unscrew("BMS connector") is False


def test_execute_disconnect_rejects_unknown_object():
    sh = make_handlers()
    assert sh.execute_disconnect("power connector") is False
