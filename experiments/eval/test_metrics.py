"""Unit tests for eval/metrics.py — pure functions, no I/O/ROS2/LLM needed.

This project had zero automated tests before this file (audited across both
src/battery_dismantle_task and src/llm_agent). metrics.py and stats.py are
the easiest, highest-value place to start: pure functions, the exact code
that produces every number in the dissertation's results tables.

Run:  python -m pytest experiments/eval/test_metrics.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import (
    plans_equal, step_prf, score_against_references, classify_failure, evaluate_trial,
)

MOVE_HOME = {"name": "moveTo", "params": {"target": "HOME"}}
GRASP_BOLTS = {"name": "grasp", "params": {"target": "TopCoverBolts"}}
RELEASE_BOLTS = {"name": "release", "params": {"target": "TopCoverBolts"}}

VALID_SKILLS = {"moveTo", "grasp", "release", "openGripper", "closeGripper",
                "inspect", "unscrew", "disconnect", "waitForStabilization", "rotateGripper"}
VALID_POSES = {"HOME", "OPEN", "CLOSE"}
VALID_OBJECTS = {"TopCoverBolts", "BatteryBox_0"}


# ---------------------------------------------------------------- plans_equal
def test_plans_equal_identical():
    assert plans_equal([MOVE_HOME], [MOVE_HOME])


def test_plans_equal_ignores_step_index():
    a = [{"step": 1, "name": "moveTo", "params": {"target": "HOME"}}]
    b = [{"step": 99, "name": "moveTo", "params": {"target": "HOME"}}]
    assert plans_equal(a, b)


def test_plans_equal_order_matters():
    assert not plans_equal([MOVE_HOME, GRASP_BOLTS], [GRASP_BOLTS, MOVE_HOME])


def test_plans_equal_both_empty():
    assert plans_equal([], [])


# ---------------------------------------------------------------- step_prf
def test_step_prf_perfect_match():
    m = step_prf([MOVE_HOME, GRASP_BOLTS], [MOVE_HOME, GRASP_BOLTS])
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0


def test_step_prf_empty_pred_nonempty_ref():
    """An empty prediction against a non-empty reference: recall=0 (missed
    everything) and precision=0 too — metrics.py's `1.0 if not r else 0.0`
    branch only fires when the REFERENCE is also empty (the "refuse is
    correct" case), not here. Both should be 0, not a vacuous 1.0."""
    m = step_prf([], [MOVE_HOME])
    assert m["recall"] == 0.0
    assert m["precision"] == 0.0


def test_step_prf_extra_hallucinated_step_penalises_precision():
    m = step_prf([MOVE_HOME, GRASP_BOLTS, RELEASE_BOLTS], [MOVE_HOME, GRASP_BOLTS])
    assert m["precision"] < 1.0
    assert m["recall"] == 1.0


def test_step_prf_both_empty_is_perfect():
    m = step_prf([], [])
    assert m["precision"] == 1.0 and m["recall"] == 1.0


# ---------------------------------------------------------------- score_against_references
def test_score_against_references_picks_best_of_multiple():
    refs = [[RELEASE_BOLTS], [MOVE_HOME, GRASP_BOLTS]]
    sc = score_against_references([MOVE_HOME, GRASP_BOLTS], refs)
    assert sc["exact"] is True
    assert sc["best_ref_index"] == 1


def test_score_against_references_empty_refs_means_refuse_is_correct():
    sc = score_against_references([], [])
    assert sc["exact"] is True


def test_score_against_references_nonempty_pred_against_refuse_reference():
    sc = score_against_references([MOVE_HOME], [])
    assert sc["exact"] is False


# ---------------------------------------------------------------- classify_failure
def test_classify_failure_exact_match_is_none():
    fail = classify_failure([MOVE_HOME], [[MOVE_HOME]], VALID_SKILLS, VALID_POSES, VALID_OBJECTS)
    assert fail is None


def test_classify_failure_correct_refusal_is_none():
    """An out-of-domain command with an empty prediction (= refused) must NOT
    be scored as a failure — this was bug #1 fixed per REVISION_MEMO.md
    section 3.4 ('correct rejections mis-scored as no_plan')."""
    fail = classify_failure([], [], VALID_SKILLS, VALID_POSES, VALID_OBJECTS,
                            command_is_out_of_domain=True)
    assert fail is None


def test_classify_failure_no_plan():
    fail = classify_failure([], [[MOVE_HOME]], VALID_SKILLS, VALID_POSES, VALID_OBJECTS)
    assert fail == "no_plan"


def test_classify_failure_hallucinated_skill():
    bad = {"name": "teleportTo", "params": {"target": "HOME"}}
    fail = classify_failure([bad], [[MOVE_HOME]], VALID_SKILLS, VALID_POSES, VALID_OBJECTS)
    assert fail == "hallucinated_skill"


def test_classify_failure_wrong_param_unknown_pose():
    bad = {"name": "moveTo", "params": {"target": "NOWHERE"}}
    fail = classify_failure([bad], [[MOVE_HOME]], VALID_SKILLS, VALID_POSES, VALID_OBJECTS)
    assert fail == "wrong_param"


def test_classify_failure_wrong_order():
    pred = [GRASP_BOLTS, MOVE_HOME]
    ref = [MOVE_HOME, GRASP_BOLTS]
    fail = classify_failure(pred, [ref], VALID_SKILLS, VALID_POSES, VALID_OBJECTS)
    assert fail == "wrong_order"


def test_classify_failure_out_of_domain_flag_when_schema_valid():
    fail = classify_failure([MOVE_HOME], [[]], VALID_SKILLS, VALID_POSES, VALID_OBJECTS,
                            command_is_out_of_domain=True)
    assert fail == "out_of_domain"


# ---------------------------------------------------------------- evaluate_trial
def test_evaluate_trial_correct():
    ev = evaluate_trial([MOVE_HOME], [[MOVE_HOME]], VALID_SKILLS, VALID_POSES, VALID_OBJECTS)
    assert ev["exact"] is True
    assert ev["is_correct"] is True
    assert ev["failure_mode"] is None


def test_evaluate_trial_incorrect_has_failure_mode_set():
    ev = evaluate_trial([], [[MOVE_HOME]], VALID_SKILLS, VALID_POSES, VALID_OBJECTS)
    assert ev["is_correct"] is False
    assert ev["failure_mode"] == "no_plan"
