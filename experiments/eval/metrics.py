"""
Plan-quality metrics for LLM-planning evaluation.

Pure functions, no I/O, no ROS, no network. Everything here is unit-testable.

Core ideas
----------
- A plan is a list of steps; a step is {"name": <skill>, "params": {...}}.
- We compare a predicted plan against a SET of acceptable reference plans
  (multiple correct solutions are allowed). The best-matching reference is used.
- Exact (task_correct): predicted == any acceptable reference (ordered, exact).
- Step-level Precision / Recall / F1: ordered LCS over (name, params) tuples.
- Failure-mode classification: one of 7 categories, by P/R shape + schema checks.
"""

from __future__ import annotations
from typing import Dict, List, Any, Tuple, Optional, Iterable

# ----------------------------------------------------------------------------
# Step normalisation / equality
# ----------------------------------------------------------------------------

def _norm_step(step: Dict[str, Any]) -> Tuple[str, Tuple[Tuple[str, Any], ...]]:
    """Canonical, hashable form of a step: (name, sorted(params items)).

    The 'step'/index field is ignored so step numbering never affects equality.
    """
    name = step.get("name", "")
    params = step.get("params", {}) or {}
    items = tuple(sorted((str(k), params[k]) for k in params))
    return (name, items)


def _norm_plan(plan: List[Dict[str, Any]]) -> List[Tuple[str, Tuple]]:
    return [_norm_step(s) for s in (plan or [])]


def plans_equal(a: List[Dict], b: List[Dict]) -> bool:
    """Ordered exact equality of two plans (ignoring step indices)."""
    return _norm_plan(a) == _norm_plan(b)


# ----------------------------------------------------------------------------
# Ordered LCS  -> step-level precision / recall / f1
# ----------------------------------------------------------------------------

def _lcs_len(x: List[Any], y: List[Any]) -> int:
    """Length of the longest common subsequence (ORDER preserved)."""
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return 0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        xi = x[i - 1]
        for j in range(1, m + 1):
            tmp = dp[j]
            dp[j] = prev + 1 if xi == y[j - 1] else max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]


def step_prf(pred: List[Dict], ref: List[Dict]) -> Dict[str, float]:
    """Step-level precision/recall/f1 of `pred` against a single `ref` via LCS.

    Precision penalises extra/hallucinated steps; Recall penalises missing steps.
    """
    p = _norm_plan(pred)
    r = _norm_plan(ref)
    match = _lcs_len(p, r)
    precision = match / len(p) if p else (1.0 if not r else 0.0)
    recall = match / len(r) if r else (1.0 if not p else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "match": match,
            "len_pred": len(p), "len_ref": len(r)}


# ----------------------------------------------------------------------------
# Multi-reference scoring
# ----------------------------------------------------------------------------

def score_against_references(pred: List[Dict],
                             references: List[List[Dict]]) -> Dict[str, Any]:
    """Score a predicted plan against a SET of acceptable reference plans.

    Returns the metrics for the BEST-matching reference (highest F1, then exact).
    `exact` is True if pred equals ANY reference exactly.
    """
    # An empty reference set means "the correct behaviour is to refuse"
    # (produce an empty plan). Model it as a single acceptable plan: [].
    if not references:
        references = [[]]

    exact = any(plans_equal(pred, ref) for ref in references)
    best = None
    best_idx = -1
    for i, ref in enumerate(references):
        m = step_prf(pred, ref)
        if best is None or m["f1"] > best["f1"]:
            best = m
            best_idx = i
    best = dict(best)
    best["exact"] = exact
    best["best_ref_index"] = best_idx
    return best


# ----------------------------------------------------------------------------
# Failure-mode classification (7 categories)
# ----------------------------------------------------------------------------

FAILURE_MODES = [
    "no_plan", "hallucinated_skill", "wrong_param", "wrong_order",
    "wrong_object", "out_of_domain", "over_decomposition",
]

_GRIPPER_SKILLS = {"openGripper", "closeGripper", "waitForStabilization"}


def classify_failure(pred: List[Dict],
                     references: List[List[Dict]],
                     valid_skills: Iterable[str],
                     valid_poses: Iterable[str],
                     valid_objects: Iterable[str],
                     command_is_out_of_domain: bool = False) -> Optional[str]:
    """Return a failure-mode label, or None if the plan is correct (exact match).

    Priority: schema faults (hallucination, bad params) are reported before
    sequence/structure faults, because they are the proximate cause.
    """
    valid_skills = set(valid_skills)
    valid_poses = set(valid_poses)
    valid_objects = set(valid_objects)

    # empty references == "refuse is correct" -> model as the single plan []
    refs_eff = references if references else [[]]

    # exact match (including a correct refusal: empty pred vs empty ref) -> not a failure
    if any(plans_equal(pred, ref) for ref in refs_eff):
        return None

    if not pred:
        return "no_plan"

    # schema-level faults first
    for s in pred:
        if s.get("name") not in valid_skills:
            return "hallucinated_skill"
    for s in pred:
        name = s.get("name")
        params = s.get("params", {}) or {}
        if name in _GRIPPER_SKILLS:
            continue
        if name == "rotateGripper":
            if "angle" not in params:
                return "wrong_param"
            continue
        # skills that need a target
        target = params.get("target")
        if target is None:
            return "wrong_param"
        if name == "moveTo" and target not in valid_poses:
            return "wrong_param"
        if name in ("grasp", "release", "inspect", "unscrew", "disconnect") \
                and target not in valid_objects:
            return "wrong_param"

    if command_is_out_of_domain:
        return "out_of_domain"

    # structure-level: compare to best reference by set / order
    best = score_against_references(pred, references)
    p = _norm_plan(pred)
    refs_norm = [_norm_plan(r) for r in references]

    # same multiset of steps but wrong order
    for r in refs_norm:
        if sorted(p) == sorted(r) and p != r:
            return "wrong_order"

    # extra steps, all reference steps covered -> over-decomposition
    if best["recall"] >= 0.999 and best["len_pred"] > best["len_ref"]:
        return "over_decomposition"

    # a grasp/release targets a valid-but-different object than the reference
    if refs_norm:
        ref = refs_norm[best["best_ref_index"]] if best["best_ref_index"] >= 0 else refs_norm[0]
        ref_objs = {it for (nm, items) in ref if nm in ("grasp", "release")
                    for (k, it) in items if k == "target"}
        pred_objs = {it for (nm, items) in p if nm in ("grasp", "release")
                     for (k, it) in items if k == "target"}
        if pred_objs and pred_objs != ref_objs and pred_objs <= valid_objects:
            return "wrong_object"

    # fallback: missing/replaced steps -> treat as wrong_order bucket
    return "wrong_order"


# ----------------------------------------------------------------------------
# Convenience: full per-trial evaluation
# ----------------------------------------------------------------------------

def evaluate_trial(pred: List[Dict],
                   references: List[List[Dict]],
                   valid_skills: Iterable[str],
                   valid_poses: Iterable[str],
                   valid_objects: Iterable[str],
                   command_is_out_of_domain: bool = False) -> Dict[str, Any]:
    """One-call evaluation producing every plan-quality field for a trial row."""
    sc = score_against_references(pred, references)
    fail = classify_failure(pred, references, valid_skills, valid_poses,
                            valid_objects, command_is_out_of_domain)
    return {
        "exact": bool(sc["exact"]),
        "step_precision": round(sc["precision"], 4),
        "step_recall": round(sc["recall"], 4),
        "step_f1": round(sc["f1"], 4),
        "plan_length": len(pred or []),
        "failure_mode": fail,          # None == correct
        "is_correct": fail is None,
    }
