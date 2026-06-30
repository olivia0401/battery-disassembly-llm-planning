"""Unit tests for eval/stats.py — pure functions, no I/O/ROS2/LLM needed.

Run:  python -m pytest experiments/eval/test_stats.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
from eval.stats import wilson_ci, mcnemar_from_pairs, noise_floor, holm_bonferroni, cohens_kappa


# ---------------------------------------------------------------- wilson_ci
def test_wilson_ci_50_50():
    c = wilson_ci(5, 10)
    assert c["p"] == 0.5
    assert c["lo"] < 0.5 < c["hi"]


def test_wilson_ci_zero_n_is_nan():
    c = wilson_ci(0, 0)
    assert math.isnan(c["p"])


def test_wilson_ci_all_success_caps_at_1_but_lower_bound_stays_honest():
    """At k=n the Wilson upper bound is explicitly clamped to 1.0 (see
    wilson_ci's `min(1.0, ...)`), but the LOWER bound still reflects "we
    can't be certain" — for small n it stays well below 1.0 rather than
    collapsing to a point estimate, unlike a naive normal-approx CI which
    degenerates to zero width at p=0 or p=1."""
    c = wilson_ci(10, 10)
    assert c["p"] == 1.0
    assert c["hi"] == 1.0
    assert c["lo"] < 1.0


def test_wilson_ci_narrows_with_more_data():
    small = wilson_ci(5, 10)
    big = wilson_ci(500, 1000)
    assert (big["hi"] - big["lo"]) < (small["hi"] - small["lo"])


# ---------------------------------------------------------------- mcnemar_from_pairs
def test_mcnemar_no_discordant_pairs_is_p1():
    a = [True, True, False, False]
    b = [True, True, False, False]
    r = mcnemar_from_pairs(a, b)
    assert r["p_value"] == 1.0
    assert r["significant"] is False


def test_mcnemar_all_discordant_one_direction_is_significant():
    # A correct, B wrong on every single one of 20 paired items
    a = [True] * 20
    b = [False] * 20
    r = mcnemar_from_pairs(a, b)
    assert r["b"] == 20 and r["c"] == 0
    assert r["p_value"] < 0.05
    assert r["significant"] is True


def test_mcnemar_length_mismatch_raises():
    try:
        mcnemar_from_pairs([True], [True, False])
        assert False, "expected AssertionError on length mismatch"
    except AssertionError:
        pass


# ---------------------------------------------------------------- noise_floor
def test_noise_floor_single_run_is_nan():
    r = noise_floor([0.5])
    assert math.isnan(r["std"])


def test_noise_floor_identical_runs_zero_std():
    r = noise_floor([0.4, 0.4, 0.4])
    assert r["std"] < 1e-9   # float arithmetic, not exactly 0.0
    assert r["band"] < 1e-9


def test_noise_floor_band_is_2x_std():
    r = noise_floor([0.3, 0.5])
    assert abs(r["band"] - 2 * r["std"]) < 1e-9


# ---------------------------------------------------------------- holm_bonferroni
def test_holm_bonferroni_single_comparison_matches_raw():
    out = holm_bonferroni({"a_vs_b": 0.03})
    assert abs(out["a_vs_b"]["p_corrected"] - 0.03) < 1e-9
    assert out["a_vs_b"]["significant"] is True


def test_holm_bonferroni_inflates_p_for_multiple_comparisons():
    """The whole point of Holm correction: three comparisons each at raw
    p=0.03 should NOT all stay significant at alpha=0.05 — at least the
    largest must be corrected upward past the threshold for some, depending
    on rank, but the smallest p must always be >= its raw value."""
    raw = {"a": 0.03, "b": 0.03, "c": 0.03}
    out = holm_bonferroni(raw)
    for k in raw:
        assert out[k]["p_corrected"] >= out[k]["p_raw"]
    # smallest-ranked p gets multiplied by m=3 -> 0.09, no longer significant
    assert all(not v["significant"] for v in out.values())


def test_holm_bonferroni_monotone_nondecreasing_by_rank():
    raw = {"a": 0.01, "b": 0.02, "c": 0.04}
    out = holm_bonferroni(raw)
    ordered = sorted(out.values(), key=lambda v: v["p_raw"])
    corrected = [v["p_corrected"] for v in ordered]
    assert corrected == sorted(corrected)  # step-down correction is monotone


def test_holm_bonferroni_caps_at_one():
    out = holm_bonferroni({"a": 0.9, "b": 0.9, "c": 0.9})
    assert all(v["p_corrected"] <= 1.0 for v in out.values())


# ---------------------------------------------------------------- cohens_kappa
def test_cohens_kappa_perfect_agreement():
    r = cohens_kappa(["Y", "N", "Y", "N"], ["Y", "N", "Y", "N"])
    assert r["kappa"] == 1.0


def test_cohens_kappa_no_agreement_beyond_chance_near_zero():
    # Two raters with identical marginal label frequencies but uncorrelated
    # assignments should land near kappa=0, not strongly positive.
    a = ["Y", "Y", "N", "N", "Y", "N", "Y", "N"]
    b = ["N", "Y", "Y", "N", "N", "Y", "Y", "N"]
    r = cohens_kappa(a, b)
    assert -1.0 <= r["kappa"] <= 1.0


def test_cohens_kappa_empty_inputs():
    r = cohens_kappa([], [])
    assert r["n"] == 0
