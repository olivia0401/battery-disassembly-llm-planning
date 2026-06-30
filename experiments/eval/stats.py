"""
Statistical helpers for rigorous reporting.

Pure functions over plain numbers / lists. Uses scipy where helpful but
Wilson CI and McNemar are implemented directly so results are transparent.

Provided
--------
- wilson_ci(k, n)            : 95% CI for a proportion (small-sample safe)
- mcnemar_from_pairs(a, b)   : paired binary comparison (two configs, same items)
- anova_tukey(groups)        : one-way ANOVA + Tukey HSD for continuous (latency)
- noise_floor(per_run_rates) : run-to-run std of a proportion (the "equivalence" band)
- ci_overlap(ci1, ci2)       : True if two CIs overlap (-> declare "equivalent")
- fmt_pct_ci(k, n)           : "92.4% [88.1, 95.3]" ready for a table cell
"""

from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
import math

try:
    from scipy import stats as _sps
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ----------------------------------------------------------------------------
# Wilson score interval for a binomial proportion
# ----------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> Dict[str, float]:
    """95% Wilson confidence interval for k successes out of n.

    Much better than the normal approximation when n is small or p near 0/1.
    Returns proportion p plus lower/upper bounds (all in [0, 1]).
    """
    if n <= 0:
        return {"p": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0, "k": 0}
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)) / denom
    return {"p": p, "lo": max(0.0, centre - half), "hi": min(1.0, centre + half),
            "n": n, "k": k}


def fmt_pct_ci(k: int, n: int) -> str:
    """Format as '92.4% [88.1, 95.3]' for a table cell."""
    c = wilson_ci(k, n)
    if n == 0:
        return "n/a"
    return f"{100*c['p']:.1f}% [{100*c['lo']:.1f}, {100*c['hi']:.1f}]"


def ci_overlap(ci1: Dict[str, float], ci2: Dict[str, float]) -> bool:
    """True if two CIs overlap -> the difference is not clearly significant."""
    return not (ci1["hi"] < ci2["lo"] or ci2["hi"] < ci1["lo"])


# ----------------------------------------------------------------------------
# McNemar test: paired binary outcomes for two configurations on the SAME items
# ----------------------------------------------------------------------------

def mcnemar_from_pairs(a_correct: Sequence[bool],
                       b_correct: Sequence[bool]) -> Dict[str, float]:
    """McNemar test comparing config A vs B over paired (same-item) outcomes.

    b = # items where A correct, B wrong ; c = # items where A wrong, B correct.
    Uses the exact binomial test (robust for small discordant counts).
    """
    assert len(a_correct) == len(b_correct), "paired arrays must align"
    b = sum(1 for a, bb in zip(a_correct, b_correct) if a and not bb)
    c = sum(1 for a, bb in zip(a_correct, b_correct) if (not a) and bb)
    n_disc = b + c
    if n_disc == 0:
        return {"b": b, "c": c, "p_value": 1.0, "significant": False}
    if _HAVE_SCIPY:
        p = _sps.binomtest(min(b, c), n_disc, 0.5).pvalue
    else:  # normal approx with continuity correction
        chi2 = (abs(b - c) - 1) ** 2 / n_disc
        p = math.erfc(math.sqrt(chi2 / 2))
    return {"b": b, "c": c, "p_value": float(p), "significant": bool(p < 0.05)}


# ----------------------------------------------------------------------------
# One-way ANOVA + Tukey HSD for continuous metrics (e.g. latency)
# ----------------------------------------------------------------------------

def anova_tukey(groups: Dict[str, Sequence[float]]) -> Dict[str, object]:
    """One-way ANOVA across named groups; Tukey HSD pairwise if ANOVA is significant.

    `groups` = {config_name: [values...]}. Returns F, p and pairwise verdicts.
    """
    names = [k for k, v in groups.items() if len(v) >= 2]
    data = [list(groups[k]) for k in names]
    if len(names) < 2:
        return {"F": float("nan"), "p_value": float("nan"), "pairwise": {}}
    if not _HAVE_SCIPY:
        return {"F": float("nan"), "p_value": float("nan"), "pairwise": {},
                "note": "scipy unavailable"}
    F, p = _sps.f_oneway(*data)
    pairwise: Dict[str, Dict[str, float]] = {}
    try:
        res = _sps.tukey_hsd(*data)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                key = f"{names[i]} vs {names[j]}"
                pij = float(res.pvalue[i, j])
                pairwise[key] = {"p_value": pij, "significant": bool(pij < 0.05)}
    except Exception as e:
        pairwise = {"_error": str(e)}
    return {"F": float(F), "p_value": float(p), "significant": bool(p < 0.05),
            "pairwise": pairwise}


# ----------------------------------------------------------------------------
# Noise floor: how much does the SAME config wobble run-to-run?
# ----------------------------------------------------------------------------

def noise_floor(per_run_rates: Sequence[float]) -> Dict[str, float]:
    """Std of a proportion measured across repeated identical runs.

    Any difference between two configs SMALLER than ~2*std should be reported
    as 'equivalent / within noise', not as a finding.
    """
    vals = list(per_run_rates)
    if len(vals) < 2:
        return {"std": float("nan"), "mean": (vals[0] if vals else float("nan")),
                "band": float("nan"), "n_runs": len(vals)}
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    std = math.sqrt(var)
    return {"std": std, "mean": mean, "band": 2 * std, "n_runs": len(vals)}


def holm_bonferroni(pvalues: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
    """Holm-Bonferroni step-down correction over a family of p-values.

    `pvalues` = {comparison_name: raw_p}. Returns the same keys with
    'p_raw', 'p_corrected' (monotone, capped at 1.0) and 'significant'
    (corrected). Use this whenever multiple comparisons are run against the
    same alpha (e.g. all pairwise tests within one RQ) — a single comparison
    declared "p<0.05" in isolation is not the same claim once it's one of
    several tests on the same data.
    """
    items = [(k, v) for k, v in pvalues.items() if v == v]  # drop NaN
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, Dict[str, float]] = {}
    running_max = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running_max = max(running_max, adj)
        out[k] = {"p_raw": p, "p_corrected": running_max,
                  "significant": bool(running_max < alpha)}
    for k, v in pvalues.items():
        if v != v:
            out[k] = {"p_raw": float("nan"), "p_corrected": float("nan"), "significant": False}
    return out


def cohens_kappa(rater_a: Sequence, rater_b: Sequence) -> Dict[str, float]:
    """Cohen's kappa for two raters over the same items (categorical labels)."""
    assert len(rater_a) == len(rater_b)
    n = len(rater_a)
    if n == 0:
        return {"kappa": float("nan"), "n": 0, "po": float("nan")}
    labels = sorted(set(rater_a) | set(rater_b))
    po = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n
    pe = 0.0
    for lab in labels:
        pa = sum(1 for a in rater_a if a == lab) / n
        pb = sum(1 for b in rater_b if b == lab) / n
        pe += pa * pb
    kappa = (po - pe) / (1 - pe) if (1 - pe) else 1.0
    return {"kappa": kappa, "n": n, "po": po, "pe": pe}
