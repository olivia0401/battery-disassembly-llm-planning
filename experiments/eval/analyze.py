"""
Analyze raw experiment results -> structured, statistically-annotated summary.

Works on BOTH the legacy result files (results/rq*_results_*.json) and the new
run_fast.py output. Recomputes the rigorous metrics (Exact, step P/R/F1,
failure modes) from the stored plans + reference_plans.json, then aggregates
with Wilson CIs, McNemar tests and failure-mode distributions.

Usage:
    python -m eval.analyze            # -> writes eval/analysis_summary.json
"""
from __future__ import annotations
import json, glob
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional

from eval.metrics import evaluate_trial
from eval.stats import wilson_ci, mcnemar_from_pairs, anova_tukey, noise_floor, holm_bonferroni

BASE = Path(__file__).resolve().parent.parent          # experiments/
SRC = BASE.parent / "src" / "llm_agent"

CONFIG_NAMES = {"SB": "Scripted Baseline", "LO": "LLM Only", "LV": "LLM+Validation",
                "LR": "LLM+RAG", "FS": "Full System"}
LEVEL_NAMES = {"NV": "No Validation", "SV": "Schema", "RV": "Rule", "FV": "Full"}


# ---------------------------------------------------------------- loading
def _latest(pattern: str) -> Optional[Path]:
    files = sorted(glob.glob(str(BASE / "results" / pattern)))
    return Path(files[-1]) if files else None


def load_json(path: Path) -> List[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path and path.exists() else []


def load_rq(n: int) -> List[dict]:
    """Prefer the new run_fast JSONL (results_fast/rqN.jsonl); else legacy json."""
    jsonl = BASE / "results_fast" / f"rq{n}.jsonl"
    if jsonl.exists():
        rows = []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        if rows:
            return rows
    return load_json(_latest(f"rq{n}_results_*.json"))


def load_refs():
    refs = json.loads((Path(__file__).parent / "reference_plans.json").read_text(encoding="utf-8"))
    return refs["commands"]


def load_vocab():
    sk = json.loads((SRC / "config" / "skills.json").read_text(encoding="utf-8"))
    return ({s["name"] for s in sk["available_skills"]},
            set(sk["available_poses"]), set(sk["available_objects"]))


# ---------------------------------------------------------------- enrichment
def enrich(rows: List[dict], refs, vocab) -> List[dict]:
    """Attach recomputed Exact / step P-R-F1 / failure_mode to every row."""
    vs, vp, vo = vocab
    out = []
    for r in rows:
        cmd = r.get("command", "")
        ref = refs.get(cmd, {})
        acc = ref.get("acceptable_reference_plans", [])
        ood = bool(ref.get("out_of_domain", False))
        plan = r.get("planned_skills", []) or []
        ev = evaluate_trial(plan, acc, vs, vp, vo, command_is_out_of_domain=ood)
        rr = dict(r)
        rr.update(ev)
        rr["safety_label"] = ref.get("safety_label", "unknown")
        rr["ref_out_of_domain"] = ood
        # plan-level ground truth for the safety confusion matrix:
        rr["should_block_truth"] = bool(ood or (not ev["is_correct"]))
        out.append(rr)
    return out


def drop_fallbacks(rows):
    """Remove demo-fallback / error rows (not real LLM output).

    Scripted-baseline rows have planner_mode None and are kept — they are a
    legitimate non-LLM configuration, not a failed LLM call.
    """
    return [r for r in rows if r.get("planner_mode") not in ("fallback_demo", "error")]


def _rate_block(rows, key):
    k = sum(1 for r in rows if r[key])
    return wilson_ci(k, len(rows))


def _trial_noise_floor(rs, key="exact"):
    """Run-to-run wobble of `key`'s rate across repeated trials of the SAME config.

    Groups rows by trial_id, computes the rate within each trial group, then
    feeds those rates into noise_floor(). Needs >=2 trials to say anything;
    with trials=1 this returns None and callers must not silently treat
    "no noise data" as "no noise" — a single-trial run gives zero information
    about how much a result could swing on a rerun.
    """
    by_trial = defaultdict(list)
    for r in rs:
        by_trial[r.get("trial_id")].append(bool(r.get(key)))
    if len(by_trial) < 2:
        return None
    rates = [sum(v) / len(v) for v in by_trial.values() if v]
    return noise_floor(rates) if len(rates) >= 2 else None


def leave_one_command_out(by_cfg: Dict[str, List[dict]]) -> Optional[Dict[str, Any]]:
    """Drop each command in turn, recompute which config wins by mean Exact,
    and report how often the held-out winner matches the full-data winner.

    Low agreement means the RQ1 ranking is being driven by one or two
    commands rather than a robust pattern across the whole set — the same
    diagnostic the sibling prompt-eval project runs per-brief.
    """
    all_commands = sorted({r["command"] for rs in by_cfg.values() for r in rs})
    if not all_commands:
        return None

    def _winner(rows_by_cfg):
        means = {cfg: (sum(r["exact"] for r in rs) / len(rs) if rs else 0.0)
                 for cfg, rs in rows_by_cfg.items()}
        return max(means, key=lambda c: means[c]) if means else None

    full_winner = _winner(by_cfg)
    counts = Counter()
    for held_out in all_commands:
        held = {cfg: [r for r in rs if r["command"] != held_out] for cfg, rs in by_cfg.items()}
        w = _winner(held)
        if w:
            counts[w] += 1
    n = len(all_commands)
    agree = counts.get(full_winner, 0) if full_winner else 0
    return {"full_winner": full_winner, "winner_counts_when_held_out": dict(counts),
            "n_commands": n,
            "agreement_with_full_data_winner": round(agree / n, 3) if n else None}


# ---------------------------------------------------------------- RQ1 ablation
def analyze_rq1(refs, vocab):
    raw_rows = enrich(load_rq(1), refs, vocab)
    rows = drop_fallbacks(raw_rows)
    if not rows:
        return None
    by_cfg = defaultdict(list)
    by_cfg_raw = defaultdict(list)
    for r in rows:
        by_cfg[r.get("configuration", "?")].append(r)
    for r in raw_rows:
        by_cfg_raw[r.get("configuration", "?")].append(r)

    per_cfg = {}
    for cfg, rs in by_cfg.items():
        exact = _rate_block(rs, "exact")
        valid = _rate_block(rs, "plan_valid")
        f1 = [r["step_f1"] for r in rs]
        fails = Counter(r["failure_mode"] for r in rs if r["failure_mode"])
        per_cfg[cfg] = {
            "name": CONFIG_NAMES.get(cfg, cfg), "n": len(rs),
            "exact": exact, "plan_valid": valid,
            "mean_f1": round(sum(f1) / len(f1), 4) if f1 else 0.0,
            "failure_modes": dict(fails),
            # new runner stores planning_time (no total_time); fall back to it
            "mean_plan_ms": round(1000 * sum(r.get("planning_time", 0) for r in rs) / len(rs), 1),
            # None when trials<2: a single-trial run has no rerun data to measure
            # wobble from, and that absence must stay visible, not silently 0.
            "noise_floor_exact": _trial_noise_floor(rs, "exact"),
            # Dropped (fallback_demo/error) rows are not random: a config that
            # fails its LLM call more often than another is comparing on a
            # different, easier-survivor subset. Surface the rate so a reader
            # can judge whether that's biasing the Exact-rate comparison.
            "n_total_attempted": len(by_cfg_raw.get(cfg, [])),
            "n_dropped_fallback": len(by_cfg_raw.get(cfg, [])) - len(rs),
            "dropped_rate": round((len(by_cfg_raw.get(cfg, [])) - len(rs)) /
                                   len(by_cfg_raw.get(cfg, [])), 3) if by_cfg_raw.get(cfg) else 0.0,
        }

    # paired McNemar on Exact (align by command+trial).
    # NOTE: LO/LV share the same plan and LR/FS share the same plan (validation
    # never regenerates), so LO-vs-LV / LR-vs-FS would be trivially p=1.0 and are
    # NOT included. Validation's effect shows up in plan_valid, not Exact.
    def paired(cfg_a, cfg_b):
        a = {(r["command"], r["trial_id"]): r["exact"] for r in by_cfg.get(cfg_a, [])}
        b = {(r["command"], r["trial_id"]): r["exact"] for r in by_cfg.get(cfg_b, [])}
        keys = sorted(set(a) & set(b))
        if not keys:
            return None
        return mcnemar_from_pairs([a[k] for k in keys], [b[k] for k in keys])

    comparisons = {f"{x} vs {y}": paired(x, y)
                   for x, y in [("SB", "LO"), ("LO", "LR"), ("SB", "FS")]}
    # Holm-Bonferroni across this RQ's comparison family: 3 tests on the same
    # data means "p<0.05" in isolation overstates significance.
    raw_p = {k: v["p_value"] for k, v in comparisons.items() if v}
    corrected = holm_bonferroni(raw_p) if raw_p else {}
    for k, v in comparisons.items():
        if v:
            v["holm"] = corrected.get(k)

    # Flag comparisons where the two configs dropped a meaningfully different
    # share of rows to fallback/error — McNemar only sees the surviving
    # intersection, so an uneven drop rate means the two sides aren't being
    # compared on quite the same item set.
    for key, v in comparisons.items():
        if not v:
            continue
        x, y = key.split(" vs ")
        dx = per_cfg.get(x, {}).get("dropped_rate", 0.0)
        dy = per_cfg.get(y, {}).get("dropped_rate", 0.0)
        v["dropped_rate_gap"] = round(abs(dx - dy), 3)
        v["uneven_dropout_warning"] = abs(dx - dy) > 0.10

    lat = {cfg: [r.get("planning_time", 0) for r in rs] for cfg, rs in by_cfg.items()}
    return {"per_config": per_cfg, "comparisons": comparisons,
            "latency_anova": anova_tukey(lat),
            "leave_one_command_out": leave_one_command_out(by_cfg)}


# ---------------------------------------------------------------- RQ2 validation/safety
def analyze_rq2(refs, vocab):
    rows = drop_fallbacks(enrich(load_rq(2), refs, vocab))
    if not rows:
        return None
    by_lvl = defaultdict(list)
    for r in rows:
        by_lvl[r.get("validation_level", "?")].append(r)

    per_lvl = {}
    for lvl, rs in by_lvl.items():
        # blocked == validator rejected (plan_valid False)
        TP = sum(1 for r in rs if r["should_block_truth"] and not r["plan_valid"])
        FP = sum(1 for r in rs if (not r["should_block_truth"]) and not r["plan_valid"])
        FN = sum(1 for r in rs if r["should_block_truth"] and r["plan_valid"])
        TN = sum(1 for r in rs if (not r["should_block_truth"]) and r["plan_valid"])
        prec = wilson_ci(TP, TP + FP) if (TP + FP) else None
        rec = wilson_ci(TP, TP + FN) if (TP + FN) else None
        fpr = wilson_ci(FP, FP + TN) if (FP + TN) else None
        passed = sum(1 for r in rs if r["plan_valid"])
        per_lvl[lvl] = {
            "name": LEVEL_NAMES.get(lvl, lvl), "n": len(rs),
            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "precision": prec, "recall": rec, "fpr": fpr,
            "pass_rate": wilson_ci(passed, len(rs)),
            "mean_val_ms": round(1000 * sum(r.get("validation_time", 0) for r in rs) / len(rs), 4),
            # Same noise-floor treatment as RQ3/RQ1: None until trials>=2.
            "noise_floor_pass_rate": _trial_noise_floor(rs, "plan_valid"),
        }
    return {"per_level": per_lvl}


# ---------------------------------------------------------------- RQ2+RQ1 merged factorial
def analyze_rq1_rq2_factorial(refs, vocab):
    """RAG(on/off) x validation-level(NV/SV/RV/FV) safety pass-rate, in one table.

    RQ2 and RQ1 were originally two separate reports that both slice "does
    validation change the outcome" -- RQ2 by validation strictness (always on
    the RAG-on plan), RQ1 by RAG on/off (crossed with only NV vs "validator.
    validate_plan()", which is the SAME call as RQ2's FV level -- confirmed
    against run_rq2_safety.py's FullValidationWrapper). This merges both into
    one 2x4 grid instead of making a reader cross-reference two reports for
    what is really one factorial question.

    IMPORTANT HONESTY NOTE: this is NOT a full 8-cell factorial. The RAG-off
    arm was only ever run through NV (=RQ1's LO) and FV (=RQ1's LV) -- SV and
    RV were never applied to a no-RAG plan in any experiment. Those two cells
    are reported as None/"not tested", not estimated or interpolated.
    """
    rag_on = {lvl: c for lvl, c in
              (analyze_rq2(refs, vocab) or {"per_level": {}})["per_level"].items()}

    rq1_rows = drop_fallbacks(enrich(load_rq(1), refs, vocab))
    by_cfg = defaultdict(list)
    for r in rq1_rows:
        by_cfg[r.get("configuration", "?")].append(r)

    def rag_off_cell(rs):
        if not rs:
            return None
        passed = sum(1 for r in rs if r["plan_valid"])
        return {"n": len(rs), "pass_rate": wilson_ci(passed, len(rs))}

    rag_off = {
        "NV": rag_off_cell(by_cfg.get("LO", [])),
        "SV": None,  # never tested without RAG
        "RV": None,  # never tested without RAG
        "FV": rag_off_cell(by_cfg.get("LV", [])),
    }

    grid = {}
    for lvl in ("NV", "SV", "RV", "FV"):
        on = rag_on.get(lvl)
        grid[lvl] = {
            "name": LEVEL_NAMES.get(lvl, lvl),
            "rag_on": {"n": on["n"], "pass_rate": on["pass_rate"]} if on else None,
            "rag_off": rag_off[lvl],
        }
    untested = [lvl for lvl, v in grid.items() if v["rag_off"] is None]
    return {"grid": grid, "untested_rag_off_levels": untested}


# ---------------------------------------------------------------- RQ3 memory
def analyze_rq3(refs, vocab):
    rows = drop_fallbacks(enrich(load_rq(3), refs, vocab))
    if not rows:
        return None
    by_k = defaultdict(list)
    for r in rows:
        by_k[int(r.get("memory_size", -1))].append(r)

    per_k = {}
    for k, rs in sorted(by_k.items()):
        exact = _rate_block(rs, "exact")
        f1 = [r["step_f1"] for r in rs]
        seen = [r for r in rs if r.get("is_seen")]
        unseen = [r for r in rs if not r.get("is_seen")]
        per_k[k] = {
            "n": len(rs), "exact": exact,
            "mean_f1": round(sum(f1) / len(f1), 4) if f1 else 0.0,
            "seen_exact": _rate_block(seen, "exact") if seen else None,
            "unseen_exact": _rate_block(unseen, "exact") if unseen else None,
            "mean_plan_ms": round(1000 * sum(r.get("planning_time", 0) for r in rs) / len(rs), 1),
            "mean_sim": round(sum(r.get("max_similarity", 0) for r in rs) / len(rs), 3),
        }

    def paired(ka, kb):
        a = {(r["command"], r["trial_id"]): r["exact"] for r in by_k.get(ka, [])}
        b = {(r["command"], r["trial_id"]): r["exact"] for r in by_k.get(kb, [])}
        keys = sorted(set(a) & set(b))
        return mcnemar_from_pairs([a[k] for k in keys], [b[k] for k in keys]) if keys else None

    sizes = sorted(by_k)
    saturation = {f"k={sizes[i]} vs k={sizes[i+1]}": paired(sizes[i], sizes[i + 1])
                  for i in range(len(sizes) - 1)}
    raw_p = {k: v["p_value"] for k, v in saturation.items() if v}
    corrected = holm_bonferroni(raw_p) if raw_p else {}
    for k, v in saturation.items():
        if v:
            v["holm"] = corrected.get(k)
    for k, rs in by_k.items():
        per_k[k]["noise_floor_exact"] = _trial_noise_floor(rs, "exact")
    return {"per_k": per_k, "saturation": saturation}


# ---------------------------------------------------------------- RQ4 perception-noise simulation
def analyze_rq4():
    """Grasp-success vs simulated perception noise. See run_rq4_perception_noise.py
    docstring for ground-truth provenance and the tolerance assumption — this is
    a geometric simulation, not real sensor data, and must stay labelled as such.
    """
    jsonl = BASE / "results_fast" / "rq4.jsonl"
    if not jsonl.exists():
        return None
    rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return None

    # Pooled across objects, at the default (5mm) compliance margin, per noise level.
    default_margin = 5.0
    by_sigma = defaultdict(list)
    for r in rows:
        if r["compliance_margin_mm"] == default_margin:
            by_sigma[r["sigma_mm"]].append(r["success"])
    per_sigma = {}
    for sigma, succ in sorted(by_sigma.items()):
        per_sigma[sigma] = wilson_ci(sum(succ), len(succ))

    # Find the crossover: smallest sigma where the CI upper bound drops below 50%
    # ("can no longer assume the grasp would land") — a single readable headline number.
    crossover_sigma = None
    for sigma in sorted(per_sigma):
        if per_sigma[sigma]["hi"] < 0.5:
            crossover_sigma = sigma
            break

    # Per-object breakdown at default margin (objects differ only by position,
    # not by tolerance, so this mainly checks the sim has no per-object artefact).
    by_obj_sigma = defaultdict(list)
    for r in rows:
        if r["compliance_margin_mm"] == default_margin:
            by_obj_sigma[(r["object"], r["sigma_mm"])].append(r["success"])
    per_object = defaultdict(dict)
    for (obj, sigma), succ in by_obj_sigma.items():
        per_object[obj][sigma] = wilson_ci(sum(succ), len(succ))

    # Sensitivity to the tolerance-margin assumption itself, at a mid noise level.
    probe_sigma = 15
    by_margin = defaultdict(list)
    for r in rows:
        if r["sigma_mm"] == probe_sigma:
            by_margin[r["compliance_margin_mm"]].append(r["success"])
    margin_sensitivity = {m: wilson_ci(sum(s), len(s)) for m, s in sorted(by_margin.items())}

    return {
        "default_compliance_margin_mm": default_margin,
        "tolerance_basis": "BOLT_RADIUS_MM (10mm, from BOLT_DIMENSIONS.width=0.02m) + compliance_margin_mm",
        "per_sigma_pooled": per_sigma,
        "crossover_sigma_mm": crossover_sigma,
        "per_object": dict(per_object),
        "margin_sensitivity_at_15mm_noise": margin_sensitivity,
        "caveat": "Geometric simulation (Gaussian pose noise vs hardcoded ground truth), "
                  "no camera, no physics engine, no real grasp-force test. Tolerance is a "
                  "documented modelling assumption, not a measurement.",
    }


def provenance():
    """Count planner modes so demo-fallbacks can never be mistaken for LLM output."""
    prov = {}
    for n in (1, 2, 3):
        rows = load_rq(n)
        modes = Counter(r.get("planner_mode", "legacy/unknown") for r in rows)
        prov[f"rq{n}"] = dict(modes)
    return prov


# ---------------------------------------------------------------- RQ5 real execution
def analyze_rq5():
    """Real-execution success rate by plan source, from the live ROS2/MoveIt run
    in results_fast/rq5.jsonl. Pure aggregation of the recorded success flags —
    does not recompute anything, so it can't change the underlying result."""
    rows = load_rq(5)
    if not rows:
        return {}
    by = defaultdict(lambda: {"ok": 0, "n": 0, "failures": []})
    for r in rows:
        s = r.get("plan_source", "?")
        by[s]["n"] += 1
        if r.get("success"):
            by[s]["ok"] += 1
        else:
            by[s]["failures"].append(r.get("command", ""))
    per_source = {
        s: {"n": c["n"], "n_success": c["ok"],
            "success": wilson_ci(c["ok"], c["n"]),
            "failures": c["failures"]}
        for s, c in sorted(by.items())
    }
    return {"per_source": per_source}


def main():
    refs, vocab = load_refs(), load_vocab()
    prov = provenance()
    summary = {
        "data_provenance": prov,
        "rq2": analyze_rq2(refs, vocab),
        "rq3": analyze_rq3(refs, vocab),
        "rq1": analyze_rq1(refs, vocab),
        "rq1_rq2_factorial": analyze_rq1_rq2_factorial(refs, vocab),
        "rq4": analyze_rq4(),
        "rq5": analyze_rq5(),
    }
    fb = sum(v.get("fallback_demo", 0) + v.get("error", 0) for v in prov.values())
    if fb:
        print(f"⚠️  WARNING: {fb} rows are demo-fallback/error (not real LLM output). "
              f"See data_provenance in the summary.")
    out = Path(__file__).parent / "analysis_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out}")
    # quick console digest
    if summary["rq1"]:
        print("\nRQ1 ablation (recomputed Exact):")
        for cfg in ["SB", "LO", "LV", "LR", "FS"]:
            c = summary["rq1"]["per_config"].get(cfg)
            if c:
                e = c["exact"]
                print(f"  {cfg:3} {c['name']:18} Exact {100*e['p']:5.1f}% "
                      f"[{100*e['lo']:.1f},{100*e['hi']:.1f}]  valid {100*c['plan_valid']['p']:.1f}%  f1 {c['mean_f1']}")
        print("  comparisons (raw p -> Holm-corrected p):")
        for k, v in summary["rq1"]["comparisons"].items():
            if not v:
                continue
            h = v.get("holm") or {}
            sig = "significant" if h.get("significant") else "NOT significant after correction"
            warn = "  ⚠️ uneven dropout — comparing different survivor subsets" if v.get("uneven_dropout_warning") else ""
            print(f"    {k}: p={v['p_value']:.4f} -> p_holm={h.get('p_corrected', float('nan')):.4f} ({sig}){warn}")
        loo = summary["rq1"].get("leave_one_command_out")
        if loo:
            print(f"  leave-one-command-out: winner stays '{loo['full_winner']}' in "
                  f"{loo['agreement_with_full_data_winner']*100:.0f}% of {loo['n_commands']} holdouts "
                  f"({loo['winner_counts_when_held_out']})")
        no_noise_data = [cfg for cfg, c in summary["rq1"]["per_config"].items()
                          if c.get("noise_floor_exact") is None]
        if no_noise_data:
            print(f"  ⚠️  No repeated-trial noise-floor data for: {no_noise_data} "
                  f"(needs trials>=2 per command; rerun with --trials 3+ before treating "
                  f"p-values above as final).")
    if summary.get("rq2"):
        print("\nRQ2 validator recall (share of unsafe requests the validator blocks):")
        for lvl, d in summary["rq2"]["per_level"].items():
            rec = d.get("recall")
            if rec:
                print(f"  {lvl} {LEVEL_NAMES.get(lvl, lvl):12} recall {100*rec['p']:5.1f}% "
                      f"[{100*rec['lo']:.1f},{100*rec['hi']:.1f}]  "
                      f"(caught {rec['k']}/{rec['n']} unsafe)")
    if summary.get("rq3"):
        print("\nRQ3 RAG memory-size sweep (exact-match plan correctness):")
        for k, d in sorted(summary["rq3"]["per_k"].items(), key=lambda x: int(x[0])):
            e = d.get("exact")
            if e:
                print(f"  memory k={k:>3}: exact {100*e['p']:5.1f}% "
                      f"[{100*e['lo']:.1f},{100*e['hi']:.1f}]  n={e['n']}")
    if summary.get("rq1_rq2_factorial"):
        fac = summary["rq1_rq2_factorial"]
        print("\nRQ2+RQ1 merged factorial (validator pass-rate, RAG on vs off):")
        print(f"  {'level':6} {'RAG=on':>18}   {'RAG=off':>18}")
        for lvl in ("NV", "SV", "RV", "FV"):
            cell = fac["grid"][lvl]
            def fmt(c):
                if not c:
                    return "not tested"
                p = c["pass_rate"]
                return f"{100*p['p']:5.1f}% (n={c['n']})"
            print(f"  {lvl:6} {fmt(cell['rag_on']):>18}   {fmt(cell['rag_off']):>18}")
        if fac["untested_rag_off_levels"]:
            print(f"  ⚠️  RAG=off was never run through {fac['untested_rag_off_levels']} "
                  f"validation levels in any experiment -- those cells are honestly "
                  f"'not tested', not estimated.")
    if summary.get("rq4"):
        r4 = summary["rq4"]
        print(f"\nRQ4 perception-noise simulation (EXPLORATORY -- geometric simulation, "
              f"not real sensor data; answers the 'perception' third of the supervisor's "
              f"feedback, complementing RQ5's real-execution check on the 'motion "
              f"planning/autonomous execution' third):")
        for sigma, ci in r4["per_sigma_pooled"].items():
            print(f"  sigma={sigma:3}mm  grasp-success {100*ci['p']:5.1f}% [{100*ci['lo']:.1f},{100*ci['hi']:.1f}]")
        print(f"  crossover (success CI upper bound < 50%): sigma={r4['crossover_sigma_mm']}mm")
    if summary.get("rq5"):
        print("\nRQ5 real ROS2/MoveIt execution (success by plan source):")
        for s, d in summary["rq5"]["per_source"].items():
            su = d["success"]
            print(f"  {s:10} {d['n_success']}/{d['n']} succeeded  "
                  f"[{100*su['lo']:.1f},{100*su['hi']:.1f}]")
        # defense probe: a failed execution here == an unsafe plan the live stack caught
        dfn = summary["rq5"]["per_source"].get("defense")
        if dfn:
            caught = dfn["n"] - dfn["n_success"]
            print(f"  (defense-in-depth probe: {caught}/{dfn['n']} unsafe plans caught "
                  f"by the live stack)")


if __name__ == "__main__":
    main()
