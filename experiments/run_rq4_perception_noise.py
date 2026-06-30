#!/usr/bin/env python3
"""
RQ4 — Grasp-success sensitivity to perception (pose-estimation) noise.

WHY THIS EXISTS
----------------
The supervisor's feedback was: the system has no perception pipeline — object
poses are hardcoded (see object_definitions.py / waypoints.json
"cartesian_hints" in the battery_dismantle_task ROS2 package), so the
dissertation cannot say anything about how the pipeline would behave with a
real camera, which always has pose-estimation error.

We do NOT have a camera or a physics engine in this loop, so this is NOT a
claim of "we built perception." It is an honest, clearly-labelled GEOMETRIC
SIMULATION: take the real ground-truth object positions the robot already
uses, inject Gaussian pose noise at increasing sigma (as a stand-in for a
real RGBD/pose-estimation pipeline's error), and ask "at what noise level
does a grasp attempt at the perceived (not true) position stop landing
within the gripper's tolerance band?" That is a legitimate, reproducible
robustness sweep — the same flavour as RQ1-RQ3's controlled ablations — and
it gives the dissertation a real "robustness under varying conditions"
result instead of zero perception discussion.

GROUND-TRUTH NUMBERS — where they come from (not invented)
------------------------------------------------------------
- TopCoverBolts / BatteryBox_0 cartesian_hints: copied verbatim from
  src/battery_dismantle_task/config/waypoints.json ("objects" -> "approach"
  -> "cartesian_hints" -> "approach_position"), the values the real
  skill_server actually targets.
- The 4 individual bolt corner positions: recomputed from the SAME formula
  as src/battery_dismantle_task/battery_dismantle_task/object_definitions.py
  (BATTERY_BASE_POSITION + BOLT_OFFSET_X/Y), so they match that file exactly.
- BOLT_DIMENSIONS.width = 0.02 m (2 cm) is from the same file.

GRASP TOLERANCE — a documented modelling ASSUMPTION, not a measurement
------------------------------------------------------------------------
There is no real grasp-force/compliance test in this project, so the
tolerance band has to be stated as an assumption: a parallel-jaw grasp on a
2 cm-wide bolt succeeds if the lateral offset between the perceived and true
position is within half the bolt width plus a small finger-compliance
margin. We use TOLERANCE_MM = bolt_radius_mm + COMPLIANCE_MARGIN_MM, with
COMPLIANCE_MARGIN_MM swept too (default 5mm) so the sensitivity to this
assumption itself is visible, not hidden in one hardcoded number.

Usage
-----
    python run_rq4_perception_noise.py --trials 300
    python -m eval.analyze        # picks up results_fast/rq4.jsonl automatically
    python -m eval.build_workbook
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

BASE = Path(__file__).parent
RESULTS = BASE / "results_fast"
RESULTS.mkdir(exist_ok=True)
OUT = RESULTS / "rq4.jsonl"

# ---------------------------------------------------------------- ground truth
# Verbatim from battery_dismantle_task/config/waypoints.json "cartesian_hints"
CARTESIAN_HINTS_M = {
    "TopCoverBolts": {"x": 0.5, "y": 0.0, "z": 0.3},
}

# Recomputed from object_definitions.py's own formula (BATTERY_BASE_POSITION +
# BOLT_OFFSET_X/Y), so these match that file's BOLT_POSITIONS exactly.
BATTERY_BASE = {"x": 0.45, "y": 0.0, "z": 0.0}
BATTERY_DIM = {"width": 0.36, "depth": 0.26, "height": 0.08}
TOP_COVER_DIM = {"width": 0.36, "depth": 0.26, "height": 0.01}
BOLT_DIM = {"width": 0.02, "depth": 0.02, "height": 0.015}
BOLT_OFFSET_X, BOLT_OFFSET_Y = 0.15, 0.10

_top_cover_z = BATTERY_BASE["z"] + BATTERY_DIM["height"] + TOP_COVER_DIM["height"] / 2
_bolt_z = _top_cover_z + TOP_COVER_DIM["height"] / 2 + BOLT_DIM["height"] / 2

BOLT_CORNERS_M = {
    f"Bolt_{name}": {
        "x": BATTERY_BASE["x"] + sx * BOLT_OFFSET_X,
        "y": BATTERY_BASE["y"] + sy * BOLT_OFFSET_Y,
        "z": _bolt_z,
    }
    for name, (sx, sy) in {
        "FrontLeft": (1, 1), "FrontRight": (1, -1),
        "BackLeft": (-1, 1), "BackRight": (-1, -1),
    }.items()
}

OBJECTS_M: Dict[str, Dict[str, float]] = {**CARTESIAN_HINTS_M, **BOLT_CORNERS_M}

BOLT_RADIUS_MM = (BOLT_DIM["width"] * 1000) / 2  # 10 mm
NOISE_LEVELS_MM = [0, 2, 5, 10, 15, 20, 30, 40, 60]
COMPLIANCE_MARGINS_MM = [0.0, 5.0, 10.0]  # sweep the assumption itself
TRIALS_PER_CELL_DEFAULT = 300
SEED = 42


# ---------------------------------------------------------------- sim
def simulate(trials: int, seed: int = SEED) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows = []
    for obj_name, pos in OBJECTS_M.items():
        for sigma_mm in NOISE_LEVELS_MM:
            sigma_m = sigma_mm / 1000.0
            # isotropic Gaussian perception error on each axis
            dx = rng.normal(0.0, sigma_m, trials) if sigma_m > 0 else np.zeros(trials)
            dy = rng.normal(0.0, sigma_m, trials) if sigma_m > 0 else np.zeros(trials)
            dz = rng.normal(0.0, sigma_m, trials) if sigma_m > 0 else np.zeros(trials)
            err_mm = np.sqrt(dx**2 + dy**2 + dz**2) * 1000.0
            for margin_mm in COMPLIANCE_MARGINS_MM:
                tol_mm = BOLT_RADIUS_MM + margin_mm
                for t in range(trials):
                    rows.append({
                        "_key": f"rq4|{obj_name}|{sigma_mm}|{margin_mm}|{t}",
                        "rq": 4,
                        "object": obj_name,
                        "sigma_mm": sigma_mm,
                        "compliance_margin_mm": margin_mm,
                        "tolerance_mm": tol_mm,
                        "trial": t,
                        "error_mm": round(float(err_mm[t]), 3),
                        "success": bool(err_mm[t] <= tol_mm),
                        "seed": seed,
                        "sim_only": True,  # NEVER let this be mistaken for real sensor data
                    })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=TRIALS_PER_CELL_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rows = simulate(args.trials, args.seed)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_objects = len(OBJECTS_M)
    n_cells = n_objects * len(NOISE_LEVELS_MM) * len(COMPLIANCE_MARGINS_MM)
    print(f"Wrote {len(rows)} rows to {OUT}")
    print(f"  {n_objects} objects x {len(NOISE_LEVELS_MM)} noise levels x "
          f"{len(COMPLIANCE_MARGINS_MM)} compliance margins x {args.trials} trials = {n_cells * args.trials}")
    print("  This is a GEOMETRIC SIMULATION (no camera, no physics engine) — "
          "see module docstring for ground-truth provenance and the tolerance assumption.")


if __name__ == "__main__":
    main()
