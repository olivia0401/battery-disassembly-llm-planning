# Rigorous evaluation pipeline (`experiments/eval/`)

Turns raw planner results into a conclusion-first 4-tab workbook with proper
metrics (Exact, step P/R/F1), failure-mode analysis, Wilson 95% CIs and
McNemar significance tests.

## Modules
| File | Purpose |
|---|---|
| `metrics.py` | LCS step Precision/Recall/F1, multi-reference Exact, 7-class failure classifier |
| `stats.py` | Wilson CI, McNemar, ANOVA+Tukey, noise floor, Cohen's kappa |
| `reference_plans.json` | ground-truth references + safety labels for all **34** commands (18 `should_pass` / 16 `should_block`); **all human-reviewed 2026-07-23**, 0 flagged `needs_human_review` |
| `gen_reference_plans.py` | regenerates the reference scaffold |
| `analyze.py` | recomputes metrics from results -> `analysis_summary.json` (+ provenance) |
| `build_workbook.py` | renders `Result_robot.xlsx` (Exec / Recommendations / Analysis / Label Validation) |
| `compute_kappa.py` | Human↔Auto Cohen's kappa from the Label-Validation tab |
| `human_ratings.json` | the 27 human Y/N ratings + the rubric they were applied with. **Committed on purpose**: they are hand-made and unreproducible, while `Result_robot.xlsx` is gitignored and rebuilt from scratch every run. `build_workbook` restores them into column H so a rebuild can't destroy them |
| `build_leakfree_assets.py` | builds `prompt_clean.txt` + `experience_cases_clean.json` + `memory_split.json` |

## Run (analysis only, no API needed)
```bash
cd experiments
python -m eval.analyze
python -m eval.build_workbook      # -> eval/Result_robot.xlsx
```

## Collect fresh data (needs a funded LLM key)
```bash
# fast, concurrent, resume-safe, leak-free, honest provenance
python run_fast.py --rq all --leakfree --trials 10 --concurrency 8
python -m eval.analyze && python -m eval.build_workbook
```
`--leakfree` uses the clean prompt (no leaked answers) and the disjoint memory
split. Every row records `planner_mode`, so demo-fallbacks (e.g. on API failure)
are flagged and never counted as LLM output.

## Data currently in the repo

`results_fast/rq1.jsonl` holds 850 rows = 34 commands x 5 configs x 5 trials.
680 of them are real LLM calls (`planner_mode="llm"`, `model="openai:gpt-4o-mini"`);
the other 170 are the SB scripted baseline, which is a non-LLM configuration by
design, not a failed call. `analyze.py` drops `fallback_demo`/`error` rows and
reports the per-config drop rate so an uneven dropout can't quietly bias a
comparison.

## Known blockers / TODO

- **Known systematic gap in the `Exact` metric.** Human-Auto kappa is **0.690**
  (agreement 23/27 = 85.2%, rated 2026-07-23) — substantial agreement, so the
  auto label is usable. But all four disagreements are the *same* pattern: the
  model prefixes a harmless `moveTo(approach_bolts)` before the reference
  action, the human calls it correct, `Exact` calls it a mismatch. So `Exact`
  is systematically strict about approach moves, and per-config Exact rates are
  a mild underestimate. Deliberately not "fixed" by widening the reference set:
  removing a known, explainable disagreement purely to push kappa toward 1.0
  would be tuning the ruler to fit the answer.
- **Text-level validation barely catches unsafe commands: safety recall is
  2/80 (2.5%) at the Schema and Full levels, 0/80 at No-Validation and Rule.**
  This is a real negative result about the validator, not a measurement
  artefact — see the two-view explanation in `experiments/README.md`. It is
  also why RQ5's defense-in-depth probe exists.

## Secrets

`src/llm_agent/.env` is gitignored (along with `*.key`) and has **never been
committed** — verified against the full commit history, not just the working
tree. Only `.env.example` is tracked. An earlier revision of this file claimed
a key had been committed and needed rotating; that claim was wrong and has been
removed.
