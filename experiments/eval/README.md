# Rigorous evaluation pipeline (`experiments/eval/`)

Turns raw planner results into a conclusion-first 4-tab workbook with proper
metrics (Exact, step P/R/F1), failure-mode analysis, Wilson 95% CIs and
McNemar significance tests.

## Modules
| File | Purpose |
|---|---|
| `metrics.py` | LCS step Precision/Recall/F1, multi-reference Exact, 7-class failure classifier |
| `stats.py` | Wilson CI, McNemar, ANOVA+Tukey, noise floor, Cohen's kappa |
| `reference_plans.json` | ground-truth references + safety labels (23/35 flagged `needs_human_review`) |
| `gen_reference_plans.py` | regenerates the reference scaffold |
| `analyze.py` | recomputes metrics from results -> `analysis_summary.json` (+ provenance) |
| `build_workbook.py` | renders `Result_robot.xlsx` (Exec / Recommendations / Analysis / Label Validation) |
| `compute_kappa.py` | Human↔Auto Cohen's kappa from the Label-Validation tab |
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

## Known blockers / TODO
- The committed OpenRouter key has **no credits** -> all live calls fall back to
  the keyword demo planner. Add credits (or run local Ollama) before collecting.
- Human-review the references flagged `needs_human_review`, fill column H of the
  Label-Validation tab, then `python -m eval.compute_kappa`.
- Rotate the key in `src/llm_agent/.env` (it is committed) and gitignore it.
