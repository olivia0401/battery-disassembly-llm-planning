# LLM-to-Action Planning for Robotic Battery Disassembly

An LLM-driven planning system that converts natural-language commands ("remove the battery cover") into validated ROS 2 / MoveIt 2 action sequences for a Kinova Gen3 arm, evaluated through 2,278 simulation trials plus ~1,500 perception-noise robustness trials, with a rigorous statistical re-evaluation of which components actually help.

## What this is

- A **Gradio web UI** → **LLM planner** (Llama-3.2, via OpenRouter/Ollama) → **skill dispatcher** → **ROS 2 / MoveIt 2** pipeline that plans and executes battery-disassembly tasks on a 9-DOF Kinova Gen3 + Robotiq-85 gripper.
- An **evaluation framework** (`experiments/`) that scores plan correctness, safety-validation effectiveness, and robustness — and re-checks every headline claim with confidence intervals and paired significance tests before trusting it.

## Key results

| Finding | Detail |
|---|---|
| **Schema validation catches ~14% of real errors** | Most plan failures are semantic (wrong object, wrong step order), not malformed JSON — a pure format-checker misses most of what actually goes wrong. 95% CI: [7.3%, 25.3%]. |
| **RAG/safety-layer benefit not statistically significant** | After Holm-Bonferroni correction across all RQ2/RQ3 comparisons, the original assumption that retrieval-augmented context and the validation layer improve planning held up only directionally, not significantly. |
| **MoveIt dynamic planning: from fully bypassed to verified closed-loop** | `motion_executor.py` originally skipped MoveIt's real planner entirely (`use_direct_execution=True`, replayed pre-baked trajectories, no collision checking). 5 independent integration bugs were diagnosed and fixed (missing skill dispatch handlers, un-attached collision geometry, an Allowed-Collision-Matrix overwrite that broke even the HOME pose, a malformed controller-manager config, and a stolen `rclpy` callback) — verified end-to-end on a cold start with real RRTConnect planning and trajectory execution. See `REVISION_MEMO.md` §8 for the full bug-by-bug diagnosis log. |
| **Perception-noise robustness boundary** | Grasp success collapses from 97.1% to 45.1% as simulated pose-estimation error grows from 5mm to 10mm — quantifies how much perception accuracy this system actually needs. |
| **0 → 48 automated tests** | The codebase had zero test coverage before this work; added 48 unit tests for the scoring functions, statistics (Wilson CI, McNemar, Holm-Bonferroni, Cohen's κ), and the ROS2-independent skill-handler logic. |

`REVISION_MEMO.md` is a full self-audit log: every place an earlier draft of this work overclaimed, what was found on re-checking, and what was fixed. It's kept because the corrections are more informative than a clean narrative would be.

## Architecture

```
User input (Web UI)
    ↓
LLM Planner — natural language → skill sequence (Llama-3.2, RAG-assisted via ChromaDB)
    ↓
Executor — sends ROS 2 commands
    ↓
Skill Server — skill dispatch + safety validation
    ↓
Motion Executor — MoveIt 2 motion planning
    ↓
ROS 2 Control → robot execution
```

- `src/battery_dismantle_task/` — ROS 2 package: skill server, motion executor, planning-scene manager, MoveIt config, URDF.
- `src/llm_agent/` — LLM planner, RAG engine (ChromaDB + sentence-transformers), command executor, validator, Gradio web UI.
- `experiments/` — evaluation pipeline: plan scoring, statistics (Wilson CI / McNemar / Holm-Bonferroni / noise floor), RQ1–RQ4 experiment runners, Excel report builder.

## Tech stack

ROS 2 Humble, MoveIt 2, Kinova Gen3 / Kortex, Robotiq-85 gripper, Llama-3.2 (OpenRouter / Ollama), ChromaDB, sentence-transformers, Gradio, pytest, pandas/statsmodels (evaluation).

## Running it

```bash
# Evaluation pipeline — no ROS2/LLM required
cd experiments
python -m pytest eval/ -v                                          # 48 tests
python run_rq4_perception_noise.py --trials 300                    # local, free, geometric simulation
python -m eval.analyze
python -m eval.build_workbook                                      # -> eval/Result_robot.xlsx

# Full system — requires ROS 2 Humble (Ubuntu/WSL2) + Ollama or OpenRouter key
cp src/llm_agent/.env.example src/llm_agent/.env   # add your OPENROUTER_API_KEY
bash FINAL_START.sh
# then open http://localhost:7862
```

RQ1–RQ3 call the LLM planner directly in Python and need no ROS2. RQ4 is a pure-Python geometric simulation (no camera, no physics engine — explicitly labelled "SIMULATED" throughout). The ROS 2/MoveIt 2 package itself needs a real ROS 2 Humble environment and won't run on Windows directly.

## Known limitations

- RQ1–RQ3 trials currently use a small local model with 1–2 trials per command; more trials would tighten the noise-floor estimates.
- `motion_executor.py` and `skill_server.py` import `rclpy`/`moveit_msgs` at module level, so they're exercised through the live ROS 2 run described above rather than unit-tested in isolation.
- RQ4's perception-noise analysis is a geometric simulation, not a camera/grasp-force test on real hardware.

## License

Apache-2.0
