# LLM-to-Action Planning for Robotic Battery Disassembly

An LLM-to-robot planning system for battery disassembly. It converts natural-language commands such as "remove the battery cover" into validated ROS 2 / MoveIt 2 action sequences for a Kinova Gen3 robotic arm.

The project focuses less on demo-only planning and more on reliability: validation, failure-mode analysis, statistical re-evaluation, perception-noise robustness, and reproducible evaluation reports.

## Demo

[![Watch the demo](https://img.youtube.com/vi/JhGjJqNxm7w/maxresdefault.jpg)](https://youtu.be/JhGjJqNxm7w)

*(click the image to watch the full demo on YouTube)*

- Natural-language command entered through the Gradio UI
- LLM generates a structured skill sequence
- Skill server validates and dispatches the actions
- MoveIt 2 plans and executes the trajectory

## Why this matters

LLM robotics demos often look successful when they only check JSON validity or replay fixed trajectories. This project tests whether the generated plans are actually correct, safe, robust under perception noise, and executable through a real motion-planning stack — and re-checks every headline claim with confidence intervals and paired significance tests before trusting it.

## What this is

- A **Gradio web UI** → **LLM planner** (GPT-4o-mini via OpenAI, or Llama-3.2 via OpenRouter/Ollama) → **skill dispatcher** → **ROS 2 / MoveIt 2** pipeline that plans and executes battery-disassembly tasks on a 9-DOF Kinova Gen3 + Robotiq-85 gripper.
- An **evaluation framework** (`experiments/`) spanning 5 research questions: component ablation, safety-validation effectiveness, RAG memory sensitivity, perception-noise robustness, and — the one that actually closes the loop — sending generated plans into the live ROS 2/MoveIt stack instead of only scoring them as text.

## Key results

| Finding | Detail |
|---|---|
| **RAG measurably improves planning accuracy** | Exact-match plan correctness rises from 44.1% (LLM only) to 52.9% (LLM+RAG), paired-significance p<0.001 — with a corrected ground truth (see below) this holds up, it isn't just noise. |
| **Text-level validation catches almost no unsafe requests — 2.5%** | Validator recall against the per-command safety labels is 2.5% ([0.7%, 8.7%]): 2 of 80 commands that should have been refused. A real-execution probe that sent unvalidated hallucinated plans straight to the live ROS 2 skill server caught 3 of 9 ([12.1%, 64.6%]), so dispatch-level structural checks catch proportionally more than the text-level validator does. Both agree the two-tier check reliably catches format/vocabulary errors but misses semantically-wrong-but-structurally-valid actions (e.g. "disconnect the BMS connector" executed as grasp-then-disconnect on the battery box instead). **An earlier revision reported 12.4% here.** That figure came from a metric that counted any plan not exactly matching a reference as an unsafe command, which credited the validator for rejecting safe-but-non-canonical plans; splitting the ground truth into separate safety and plan-quality views corrected it. See "RQ2 is reported as two views" in `experiments/README.md`. |
| **The automatic scorer is validated against a human, not assumed correct** | Cohen's κ = 0.690 (23/27 agreement) between the auto exact-match label and human ratings of the same outputs. All four disagreements are one pattern — the model prefixes a harmless `moveTo(approach_bolts)` before the reference action — so reported exact-match rates are a mild underestimate. Left uncorrected on purpose: widening the reference set to erase a known, explainable disagreement would be tuning the ruler to fit the answer. Ratings and the rubric are committed in `experiments/eval/human_ratings.json`. |
| **A ground-truth bug was found and fixed, not just the model** | 9 of 34 reference plans required a position/object with no real joint-angle or scene definition anywhere in the codebase — verified by cross-checking `waypoints.json` and `visual_state_manager.py` line by line. Fixing the labels (not the robot) changed the ablation study's headline numbers. |
| **Real ROS 2 execution now closes the loop** | RQ5 sends the same plans RQ1 scores as text into the live MoveIt stack: 17/18 achievable commands executed successfully regardless of whether the plan came from the validated or unvalidated configuration — the one structural failure (a "visually inspect for damage" command) fails for every plan source because there's no camera, not because of plan quality. |
| **Perception-noise robustness boundary** | Grasp success collapses from 97.1% to 45.1% as simulated pose-estimation error grows from 5mm to 10mm — an exploratory geometric simulation, not a claim about real camera hardware, but it quantifies how much perception accuracy this system would need. |
| **0 → 48 automated tests, plus a data-integrity bug fixed mid-project** | Added 48 unit tests for the scoring functions and statistics (Wilson CI, McNemar, Holm-Bonferroni, Cohen's κ). Also found and fixed a resume-logic bug where a failed LLM call that silently fell back to a non-LLM demo plan was being marked "done" instead of retried — a class of bug that permanently and invisibly corrupts a dataset if it isn't caught. |

`REVISION_MEMO.md` is a full self-audit log: every place an earlier draft of this work overclaimed, what was found on re-checking, and what was fixed. It's kept because the corrections are more informative than a clean narrative would be.

## Engineering highlights

- Fixed MoveIt collision-scene attachment (`scene_manager.attach_object`/`detach_object`) so grasped objects join the gripper's collision model instead of being treated as free-floating obstacles.
- Rewrote the Allowed-Collision-Matrix setup from "overwrite" to "read-merge-reapply", fixing a bug where even the robot's HOME pose was flagged as colliding.
- Corrected `moveit_controllers.yaml` (missing top-level key) so MoveIt's controller manager actually registers controllers instead of planning successfully and then failing to execute.
- Removed an unsafe nested `rclpy.spin_once()` call that was stealing the action-client callback, causing the skill server to report a timeout even when execution had already succeeded.
- Implemented 5 previously-undispatched skills (`inspect`, `unscrew`, `disconnect`, `waitForStabilization`, `rotateGripper`) using only existing motion primitives — no invented force/torque sensing — with correct rejection of objects that have no defined coordinates.

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

## Tech stack

ROS 2 Humble, MoveIt 2, Kinova Gen3 / Kortex, Robotiq-85 gripper, GPT-4o-mini (OpenAI) / Llama-3.2 (OpenRouter / Ollama), ChromaDB, sentence-transformers, Gradio, pytest, pandas/statsmodels (evaluation).

## Repository structure

```
├── src/
│   ├── battery_dismantle_task/   # ROS 2 package: skill server, motion executor,
│   │                              #   planning-scene manager, MoveIt config, URDF
│   └── llm_agent/                 # LLM planner, RAG engine (ChromaDB), executor,
│                                   #   validator, Gradio web UI
├── experiments/                   # Evaluation pipeline: RQ1 ablation, RQ2 safety
│   └── eval/                      #   validation, RQ3 memory sweep, RQ4 perception-noise
│                                   #   sim, RQ5 real ROS2/MoveIt execution + defense
│                                   #   probe; stats (Wilson CI/McNemar/Holm-Bonferroni)
├── REVISION_MEMO.md               # Self-audit log of corrected claims
├── PROJECT_STRUCTURE.md           # Detailed module-by-module breakdown
└── FINAL_START.sh                 # Full-system launch script
```

## Reproducibility / running it

### 1. Evaluation pipeline (no ROS2 required for RQ1-4)

```bash
cd experiments
python -m pytest eval/ -v                                          # 48 tests
python run_rq4_perception_noise.py --trials 300                    # local, free, geometric simulation
python -m eval.analyze
python -m eval.build_workbook                                      # -> eval/Result_robot.xlsx
```

RQ1–RQ3 call the LLM planner directly in Python (no ROS2 needed) but do require an LLM backend — OpenAI, or Ollama/OpenRouter for free/local:

```bash
cd experiments
python run_fast.py --rq all --leakfree --trials 5 --backend openai --concurrency 4
# or: --backend ollama (free/local, much slower) / --backend openrouter
```

RQ5 needs a live ROS 2 stack and sends the plans RQ1 generated straight to it:

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash
python run_rq5_real_execution.py                     # should_pass: validated vs unvalidated vs reference
python run_rq5_real_execution.py --sources defense   # defense-in-depth probe
```

The older per-RQ scripts (`run_rq1_ablation.py`, `run_rq2_safety.py`, `run_rq3_memory.py`) still work standalone — see `--help` on each — but `run_fast.py` is faster (shares one plan-generation pass across RQ1/RQ2) and is what generates the data RQ5 consumes.

### 2. Full ROS 2 system

Requirements: Ubuntu 22.04 or WSL2, ROS 2 Humble, MoveIt 2, Python 3.x, and an OpenAI/OpenRouter API key or an Ollama install.

```bash
cp src/llm_agent/.env.example src/llm_agent/.env   # add your API key
bash FINAL_START.sh
# then open http://localhost:7862
```

## Known limitations

- RQ5 executes each command once per plan source rather than repeated trials, so its success-rate confidence intervals are wide (e.g. [74%,99%] for 17/18) — real motion planning (RRTConnect) has some run-to-run variance this doesn't capture.
- `motion_executor.py` and `skill_server.py` import `rclpy`/`moveit_msgs` at module level, so they're exercised through the live ROS 2 run described above rather than unit-tested in isolation.
- RQ4's perception-noise analysis is an exploratory geometric simulation (no camera, no physics engine), not a camera/grasp-force test on real hardware.
- Every result row records the exact model id and an ISO timestamp — cite the timestamp when quoting numbers, since a cloud provider's model alias can change behavior without a version bump.

## License

Apache-2.0
