# LLM-to-Action Planning for Robotic Battery Disassembly

An LLM-to-robot planning system for battery disassembly. It converts natural-language commands such as "remove the battery cover" into validated ROS 2 / MoveIt 2 action sequences for a Kinova Gen3 robotic arm.

The project focuses less on demo-only planning and more on reliability: validation, failure-mode analysis, statistical re-evaluation, perception-noise robustness, and reproducible evaluation reports.

## Demo

<!-- TODO: add a screenshot or short GIF of the Gradio UI + RViz planning view here, e.g.
![System demo](docs/demo.gif)
-->

- Natural-language command entered through the Gradio UI
- LLM generates a structured skill sequence
- Skill server validates and dispatches the actions
- MoveIt 2 plans and executes the trajectory

## Why this matters

LLM robotics demos often look successful when they only check JSON validity or replay fixed trajectories. This project tests whether the generated plans are actually correct, safe, robust under perception noise, and executable through a real motion-planning stack — and re-checks every headline claim with confidence intervals and paired significance tests before trusting it.

## What this is

- A **Gradio web UI** → **LLM planner** (Llama-3.2, via OpenRouter/Ollama) → **skill dispatcher** → **ROS 2 / MoveIt 2** pipeline that plans and executes battery-disassembly tasks on a 9-DOF Kinova Gen3 + Robotiq-85 gripper.
- An **evaluation framework** (`experiments/`) that scores plan correctness, safety-validation effectiveness, and robustness across 2,278 simulation trials plus ~1,500 perception-noise robustness trials.

## Key results

| Finding | Detail |
|---|---|
| **Schema validation catches ~14% of real errors** | Most plan failures are semantic (wrong object, wrong step order), not malformed JSON — a pure format-checker misses most of what actually goes wrong. 95% CI: [7.3%, 25.3%]. |
| **Statistical re-evaluation prevented overclaiming** | After paired tests and Holm-Bonferroni correction, RAG and the validation layer showed directional but not statistically significant improvements. This changed the project conclusion from "RAG helps" to a more defensible reliability finding. |
| **MoveIt dynamic planning fixed and verified** | Diagnosed and fixed 5 ROS 2 / MoveIt 2 integration bugs, moving the system from pre-baked trajectory replay to verified RRTConnect planning and trajectory execution. Full diagnosis in `REVISION_MEMO.md` §8. |
| **Perception-noise robustness boundary** | Grasp success collapses from 97.1% to 45.1% as simulated pose-estimation error grows from 5mm to 10mm — quantifies how much perception accuracy this system actually needs. |
| **0 → 48 automated tests** | The codebase had zero test coverage before this work; added 48 unit tests for the scoring functions, statistics (Wilson CI, McNemar, Holm-Bonferroni, Cohen's κ), and the ROS2-independent skill-handler logic. |

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

ROS 2 Humble, MoveIt 2, Kinova Gen3 / Kortex, Robotiq-85 gripper, Llama-3.2 (OpenRouter / Ollama), ChromaDB, sentence-transformers, Gradio, pytest, pandas/statsmodels (evaluation).

## Repository structure

```
├── src/
│   ├── battery_dismantle_task/   # ROS 2 package: skill server, motion executor,
│   │                              #   planning-scene manager, MoveIt config, URDF
│   └── llm_agent/                 # LLM planner, RAG engine (ChromaDB), executor,
│                                   #   validator, Gradio web UI
├── experiments/                   # Evaluation pipeline: plan scoring, statistics
│   └── eval/                      #   (Wilson CI / McNemar / Holm-Bonferroni / noise
│                                   #   floor), RQ1-RQ4 runners, Excel report builder
├── REVISION_MEMO.md               # Self-audit log of corrected claims
├── PROJECT_STRUCTURE.md           # Detailed module-by-module breakdown
└── FINAL_START.sh                 # Full-system launch script
```

## Reproducibility / running it

### 1. Evaluation pipeline (no ROS2/LLM required)

```bash
cd experiments
python -m pytest eval/ -v                                          # 48 tests
python run_rq4_perception_noise.py --trials 300                    # local, free, geometric simulation
python -m eval.analyze
python -m eval.build_workbook                                      # -> eval/Result_robot.xlsx
```

RQ1–RQ3 call the LLM planner directly in Python (no ROS2 needed) but do require an LLM backend. The combined leak-free runner supports both Ollama (local) and OpenRouter (cloud):

```bash
cd experiments
python run_fast.py --rq all --leakfree --trials 5 --backend ollama --concurrency 1
# or: --backend openrouter (uses OPENROUTER_API_KEY from src/llm_agent/.env)
```

Individual RQ scripts (`run_rq1_safety.py`, `run_rq2_memory.py`, `run_rq3_ablation.py`) are also runnable directly — see `--help` on each for their `--commands`/`--trials`/`--resume` options.

### 2. Full ROS 2 system

Requirements: Ubuntu 22.04 or WSL2, ROS 2 Humble, MoveIt 2, Python 3.x, and an Ollama install or OpenRouter API key.

```bash
cp src/llm_agent/.env.example src/llm_agent/.env   # add your OPENROUTER_API_KEY
bash FINAL_START.sh
# then open http://localhost:7862
```

## Known limitations

- RQ1–RQ3 trials currently use a small local model with 1–2 trials per command; more trials would tighten the noise-floor estimates.
- `motion_executor.py` and `skill_server.py` import `rclpy`/`moveit_msgs` at module level, so they're exercised through the live ROS 2 run described above rather than unit-tested in isolation.
- RQ4's perception-noise analysis is a geometric simulation (no camera, no physics engine), not a camera/grasp-force test on real hardware.

## License

Apache-2.0
