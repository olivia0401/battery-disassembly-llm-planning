# Project structure

Module-by-module map of the repository. See `README.md` for results and how to run things.

```
.
├── src/
│   ├── battery_dismantle_task/            # ROS 2 (ament_python) package
│   │   ├── battery_dismantle_task/
│   │   │   ├── skill_server.py            # Receives LLM commands, dispatches skills, publishes feedback
│   │   │   ├── skill_handlers.py          # Higher-level skills (grasp, release, dismantle, inspect, unscrew, ...)
│   │   │   ├── motion_executor.py         # MoveIt 2 planning + trajectory execution
│   │   │   ├── motion_config.py           # Motion timeouts and tuning constants
│   │   │   ├── scene_manager.py           # Planning-scene attach/detach and ACM merge
│   │   │   ├── object_definitions.py      # Scene object dimensions and positions
│   │   │   ├── visual_state_manager.py    # Creates scene objects and mirrors grasp/release in RViz
│   │   │   ├── session.py                 # Per-command session context
│   │   │   ├── publish_initial_joint_states.py
│   │   │   └── interactive_joint_control.py  # Optional: drive joints from RViz markers (not launched by default)
│   │   ├── config/                        # waypoints.json, joint limits, MoveIt config, RViz config
│   │   ├── launch/
│   │   │   ├── fake_execution_complete.launch.py      # Full stack used by FINAL_START.sh and RQ5
│   │   │   ├── run_dismantle_task.launch.py           # MoveIt + RViz only
│   │   │   └── start_controllers_interactive.launch.py
│   │   ├── urdf/, grippers/                # Kinova Gen3 + Robotiq 2F-85 description
│   │   ├── package.xml, setup.py, setup.cfg
│   │
│   └── llm_agent/                          # LLM side (plain Python)
│       ├── planner.py                      # Natural language -> skill sequence (optionally RAG-assisted)
│       ├── llm_client.py                   # OpenAI / OpenRouter / Ollama / Chutes backends
│       ├── rag_engine.py                   # ChromaDB + sentence-transformers retrieval
│       ├── validator.py                    # Schema + sequence-rule validation
│       ├── executor.py                     # Publishes commands to ROS 2 (or mock mode)
│       ├── topics.py                       # ROS 2 topic-name constants
│       ├── web_ui.py                       # Gradio UI (port 7862)
│       ├── config/                         # skills.json, safety.yaml, prompt.txt, prompt_clean.txt
│       ├── rag_data/                       # Experience cases for retrieval
│       ├── requirements.txt
│       └── .env.example
│
├── experiments/                            # Evaluation (RQ1-RQ5); see experiments/README.md
│   ├── run_fast.py                         # RQ1-RQ3 data collection (produced results_fast/)
│   ├── run_rq4_perception_noise.py         # RQ4 geometric simulation
│   ├── run_rq5_real_execution.py           # RQ5 live ROS 2 execution
│   ├── run_rq1_ablation.py, run_rq2_safety.py, run_rq3_memory.py  # legacy per-RQ scripts
│   ├── unified_test_suite.json             # 34 test commands
│   ├── results_fast/                       # rq1..rq5 JSONL result data
│   └── eval/                               # metrics, stats, analysis, workbook, tests
│
├── .github/workflows/ci.yml                # Runs the pure-Python unit tests
├── FINAL_START.sh                          # Full-system launch script
├── CHANGELOG.md
└── README.md
```

## Skills dispatched by `skill_server.py`

- Primitives: `moveTo`, `openGripper`, `closeGripper`, `approach`, `place`, `retreat`
- Higher-level: `grasp`, `release`, `dismantle`, `sequence`, `inspect`, `unscrew`, `disconnect`, `waitForStabilization`, `rotateGripper`
- Selection: `selectObject`, `selectPlace`

## Message formats

Command (`llm_cmd/v1`), published by `executor.py` (`command_id` is optional and echoed back in feedback):

```json
{"schema": "llm_cmd/v1", "skill": "moveTo", "target": "HOME"}
```

Feedback (`llm_fb/v1`), published by `skill_server.py`:

```json
{"schema": "llm_fb/v1", "status": "success", "message": "Skill 'moveTo' completed", "command_id": "cmd_001"}
```

## Adding a skill

1. Implement it in `skill_handlers.py`.
2. Add a branch to the dispatcher in `skill_server.py`.
3. Describe it in `src/llm_agent/config/skills.json` so the planner and validator know about it.

## Logs

`FINAL_START.sh` writes to `$LOG_DIR` (default `/tmp`): `ros2_launch.log` and `webui.log`.
RQ5 writes `/tmp/rq5_ros2_launch.log`.
