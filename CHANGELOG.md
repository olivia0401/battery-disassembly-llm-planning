# Changelog

Corrections and fixes that changed what this project claims. Earlier drafts overclaimed in several places; this lists what was re-checked and what changed.

## Evaluation corrections

- **Validator safety recall: 12.4% -> 2.5%.** The old figure used a combined ground truth (`out_of_domain or not exact_match`), which counted rejections of safe but non-canonical plans as safety wins. RQ2 is now reported as two separate views: safety (per-command `safety_label`) and plan quality. See `experiments/README.md`.
- **Reference plans corrected.** 9 of 34 reference plans needed a position or object with no joint-angle or scene definition anywhere in the code. They were relabelled, which changed the ablation's headline numbers. All 34 references were then human-reviewed (2026-07-23).
- **Scorer validated against a human.** Human-Auto Cohen's kappa = 0.690 (23/27). All four disagreements share one pattern (a harmless extra `moveTo(approach_bolts)`), so Exact is a mild underestimate. This is left uncorrected on purpose.
- **Leak-free re-run.** A clean prompt (no gold answers) plus RAG memory disjoint from the test set (`eval/memory_split.json`), with gpt-4o-mini, 5 trials per configuration.
- **Resume-logic bug fixed.** A failed LLM call that silently fell back to the keyword demo planner was being marked "done" and never retried. Fallback rows are now flagged via `planner_mode` and excluded.
- **Analysis bugs fixed.** Correct refusals were scored as `no_plan` failures. Comparisons between configurations that share identical plans (trivially p = 1) were removed. The latency column read a field the runner never wrote.
- **Statistics.** Wilson CIs, McNemar with Holm-Bonferroni correction, a noise floor, and leave-one-command-out stability. Unit tests now cover the metrics and statistics code.

## Engineering fixes (ROS 2 / MoveIt)

- `skill_server.py` had no handlers for 5 skills declared in `skills.json` (`inspect`, `unscrew`, `disconnect`, `waitForStabilization`, `rotateGripper`). They are now implemented with existing motion primitives, with a clamp on wrist rotation.
- Grasped objects are attached to the gripper's collision model (`scene_manager.attach_object` / `detach_object`).
- The Allowed Collision Matrix is merged instead of overwritten. Previously even the HOME pose was reported as colliding.
- `moveit_controllers.yaml` was missing its top-level `moveit_simple_controller_manager` key, so no controllers registered.
- A nested `rclpy.spin_once()` in `motion_executor.py` was stealing the action-client callback, which caused false timeouts.
- RQ5 now sends generated plans to the live MoveIt stack instead of scoring them only as text.
