# LLM-to-Action Planning for Robotic Battery Disassembly

A Gradio app turns natural-language disassembly requests into validated ROS 2 / MoveIt 2 actions for a Kinova Gen3 arm. The evaluation code is in `experiments/`.

[Watch the demo](https://youtu.be/JhGjJqNxm7w)

## Key findings

- RAG raised exact-match plan accuracy from 44.1% to 52.9%; the scripted baseline scored 64.7%.
- Simulated MoveIt execution succeeded for 17 of 18 achievable commands. No physical-robot validation is claimed.
- The text validator caught only 2 of 80 commands labelled unsafe. Do not treat it as a safety guarantee.

## Run evaluations

Install the Python requirements listed in `experiments/README.md`, then run the documented evaluation commands there. Unit tests for evaluation and validation:

```bash
python -m pytest experiments/eval/test_metrics.py experiments/eval/test_stats.py experiments/eval/test_validator.py experiments/eval/test_executor.py -q
```

The full robot system requires ROS 2, MoveIt 2 and the Kinova/Robotiq setup; see `README` sections in `src/` and `experiments/` for environment-specific instructions. API keys may incur provider charges.

## Limitations

Results are from scripted trials and simulated execution. The perception-noise study is geometric simulation, not camera hardware testing. Review `CHANGELOG.md` and `experiments/README.md` for evaluation definitions and corrections.
