"""Test command serialization and immediate feedback without a ROS install."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src' / 'llm_agent'))
import executor as module


@pytest.mark.parametrize('action', [
    {'name': 'rotateGripper', 'params': {'angle': -45}},
    {'name': 'waitForStabilization', 'params': {'seconds': 2}},
    {'name': 'moveTo', 'params': {'target': 'HOME'}},
])
def test_ros_payload_preserves_parameters_and_immediate_feedback(monkeypatch, action):
    executor = module.Executor(use_ros=False)
    messages = []
    monkeypatch.setattr(module, 'String', SimpleNamespace)
    def publish(message):
        messages.append(json.loads(message.data))
        executor.last_feedback = '{"status": "success"}'
    executor.command_pub = SimpleNamespace(publish=publish)
    assert executor._execute_ros(action, timeout=0)
    assert messages[0]['params'] == action['params']
    assert messages[0]['skill'] == action['name']
    if 'target' in action['params']:
        assert messages[0]['target'] == action['params']['target']
