"""Malformed model outputs must be refused before they reach execution."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src' / 'llm_agent'))
from validator import Validator


@pytest.mark.parametrize('plan', [None, [], {'plan': None}, {'plan': [None]},
    {'plan': [{} , {}]}, {'plan': [{'name': [], 'params': {}}]},
    {'plan': [{'name': 'moveTo', 'params': []}]},
    {'plan': [{'name': 'moveTo', 'params': {'target': []}}]}])
def test_malformed_plan_is_rejected_without_raising(plan):
    valid, errors = Validator().validate_plan(plan)
    assert not valid and errors


@pytest.mark.parametrize('angle', [float('nan'), float('inf'), 'NaN', True, None])
def test_rotation_rejects_nonfinite_angles(angle):
    assert not Validator().validate_plan(
        {'plan': [{'name': 'rotateGripper', 'params': {'angle': angle}}]})[0]


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), None, '3', True])
def test_runtime_rejects_invalid_sensor_readings(value):
    assert not Validator().validate_runtime(value, 20)[0]
    assert not Validator().validate_runtime(0, value)[0]


def test_valid_plan_and_sensor_readings_are_accepted():
    validator = Validator()
    assert validator.validate_plan(
        {'plan': [{'name': 'moveTo', 'params': {'target': 'HOME'}}]})[0]
    assert validator.validate_runtime(0, 20)[0]
