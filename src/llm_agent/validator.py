"""
Two-tier validation - Schema + Runtime.
"""
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple


class Validator:
    """Plan and runtime validator."""

    def __init__(self, config_dir: Path = None):
        """Load the skill and safety configuration."""
        if config_dir is None:
            config_dir = Path(__file__).parent / "config"

        # Load configuration
        with open(config_dir / "skills.json") as f:
            self.skills_config = json.load(f)

        with open(config_dir / "safety.yaml") as f:
            self.safety_config = yaml.safe_load(f)

        self.valid_skills = {s['name'] for s in self.skills_config['available_skills']}
        self.valid_poses = set(self.skills_config['available_poses'])
        self.valid_objects = set(self.skills_config['available_objects'])

        # Required params per skill, read straight from skills.json so the
        # validator can't drift out of sync with the skill definitions.
        self.required_params = {
            s['name']: s.get('required_params', [])
            for s in self.skills_config['available_skills']
        }

        # How each skill's "target" param is interpreted. moveTo takes a named
        # pose; grasp/release/inspect/unscrew/disconnect all act on a physical
        # object. This is what lets validate_plan reject "unscrew(HOME)" or
        # "moveTo(TopCoverBolts)" instead of only checking that *some* target
        # string is present.
        self.pose_target_skills = {"moveTo"}
        self.object_target_skills = {
            "grasp", "release", "inspect", "unscrew", "disconnect",
        }

    def validate_plan(self, plan: Dict) -> Tuple[bool, List[str]]:
        """
        Schema validation - check plan format and content.

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # 1. Check format
        if 'plan' not in plan:
            errors.append("Missing 'plan' key")
            return False, errors

        actions = plan['plan']
        if not isinstance(actions, list):
            errors.append("'plan' must be a list")
            return False, errors

        # 2. Check each action
        for i, action in enumerate(actions):
            # Check required fields
            if 'name' not in action:
                errors.append(f"Step {i+1}: missing 'name'")
                continue

            if 'params' not in action:
                errors.append(f"Step {i+1}: missing 'params'")
                continue

            skill_name = action['name']
            params = action['params']

            # Check the skill exists
            if skill_name not in self.valid_skills:
                errors.append(f"Step {i+1}: invalid skill '{skill_name}'")
                continue

            # 1) Every required parameter declared for the skill must be present (from skills.json)
            missing = [p for p in self.required_params[skill_name]
                       if p not in params]
            if missing:
                errors.append(
                    f"Step {i+1}: {skill_name} missing "
                    f"param(s): {', '.join(missing)}"
                )
                continue

            # 2) Check parameter values are legal for the skill type
            if skill_name in self.pose_target_skills:
                target = params['target']
                if target not in self.valid_poses:
                    errors.append(f"Step {i+1}: invalid pose '{target}'")
            elif skill_name in self.object_target_skills:
                target = params['target']
                if target not in self.valid_objects:
                    errors.append(
                        f"Step {i+1}: {skill_name} needs an object target, "
                        f"got invalid/pose value '{target}'"
                    )
            elif skill_name == "rotateGripper":
                angle = params['angle']
                try:
                    float(angle)
                except (TypeError, ValueError):
                    errors.append(
                        f"Step {i+1}: rotateGripper angle must be numeric, "
                        f"got '{angle}'"
                    )

        # 3. Check sequence constraints
        errors.extend(self._check_sequence_constraints(actions))

        return len(errors) == 0, errors

    def _check_sequence_constraints(self, actions: List[Dict]) -> List[str]:
        """Check skill-sequence constraints."""
        errors = []

        # Check forbidden sequences
        for i in range(len(actions) - 1):
            current = actions[i]['name']
            next_action = actions[i+1]['name']

            for forbidden in self.safety_config['forbidden_sequences']:
                if [current, next_action] == forbidden:
                    errors.append(
                        f"Forbidden sequence: {current} → {next_action}"
                    )

        # 'Must end at HOME' rule - currently disabled
        # if actions and actions[-1]['params'].get('target') != 'HOME':
        #     errors.append("Plan must end at HOME pose")

        return errors

    def validate_runtime(self, force: float, temperature: float) -> Tuple[bool, str]:
        """
        Runtime validation - check sensor data (not wired to any sensor).

        Returns:
            (is_safe, message)
        """
        limits = self.safety_config['runtime_limits']

        if force > limits['force_max']:
            return False, f"Force {force}N exceeds limit {limits['force_max']}N"

        if temperature > limits['temperature_max']:
            return False, f"Temperature {temperature}°C exceeds limit {limits['temperature_max']}°C"

        return True, "Runtime checks passed"


# Manual smoke test
def test():
    validator = Validator()

    # Valid plan
    valid_plan = {
        "plan": [
            {"step": 1, "name": "moveTo", "params": {"target": "HOME"}},
            {"step": 2, "name": "grasp", "params": {"target": "TopCoverBolts"}},
            {"step": 3, "name": "moveTo", "params": {"target": "HOME"}}
        ]
    }

    is_valid, errors = validator.validate_plan(valid_plan)
    print(f"Valid plan: {is_valid}")
    if errors:
        for error in errors:
            print(f"  - {error}")

    # Invalid plan
    invalid_plan = {
        "plan": [
            {"step": 1, "name": "grasp", "params": {"target": "InvalidObject"}},
            {"step": 2, "name": "grasp", "params": {"target": "TopCoverBolts"}}
        ]
    }

    is_valid, errors = validator.validate_plan(invalid_plan)
    print(f"\nInvalid plan: {is_valid}")
    for error in errors:
        print(f"  - {error}")

    # Runtime check
    safe, msg = validator.validate_runtime(force=15.0, temperature=30.0)
    print(f"\nRuntime check: {safe} - {msg}")


if __name__ == "__main__":
    test()
