#!/usr/bin/env python3
"""
RQ1: Safety Validation Experiment (Section 3.7.3)

Research Question: Does two-tier validation improve safety?

Test Protocol:
- 35 commands from unified test suite
- 4 validation levels: NV, SV, RV, FV
- 3 trials per command
- Total: 35 × 4 × 3 = 420 trials

Validation Levels:
- NV (No Validation): Plans forwarded directly to execution
- SV (Schema-based Validation): Enforces structural correctness of generated plans
- RV (Rule-based Validation): Checks sequence constraints based on safety rules
- FV (Full Validation): Combines schema checks with rule-based constraints

Metrics (Section 3.7.2):
- M1: Invalid plans executed
- M2: Unsafe execution attempts
- M3: Unsafe plans blocked
- M4: False positives (valid plans incorrectly blocked)
- M5: Added latency (validation overhead in ms)
"""

import sys
import os
import json
import time
import csv
import asyncio
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'llm_agent'))

from planner import Planner
from validator import Validator


@dataclass
class RQ1TrialResult:
    """RQ1 trial result with safety-specific metrics."""
    trial_id: int
    validation_level: str
    command: str
    is_unsafe_test: bool
    category: str
    plan_generated: bool
    plan_valid: bool
    validation_time: float
    safety_violations: List[str]
    false_positive: bool
    plan_length: int
    error_type: Optional[str]
    planned_skills: List[Dict[str, Any]]
    timestamp: str


class NoValidation:
    """No validation - all plans accepted."""

    def validate(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Accept all plans."""
        return {
            'valid': True,
            'violations': []
        }


class SchemaOnlyValidation:
    """Schema validation only - check skill names, parameters, but NO sequence constraints."""

    def __init__(self, skills_path: str):
        """Initialize with skill definitions."""
        with open(skills_path, 'r') as f:
            self.skills = json.load(f)

        # Extract valid entities
        self.valid_skills = {s['name'] for s in self.skills.get('available_skills', [])}
        self.valid_poses = set(self.skills.get('available_poses', []))
        self.valid_objects = set(self.skills.get('available_objects', []))

    def validate(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Schema-only validation: check format, skill names, parameters.
        Does NOT check sequence constraints.
        """
        violations = []

        for i, step in enumerate(plan):
            # Check required fields
            if 'name' not in step:
                violations.append(f"Step {i+1}: missing 'name'")
                continue

            if 'params' not in step:
                violations.append(f"Step {i+1}: missing 'params'")
                continue

            skill_name = step.get('name')
            params = step.get('params', {})

            # Check skill exists
            if skill_name not in self.valid_skills:
                violations.append(f"Step {i+1}: invalid skill '{skill_name}'")
                continue

            # Check required parameters
            if skill_name in ["openGripper", "closeGripper"]:
                # These don't need target parameter
                pass
            else:
                # moveTo, grasp, release need target
                if 'target' not in params:
                    violations.append(f"Step {i+1}: missing 'target' for {skill_name}")
                    continue

                target = params['target']

                # Validate target type
                if skill_name == "moveTo":
                    if target not in self.valid_poses:
                        violations.append(f"Step {i+1}: invalid pose '{target}' (not in available_poses)")
                elif skill_name in ["grasp", "release"]:
                    if target not in self.valid_objects:
                        violations.append(f"Step {i+1}: invalid object '{target}' (not in available_objects)")

        return {
            'valid': len(violations) == 0,
            'violations': violations
        }


class RuleBasedValidation:
    """Rule-based validation only - check sequence constraints but NOT schema."""

    def __init__(self, validator: Validator):
        """Initialize with Validator instance."""
        self.validator = validator

    def validate(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Rule-based validation: check sequence constraints only.
        Does NOT check schema (skill names, parameters).
        """
        # Only check sequence constraints, not schema
        violations = self.validator._check_sequence_constraints(plan)

        return {
            'valid': len(violations) == 0,
            'violations': violations
        }


class FullValidationWrapper:
    """Wrapper for Validator.validate_plan() to match RQ1 interface."""

    def __init__(self, validator: Validator):
        """Initialize with Validator instance."""
        self.validator = validator

    def validate(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate using full validation (schema + rules)."""
        # Wrap plan in dict format expected by Validator
        plan_dict = {"plan": plan}
        is_valid, errors = self.validator.validate_plan(plan_dict)
        return {
            'valid': is_valid,
            'violations': errors
        }


class RQ1Experiment:
    """Safety validation experiment."""

    def __init__(self,
                 test_commands_path: str,
                 skills_path: str,
                 prompt_path: str,
                 results_dir: str,
                 num_trials: int = 5):
        """
        Initialize RQ1 experiment.

        Args:
            test_commands_path: JSON file with all test commands (unified suite)
        """
        self.num_trials = num_trials
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load test commands from unified suite
        with open(test_commands_path, 'r') as f:
            data = json.load(f)
            self.safe_commands = data['commands']

        # Get config_dir
        config_dir = Path(skills_path).parent

        # Initialize validator instance for RV and FV
        validator = Validator(config_dir)

        # Initialize validators (using paper abbreviations)
        self.validators = {
            'NV': NoValidation(),
            'SV': SchemaOnlyValidation(skills_path),
            'RV': RuleBasedValidation(validator),
            'FV': FullValidationWrapper(validator)
        }

        # Initialize planner (with RAG for realistic plans)
        self.planner = Planner(
            config_dir=config_dir,
            backend="openrouter",
            enable_rag=True
        )

        # Results
        self.results: List[RQ1TrialResult] = []

    async def run_single_trial(self,
                         validation_level: str,
                         command: str,
                         is_unsafe: bool,
                         category: str,
                         trial_num: int) -> RQ1TrialResult:
        """Execute single safety validation trial."""
        validator = self.validators[validation_level]

        # Generate plan using LLM planner for all commands
        try:
            result = await self.planner.plan(command, use_llm=True)
            planned_skills = result.get('plan', [])
            plan_generated = len(planned_skills) > 0
        except:
            planned_skills = []
            plan_generated = False

        # Validate plan
        validation_start = time.time()
        validation_result = validator.validate(planned_skills)
        validation_time = time.time() - validation_start

        plan_valid = validation_result.get('valid', False)
        violations = validation_result.get('violations', [])

        # Determine false positive (functional command incorrectly rejected)
        # Stress test commands are expected to have higher failure rates
        false_positive = category.startswith('functional') and (not plan_valid)

        return RQ1TrialResult(
            trial_id=trial_num,
            validation_level=validation_level,
            command=command,
            is_unsafe_test=is_unsafe,
            category=category,
            plan_generated=plan_generated,
            plan_valid=plan_valid,
            validation_time=validation_time,
            safety_violations=violations,
            false_positive=false_positive,
            plan_length=len(planned_skills),
            error_type=violations[0] if violations else None,
            planned_skills=planned_skills,
            timestamp=datetime.now().isoformat()
        )

    async def run_all_trials(self):
        """Execute complete RQ1 experiment."""
        # Combine safe and unsafe commands
        all_tests = []

        # Add safe commands
        for category, commands in self.safe_commands.items():
            for cmd in commands:
                all_tests.append({
                    'command': cmd,
                    'is_unsafe': False,
                    'category': category
                })

        total_tests = len(all_tests) * len(self.validators) * self.num_trials
        current_test = 0

        print(f"\n{'='*60}")
        print(f"Starting RQ1 Safety Validation Experiment")
        print(f"{'='*60}")
        print(f"Total commands: {sum(len(c) for c in self.safe_commands.values())}")
        print(f"Validation levels: {len(self.validators)}")
        print(f"Total tests: {total_tests}")
        print(f"{'='*60}\n")

        for test_idx, test_case in enumerate(all_tests, 1):
            command = test_case['command']
            is_unsafe = test_case['is_unsafe']
            category = test_case['category']

            print(f"\n[{test_idx}/{len(all_tests)}] \"{command}\" ({'UNSAFE' if is_unsafe else 'SAFE'})")

            for val_level in self.validators.keys():
                print(f"  Validation: {val_level}")

                for trial in range(1, self.num_trials + 1):
                    current_test += 1
                    progress = (current_test / total_tests) * 100

                    print(f"    Trial {trial}/{self.num_trials} ", end='', flush=True)

                    result = await self.run_single_trial(
                        validation_level=val_level,
                        command=command,
                        is_unsafe=is_unsafe,
                        category=category,
                        trial_num=trial
                    )

                    self.results.append(result)

                    # Expected behavior
                    if is_unsafe:
                        # Should be caught
                        status = "✓ CAUGHT" if not result.plan_valid else "✗ MISSED"
                    else:
                        # Should pass
                        status = "✓ PASS" if result.plan_valid else "✗ REJECT"

                    print(f"[{status}] {result.validation_time*1000:.1f}ms ({progress:.1f}%)")

                    if current_test % 50 == 0:
                        self.save_results()

        self.save_results()
        print(f"\n{'='*60}")
        print(f"RQ1 Experiment completed! {len(self.results)} trials.")
        print(f"{'='*60}\n")

    def save_results(self):
        """Save RQ1 results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV
        csv_path = self.results_dir / f"rq1_results_{timestamp}.csv"
        with open(csv_path, 'w', newline='') as f:
            if self.results:
                fieldnames = list(asdict(self.results[0]).keys())
                fieldnames.remove('planned_skills')
                fieldnames.remove('safety_violations')

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in self.results:
                    row = asdict(result)
                    row.pop('planned_skills')
                    row.pop('safety_violations')
                    writer.writerow(row)

        # JSON
        json_path = self.results_dir / f"rq1_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

        print(f"Results saved: {csv_path}")


async def main_async():
    """Async main function."""
    import argparse

    parser = argparse.ArgumentParser(description='RQ1 Safety Validation Experiment')
    parser.add_argument('--commands', default='unified_test_suite.json',
                        help='Test commands file (default: unified_test_suite.json)')
    parser.add_argument('--trials', type=int, default=3,
                        help='Number of trials per command (default: 3)')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    test_commands = base_dir / 'experiments' / args.commands
    skills_json = base_dir / 'src' / 'llm_agent' / 'config' / 'skills.json'
    prompt_txt = base_dir / 'src' / 'llm_agent' / 'config' / 'prompt.txt'
    results_dir = base_dir / 'experiments' / 'results'

    for path in [test_commands, skills_json, prompt_txt]:
        if not path.exists():
            print(f"ERROR: {path} not found")
            sys.exit(1)

    experiment = RQ1Experiment(
        test_commands_path=str(test_commands),
        skills_path=str(skills_json),
        prompt_path=str(prompt_txt),
        results_dir=str(results_dir),
        num_trials=args.trials
    )

    try:
        await experiment.run_all_trials()
        print("\n✓ RQ1 Experiment completed!")
        print(f"  Results: {results_dir}")
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        experiment.save_results()
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        experiment.save_results()
        sys.exit(1)


def main():
    """Synchronous entry point."""
    asyncio.run(main_async())


if __name__ == '__main__':
    main()
