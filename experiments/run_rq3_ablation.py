#!/usr/bin/env python3
"""
RQ3: Ablation Study (Section 3.7.6)

Research Question: What is the contribution of each system component?

Test Protocol:
- 20 commands from test_commands_minimal.json
- 5 planning configurations (see below)
- 3 trials per command
- Total: 20 × 5 × 3 = 300 trials

Planning Configurations:
1. Scripted Baseline (SB): Hand-coded plans for canonical tasks only
2. LLM-Only (LO): LLM planning without RAG and without validation
3. LLM+Validation (LV): LLM with two-tier validation but without RAG
4. LLM+RAG (LR): LLM with RAG retrieval but without validation
5. Full System (FS): Complete system (LLM + RAG + Validation)

Metrics (Section 3.7.2):
- Plan validity rate
- Execution success rate
- Generalization performance on paraphrased/varied commands
- Planning latency (wall-clock time)

This design enables measurement of:
- SB vs LO: benefit of LLM-based planning over scripted plans
- LO vs LV: isolated contribution of validation
- LO vs LR: isolated contribution of RAG retrieval
- FS vs LV/LR: synergistic effect of combining RAG and validation
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
from executor import Executor


@dataclass
class TrialResult:
    """Single trial result data."""
    trial_id: int
    configuration: str
    command: str
    category: str
    success: bool
    planning_time: float
    execution_time: float
    total_time: float
    plan_valid: bool
    plan_length: int
    error_type: Optional[str]
    error_message: Optional[str]
    planned_skills: List[Dict[str, Any]]
    ground_truth_match: bool
    timestamp: str


class ScriptedBaseline:
    """
    Baseline configuration using keyword-based skill mapping.

    This represents traditional robot control without LLM intelligence,
    using simple pattern matching to map commands to skill sequences.
    """

    def __init__(self, skills_path: str):
        """Initialize with predefined skill mappings."""
        with open(skills_path, 'r') as f:
            self.skills_config = json.load(f)

        # Hardcoded keyword → skill mappings
        self.mappings = {
            'home': [{'name': 'moveTo', 'params': {'target': 'HOME'}}],
            'grasp bolts': [
                {'name': 'moveTo', 'params': {'target': 'approach_bolts'}},
                {'name': 'openGripper', 'params': {}},
                {'name': 'moveTo', 'params': {'target': 'grasp_bolts'}},
                {'name': 'closeGripper', 'params': {}},
                {'name': 'moveTo', 'params': {'target': 'approach_bolts'}}
            ],
            'grasp battery': [
                {'name': 'moveTo', 'params': {'target': 'approach_battery'}},
                {'name': 'openGripper', 'params': {}},
                {'name': 'moveTo', 'params': {'target': 'grasp_battery'}},
                {'name': 'closeGripper', 'params': {}},
                {'name': 'moveTo', 'params': {'target': 'approach_battery'}}
            ],
            'release': [{'name': 'openGripper', 'params': {}}],
            'placement': [{'name': 'moveTo', 'params': {'target': 'place_bolts'}}],
        }

    def plan(self, command: str) -> List[Dict[str, Any]]:
        """
        Map command to skill sequence using keyword matching.

        Returns empty list if no match found (failure case).
        """
        command_lower = command.lower()

        # Try exact keyword matches
        for keyword, skills in self.mappings.items():
            if keyword in command_lower:
                return skills

        # No match found
        return []


class LLMOnlyPlanner:
    """
    LLM-Only configuration: GPT-4 planning WITHOUT RAG retrieval.

    This tests pure LLM planning ability without experience memory.
    """

    def __init__(self, config_dir: Path):
        """Initialize planner with RAG explicitly disabled."""
        self.planner = Planner(
            config_dir=config_dir,
            backend="openrouter",
            enable_rag=False  # KEY: Disable RAG for this configuration
        )

    def plan(self, command: str) -> List[Dict[str, Any]]:
        """Generate plan using LLM only."""
        result = asyncio.run(self.planner.plan(command))
        return result.get('plan', [])


class LLMWithValidationPlanner:
    """
    LLM + Validation (no RAG): Tests if validation helps without RAG.
    """

    def __init__(self, config_dir: Path):
        """Initialize planner with RAG disabled but validation enabled."""
        self.planner = Planner(
            config_dir=config_dir,
            backend="openrouter",
            enable_rag=False  # KEY: No RAG
        )

    def plan(self, command: str) -> List[Dict[str, Any]]:
        """Generate plan using LLM only (validation checked separately)."""
        result = asyncio.run(self.planner.plan(command))
        return result.get('plan', [])


class RAGWithoutValidationPlanner:
    """
    LLM + RAG (no Validation): Tests RAG benefit without validation overhead.
    """

    def __init__(self, config_dir: Path, rag_limit: Optional[int] = None):
        """Initialize planner with RAG enabled."""
        self.planner = Planner(
            config_dir=config_dir,
            backend="openrouter",
            enable_rag=True,  # KEY: Enable RAG
            rag_limit=rag_limit
        )

    def plan(self, command: str) -> List[Dict[str, Any]]:
        """Generate plan using LLM + RAG (validation skipped in experiment)."""
        result = asyncio.run(self.planner.plan(command))
        return result.get('plan', [])


class FullRAGPlanner:
    """
    Full System: LLM + RAG + Validation.

    This is the complete system as described in the dissertation.
    """

    def __init__(self, config_dir: Path, rag_limit: Optional[int] = None):
        """Initialize planner with RAG enabled."""
        self.planner = Planner(
            config_dir=config_dir,
            backend="openrouter",
            enable_rag=True,  # KEY: Enable RAG
            rag_limit=rag_limit  # For RQ2 memory size experiments
        )

    def plan(self, command: str) -> List[Dict[str, Any]]:
        """Generate plan using LLM + RAG + Validation."""
        result = asyncio.run(self.planner.plan(command))
        return result.get('plan', [])


class RQ1Experiment:
    """Main experiment coordinator for RQ1 ablation study."""

    def __init__(self,
                 test_commands_path: str,
                 skills_path: str,
                 prompt_path: str,
                 results_dir: str,
                 num_trials: int = 3,
                 selected_configs: Optional[List[str]] = None,
                 resume_from: Optional[str] = None):
        """
        Initialize experiment.

        Args:
            test_commands_path: Path to test_commands.json
            skills_path: Path to skills.json
            prompt_path: Path to prompt.txt
            results_dir: Directory to save results
            num_trials: Number of trials per command per configuration
            selected_configs: List of config names to run (default: all)
            resume_from: Path to previous results file to resume from
        """
        self.num_trials = num_trials
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load test commands
        with open(test_commands_path, 'r') as f:
            data = json.load(f)
            self.test_commands = data['commands']
            self.ground_truth = data.get('ground_truth', {})

        # Get config_dir from skills_path
        config_dir = Path(skills_path).parent

        # Initialize configurations (5 ablation levels)
        print("Initializing configurations...")
        all_configs = {
            'SB': ScriptedBaseline(skills_path),           # 1. Scripted Baseline
            'LO': LLMOnlyPlanner(config_dir),             # 2. LLM-Only
            'LV': LLMWithValidationPlanner(config_dir),  # 3. LLM + Validation
            'LR': RAGWithoutValidationPlanner(config_dir), # 4. LLM + RAG
            'FS': FullRAGPlanner(config_dir)           # 5. Full System
        }

        # Filter to selected configs if specified
        if selected_configs:
            self.configs = {k: v for k, v in all_configs.items() if k in selected_configs}
            print(f"Running configurations: {list(self.configs.keys())}")
        else:
            self.configs = all_configs

        # Initialize validator (shared across configs)
        self.validator = Validator(config_dir)

        # Results storage
        self.results: List[TrialResult] = []

        # Load previous results if resuming
        self.completed_tests = set()
        if resume_from and Path(resume_from).exists():
            self._load_previous_results(resume_from)

    def _load_previous_results(self, resume_path: str):
        """Load previous results from CSV and populate completed tests set."""
        import csv

        print(f"\n{'='*60}")
        print(f"Resuming from: {resume_path}")
        print(f"{'='*60}")

        with open(resume_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Create unique test identifier
                test_key = (
                    row['configuration'],
                    row['command'],
                    row['category'],
                    int(row['trial_id'])
                )
                self.completed_tests.add(test_key)

                # Reconstruct result object for final save
                result = TrialResult(
                    trial_id=int(row['trial_id']),
                    configuration=row['configuration'],
                    command=row['command'],
                    category=row['category'],
                    success=row['success'].lower() == 'true',
                    planning_time=float(row['planning_time']),
                    execution_time=float(row['execution_time']),
                    total_time=float(row['total_time']),
                    plan_valid=row['plan_valid'].lower() == 'true',
                    plan_length=int(row['plan_length']),
                    error_type=row['error_type'] if row['error_type'] else None,
                    error_message=row['error_message'] if row['error_message'] else None,
                    planned_skills=[],  # Not stored in CSV
                    ground_truth_match=row['ground_truth_match'].lower() == 'true',
                    timestamp=row['timestamp']
                )
                self.results.append(result)

        print(f"Loaded {len(self.results)} previous results")
        print(f"Completed tests: {len(self.completed_tests)}")
        print(f"{'='*60}\n")

    def _is_test_completed(self, config_name: str, command: str, category: str, trial_num: int) -> bool:
        """Check if a specific test has already been completed."""
        test_key = (config_name, command, category, trial_num)
        return test_key in self.completed_tests

    def _plans_equivalent(self, plan1: List[Dict], plan2: List[Dict]) -> bool:
        """
        Check if two plans are semantically equivalent.

        Flexible matching that allows:
        - Different step numbers (as long as order is same)
        - Semantically equivalent action sequences
        """
        if len(plan1) != len(plan2):
            return False

        for s1, s2 in zip(plan1, plan2):
            # Compare name and params, ignore step field
            if s1.get('name') != s2.get('name'):
                return False
            if s1.get('params') != s2.get('params'):
                return False

        return True

    def run_single_trial(self,
                         config_name: str,
                         command: str,
                         category: str,
                         trial_num: int) -> TrialResult:
        """
        Execute single trial and record metrics.

        Returns:
            TrialResult with all metrics populated
        """
        planner = self.configs[config_name]

        # Measure planning time
        start_time = time.time()
        try:
            planned_skills = planner.plan(command)
            planning_time = time.time() - start_time

            # Validate plan
            plan_valid = True
            error_type = None
            error_message = None

            if not planned_skills:
                plan_valid = False
                error_type = "NO_PLAN"
                error_message = "Planner returned empty skill sequence"
            else:
                # Validation depends on configuration
                # Configs WITHOUT validation: SB, LO, LR
                # Configs WITH validation: LV, FS
                if config_name in ['SB', 'LO', 'LR']:
                    # Skip validation for these configs
                    plan_valid = True  # Assume valid if plan exists
                    error_type = None
                    error_message = None
                else:
                    # Run validator - validate_plan expects {'plan': [...]} format
                    plan_valid, errors = self.validator.validate_plan({'plan': planned_skills})
                    if not plan_valid:
                        error_type = "VALIDATION_FAILED"
                        error_message = '; '.join(errors) if errors else 'Unknown validation error'

            # Check ground truth match (if available)
            ground_truth_match = False
            if command in self.ground_truth:
                expected = self.ground_truth[command]
                # Flexible matching: compare key fields (name, params) ignoring step order differences
                ground_truth_match = self._plans_equivalent(planned_skills, expected)

            # Real execution using Executor
            if plan_valid and len(planned_skills) > 0:
                try:
                    from executor import Executor

                    executor = Executor(use_ros=False)  # Use Mock mode for speed
                    exec_start = time.time()

                    # Execute the plan
                    execution_result = executor.execute({'plan': planned_skills})
                    execution_time = time.time() - exec_start

                    # Success only if execution completed without errors
                    success = execution_result.get('success', False)

                    if not success:
                        error_type = "EXECUTION_FAILED"
                        error_message = execution_result.get('error', 'Unknown execution error')

                    # Cleanup
                    executor.shutdown()

                except Exception as exec_error:
                    execution_time = 0.0
                    success = False
                    error_type = "EXECUTION_EXCEPTION"
                    error_message = str(exec_error)
                    print(f"   Execution error: {exec_error}")
            else:
                # No valid plan to execute
                execution_time = 0.0
                success = False

            total_time = planning_time + execution_time

        except Exception as e:
            planning_time = time.time() - start_time
            planned_skills = []
            plan_valid = False
            error_type = "EXCEPTION"
            error_message = str(e)
            execution_time = 0.0
            total_time = planning_time
            success = False
            ground_truth_match = False

        return TrialResult(
            trial_id=trial_num,
            configuration=config_name,
            command=command,
            category=category,
            success=success,
            planning_time=planning_time,
            execution_time=execution_time,
            total_time=total_time,
            plan_valid=plan_valid,
            plan_length=len(planned_skills),
            error_type=error_type,
            error_message=error_message,
            planned_skills=planned_skills,
            ground_truth_match=ground_truth_match,
            timestamp=datetime.now().isoformat()
        )

    def run_all_trials(self):
        """Execute complete RQ3 experiment."""
        total_tests = sum(len(cmds) for cmds in self.test_commands.values()) * len(self.configs) * self.num_trials
        current_test = 0

        print(f"\n{'='*60}")
        print(f"Starting RQ3 Ablation Study")
        print(f"{'='*60}")
        print(f"Total tests: {total_tests}")
        print(f"Configurations: {len(self.configs)}")
        print(f"Trials per command: {self.num_trials}")
        print(f"Results will be saved to: {self.results_dir}")
        print(f"{'='*60}\n")

        for category, commands in self.test_commands.items():
            print(f"\nCategory: {category.upper()} ({len(commands)} commands)")

            for cmd_idx, command in enumerate(commands, 1):
                print(f"\n  [{cmd_idx}/{len(commands)}] Testing: \"{command}\"")

                for config_name in self.configs.keys():
                    print(f"    Config: {config_name}")

                    for trial in range(1, self.num_trials + 1):
                        current_test += 1
                        progress = (current_test / total_tests) * 100

                        # Check if test already completed
                        if self._is_test_completed(config_name, command, category, trial):
                            print(f"      Trial {trial}/{self.num_trials} [SKIPPED] (already completed) ({progress:.1f}%)")
                            continue

                        print(f"      Trial {trial}/{self.num_trials} ", end='', flush=True)

                        result = self.run_single_trial(
                            config_name=config_name,
                            command=command,
                            category=category,
                            trial_num=trial
                        )

                        self.results.append(result)

                        # Print result
                        status = "✓ PASS" if result.success else "✗ FAIL"
                        print(f"[{status}] {result.planning_time:.2f}s ({progress:.1f}%)")

                        # Save intermediate results every 50 tests
                        if current_test % 50 == 0:
                            self.save_results()

        # Final save
        self.save_results()
        print(f"\n{'='*60}")
        print(f"Experiment completed! {len(self.results)} trials recorded.")
        print(f"{'='*60}\n")

    def save_results(self):
        """Save results to CSV and JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as CSV
        csv_path = self.results_dir / f"rq3_results_{timestamp}.csv"
        with open(csv_path, 'w', newline='') as f:
            if self.results:
                fieldnames = list(asdict(self.results[0]).keys())
                # Exclude complex fields from CSV
                fieldnames.remove('planned_skills')

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in self.results:
                    row = asdict(result)
                    row.pop('planned_skills')
                    writer.writerow(row)

        # Save as JSON (with full data)
        json_path = self.results_dir / f"rq3_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

        print(f"Results saved: {csv_path}")


def main():
    """Run RQ1 experiment."""
    import argparse

    parser = argparse.ArgumentParser(description='RQ3 Ablation Study')
    parser.add_argument('--commands', default='test_commands_minimal.json',
                        help='Test commands file (default: test_commands_minimal.json - 20 commands)')
    parser.add_argument('--trials', type=int, default=3,
                        help='Number of trials per command (default: 3 per Section 3.7.6)')
    parser.add_argument('--configs', type=str, nargs='+', default=None,
                        help='Specific configurations to run: SB LO LV LR FS (default: all)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from previous results file (CSV path)')
    args = parser.parse_args()

    # Paths
    base_dir = Path(__file__).parent.parent
    test_commands = base_dir / 'experiments' / args.commands
    skills_json = base_dir / 'src' / 'llm_agent' / 'config' / 'skills.json'
    prompt_txt = base_dir / 'src' / 'llm_agent' / 'config' / 'prompt.txt'
    results_dir = base_dir / 'experiments' / 'results'

    # Check files exist
    for path in [test_commands, skills_json, prompt_txt]:
        if not path.exists():
            print(f"ERROR: Required file not found: {path}")
            sys.exit(1)

    # Create experiment
    experiment = RQ1Experiment(
        test_commands_path=str(test_commands),
        skills_path=str(skills_json),
        prompt_path=str(prompt_txt),
        results_dir=str(results_dir),
        num_trials=args.trials,
        selected_configs=args.configs,
        resume_from=args.resume
    )

    # Run experiment
    try:
        experiment.run_all_trials()
        print("\n✓ Experiment completed successfully!")
        print(f"  Results: {results_dir}")
        print(f"  Total trials: {len(experiment.results)}")
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        experiment.save_results()
        print("Partial results saved.")
    except Exception as e:
        print(f"\n\nERROR: Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        experiment.save_results()
        sys.exit(1)


if __name__ == '__main__':
    main()
