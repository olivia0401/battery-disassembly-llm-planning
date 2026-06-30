#!/usr/bin/env python3
"""
RQ2: Memory Size Experiment (Section 3.7.4)

Research Question: What is the effect of retrieval memory size?

Test Protocol:
- 20 commands from test_commands_minimal.json
- 4 memory sizes: k=0 (no retrieval), k=10, k=20, k=35 (full memory)
- 3 trials per command
- Total: 20 × 4 × 3 = 240 trials

Metrics (Section 3.7.2):
- Plan validity rate as function of k
- Execution success rate for seen vs. unseen tasks
- Generalization performance (primitive/bolt_removal/short_disassembly)

Seen vs. Unseen Definition (Section 3.7.4):
- Seen: command intent matches at least one episode in retrieval memory
- Unseen: novel paraphrases/variants with no corresponding episode
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
class RQ2TrialResult:
    """RQ2 trial result with memory-specific metrics."""
    trial_id: int
    memory_size: int
    command: str
    category: str
    success: bool
    planning_time: float
    retrieval_time: float
    num_cases_retrieved: int
    plan_valid: bool
    plan_length: int
    error_type: Optional[str]
    planned_skills: List[Dict[str, Any]]
    is_seen: bool  # NEW: Whether command has similar case in memory
    max_similarity: float  # NEW: Highest similarity score from retrieval
    timestamp: str


class RQ2Experiment:
    """Memory size experiment for RQ2."""

    def __init__(self,
                 test_commands_path: str,
                 skills_path: str,
                 prompt_path: str,
                 results_dir: str,
                 memory_sizes: List[int] = [0, 10, 20, 35],
                 num_trials: int = 5,
                 random_seed: int = 42,
                 resume_from: Optional[str] = None):
        """
        Initialize RQ2 experiment.

        Args:
            memory_sizes: List of RAG memory limits to test
            random_seed: Seed for deterministic case sampling
            resume_from: Path to previous results file to resume from
        """
        self.memory_sizes = memory_sizes
        self.num_trials = num_trials
        self.random_seed = random_seed
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load test commands
        with open(test_commands_path, 'r') as f:
            data = json.load(f)
            self.test_commands = data['commands']

        # Initialize validator
        self.config_dir = Path(skills_path).parent
        self.validator = Validator(self.config_dir)
        self.skills_path = skills_path
        self.prompt_path = prompt_path

        # Results storage
        self.results: List[RQ2TrialResult] = []

        # Load previous results if resuming
        self.completed_tests = set()
        if resume_from and Path(resume_from).exists():
            self._load_previous_results(resume_from)

        # Similarity threshold for seen/unseen classification
        self.seen_threshold = 0.7  # Commands with similarity >= 0.7 are "seen"

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
                    int(row['memory_size']),
                    row['command'],
                    row['category'],
                    int(row['trial_id'])
                )
                self.completed_tests.add(test_key)

                # Reconstruct result object for final save
                result = RQ2TrialResult(
                    trial_id=int(row['trial_id']),
                    memory_size=int(row['memory_size']),
                    command=row['command'],
                    category=row['category'],
                    success=row['success'].lower() == 'true',
                    planning_time=float(row['planning_time']),
                    retrieval_time=float(row['retrieval_time']),
                    num_cases_retrieved=int(row['num_cases_retrieved']),
                    plan_valid=row['plan_valid'].lower() == 'true',
                    plan_length=int(row['plan_length']),
                    error_type=row['error_type'] if row['error_type'] else None,
                    planned_skills=[],  # Not stored in CSV
                    is_seen=row['is_seen'].lower() == 'true',
                    max_similarity=float(row['max_similarity']),
                    timestamp=row['timestamp']
                )
                self.results.append(result)

        print(f"Loaded {len(self.results)} previous results")
        print(f"Completed tests: {len(self.completed_tests)}")
        print(f"{'='*60}\n")

    def _is_test_completed(self, memory_size: int, command: str, category: str, trial_num: int) -> bool:
        """Check if a specific test has already been completed."""
        test_key = (memory_size, command, category, trial_num)
        return test_key in self.completed_tests

    def _check_if_seen(self, planner: Planner, command: str) -> tuple[bool, float]:
        """
        Check if command is 'seen' based on RAG memory similarity.

        Args:
            planner: Planner instance with RAG engine
            command: Natural language command

        Returns:
            (is_seen, max_similarity): True if similar case exists in memory
        """
        try:
            # Access RAG engine if available
            if not hasattr(planner, 'rag') or planner.rag is None:
                return False, 0.0

            rag_engine = planner.rag
            if not getattr(rag_engine, 'enabled', False):
                return False, 0.0

            # Check collection count
            try:
                if rag_engine.collection.count() == 0:
                    return False, 0.0
            except:
                return False, 0.0

            # Retrieve top similar cases
            similar_cases = rag_engine.retrieve_similar_cases(command, n_results=3)

            if not similar_cases:
                return False, 0.0

            # Get max similarity score
            max_sim = max(case.get('similarity_score', 0.0) for case in similar_cases)

            # Consider 'seen' if similarity >= threshold
            is_seen = max_sim >= self.seen_threshold

            return is_seen, max_sim

        except Exception as e:
            print(f"    Warning: Could not check seen/unseen status: {e}")
            return False, 0.0

    def create_planner(self, memory_size: int) -> Planner:
        """
        Create planner with specific memory size configuration.

        Args:
            memory_size: Number of RAG cases to use (0 = disable RAG)
        """
        if memory_size == 0:
            # No memory = LLM-Only
            return Planner(
                config_dir=self.config_dir,
                backend="openrouter",
                enable_rag=False
            )
        else:
            # RAG enabled with limited memory
            return Planner(
                config_dir=self.config_dir,
                backend="openrouter",
                enable_rag=True,
                rag_limit=memory_size,
                rag_seed=self.random_seed  # Deterministic sampling
            )

    async def run_single_trial(self,
                         memory_size: int,
                         command: str,
                         category: str,
                         trial_num: int) -> RQ2TrialResult:
        """Execute single trial with specific memory configuration."""
        planner = self.create_planner(memory_size)

        # Check if command is "seen" in RAG memory (before planning)
        is_seen, max_similarity = self._check_if_seen(planner, command)

        start_time = time.time()
        try:
            # Plan task (with timing for retrieval)
            retrieval_start = time.time()
            result = await planner.plan(command, use_llm=True)
            planning_time = time.time() - start_time

            planned_skills = result.get('plan', [])

            # Get retrieval info from meta
            meta = result.get('meta', {})
            retrieval_info = meta.get('retrieval', {})
            num_cases_retrieved = retrieval_info.get('k_returned', 0)

            # Retrieval time is part of planning time (not separately tracked)
            retrieval_time = meta.get('timing', {}).get('planning_wall_s', 0.0)

            # Validate
            plan_valid = True
            error_type = None

            if not planned_skills:
                plan_valid = False
                error_type = "NO_PLAN"
            else:
                plan_dict = {"plan": planned_skills}
                is_valid, errors = self.validator.validate_plan(plan_dict)
                plan_valid = is_valid
                if not plan_valid:
                    error_type = "VALIDATION_FAILED"

            # Real execution
            execution_time = 0.0
            if plan_valid and len(planned_skills) > 0:
                try:
                    from executor import Executor

                    executor = Executor(use_ros=False)  # Use Mock mode for speed
                    exec_start = time.time()
                    execution_result = executor.execute({'plan': planned_skills})
                    execution_time = time.time() - exec_start
                    success = execution_result.get('success', False)

                    if not success:
                        error_type = "EXECUTION_FAILED"

                    # Cleanup
                    executor.shutdown()
                except Exception as exec_error:
                    success = False
                    error_type = "EXECUTION_EXCEPTION"
                    print(f"   Execution error: {exec_error}")
            else:
                success = False

        except Exception as e:
            planning_time = time.time() - start_time
            retrieval_time = 0.0
            num_cases_retrieved = 0
            planned_skills = []
            plan_valid = False
            error_type = "EXCEPTION"
            success = False

        return RQ2TrialResult(
            trial_id=trial_num,
            memory_size=memory_size,
            command=command,
            category=category,
            success=success,
            planning_time=planning_time,
            retrieval_time=retrieval_time,
            num_cases_retrieved=num_cases_retrieved,
            plan_valid=plan_valid,
            plan_length=len(planned_skills),
            error_type=error_type,
            planned_skills=planned_skills,
            is_seen=is_seen,  # NEW
            max_similarity=max_similarity,  # NEW
            timestamp=datetime.now().isoformat()
        )

    async def run_all_trials(self):
        """Execute complete RQ2 experiment."""
        total_commands = sum(len(cmds) for cmds in self.test_commands.values())
        total_tests = total_commands * len(self.memory_sizes) * self.num_trials
        current_test = 0

        print(f"\n{'='*60}")
        print(f"Starting RQ2 Memory Size Experiment")
        print(f"{'='*60}")
        print(f"Memory sizes: {self.memory_sizes}")
        print(f"Total tests: {total_tests}")
        print(f"Random seed: {self.random_seed}")
        print(f"{'='*60}\n")

        for category, commands in self.test_commands.items():
            print(f"\nCategory: {category.upper()} ({len(commands)} commands)")

            for cmd_idx, command in enumerate(commands, 1):
                print(f"\n  [{cmd_idx}/{len(commands)}] \"{command}\"")

                for memory_size in self.memory_sizes:
                    print(f"    Memory k={memory_size}")

                    for trial in range(1, self.num_trials + 1):
                        current_test += 1
                        progress = (current_test / total_tests) * 100

                        # Check if test already completed
                        if self._is_test_completed(memory_size, command, category, trial):
                            print(f"      Trial {trial}/{self.num_trials} [SKIPPED] (already completed) ({progress:.1f}%)")
                            continue

                        print(f"      Trial {trial}/{self.num_trials} ", end='', flush=True)

                        result = await self.run_single_trial(
                            memory_size=memory_size,
                            command=command,
                            category=category,
                            trial_num=trial
                        )

                        self.results.append(result)

                        status = "✓" if result.success else "✗"
                        print(f"[{status}] {result.planning_time:.2f}s ({progress:.1f}%)")

                        if current_test % 50 == 0:
                            self.save_results()

        self.save_results()
        print(f"\n{'='*60}")
        print(f"RQ2 Experiment completed! {len(self.results)} trials.")
        print(f"{'='*60}\n")

    def save_results(self):
        """Save RQ2 results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV
        csv_path = self.results_dir / f"rq2_results_{timestamp}.csv"
        with open(csv_path, 'w', newline='') as f:
            if self.results:
                fieldnames = list(asdict(self.results[0]).keys())
                fieldnames.remove('planned_skills')

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in self.results:
                    row = asdict(result)
                    row.pop('planned_skills')
                    writer.writerow(row)

        # JSON
        json_path = self.results_dir / f"rq2_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

        print(f"Results saved: {csv_path}")


async def main_async():
    """Async main function."""
    import argparse

    parser = argparse.ArgumentParser(description='RQ2 Memory Size Experiment')
    parser.add_argument('--commands', default='test_commands_minimal.json',
                        help='Test commands file (default: test_commands_minimal.json - 20 commands)')
    parser.add_argument('--memory-sizes', type=int, nargs='+', default=[0, 10, 20, 35],
                        help='Memory sizes to test (default: 0 10 20 35 per Section 3.7.4)')
    parser.add_argument('--trials', type=int, default=3,
                        help='Number of trials per command (default: 3 per Section 3.7.4)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from previous results file (CSV path)')
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

    experiment = RQ2Experiment(
        test_commands_path=str(test_commands),
        skills_path=str(skills_json),
        prompt_path=str(prompt_txt),
        results_dir=str(results_dir),
        memory_sizes=args.memory_sizes,
        num_trials=args.trials,
        random_seed=42,
        resume_from=args.resume
    )

    try:
        await experiment.run_all_trials()
        print("\n✓ RQ2 Experiment completed!")
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
