#!/usr/bin/env python3
"""
RQ5: Real-execution verification.

Every other experiment (RQ1-RQ4, run_fast.py) scores the LLM's generated plan
as TEXT against a reference plan -- none of them ever send a single command
to ROS2/MoveIt. This script closes that gap: for each should_pass command, it
sends (a) the LLM's actual generated plan and (b) the ground-truth reference
plan to the real, running skill_server via src/llm_agent/executor.py, and
records whether the simulated robot actually completes each step.

Requires the ROS2 stack already running (same launch start.sh uses):
    ros2 launch battery_dismantle_task fake_execution_complete.launch.py
This script restarts that stack before each command (no scene-reset service
exists, so a fresh launch is the only way to guarantee a clean starting
scene between commands).

Usage:
    python3 run_rq5_real_execution.py                 # full 18-command sweep
    python3 run_rq5_real_execution.py --commands "Go to the home position"
"""
from __future__ import annotations
import sys, os, json, time, subprocess, argparse
from pathlib import Path
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE = Path(__file__).parent
DISSERTATION = BASE.parent
SRC = DISSERTATION / "src" / "llm_agent"
RESULTS = BASE / "results_fast"
RESULTS.mkdir(exist_ok=True)

sys.path.insert(0, str(SRC))
from executor import Executor  # noqa: E402

from run_fast import JsonlSink  # noqa: E402 -- reuse the same resume-safe sink
from eval.stats import wilson_ci  # noqa: E402 -- report success rates with real uncertainty, not bare %

LLM_PLANS_SOURCE = RESULTS / "rq1.jsonl"
VALIDATION_PLANS_SOURCE = RESULTS / "rq2.jsonl"  # NV level = raw unvalidated LLM output
REFERENCE_PLANS = BASE / "eval" / "reference_plans.json"

# rviz2 deliberately excluded: an already-running RViz (started once,
# separately, before this script) should survive every restart so it can
# actually be watched, instead of being killed and relaunched before anyone
# can see it -- confirmed via its own log that it was being SIGINT'd every
# 30-80s by this exact restart loop. LAUNCH_CMD passes use_rviz:=false so
# each restart doesn't also spawn a competing second RViz window.
RESTART_KILL_PATTERNS = [
    "ros2 launch", "skill_server_node", "visual_state_manager_node",
    "robot_state_publisher", "move_group", "ros2_control_node",
]
LAUNCH_CMD = ["ros2", "launch", "battery_dismantle_task", "fake_execution_complete.launch.py",
              "use_rviz:=false"]
JOINT_INIT_CMD = [
    "ros2", "topic", "pub", "--once", "/joint_states", "sensor_msgs/msg/JointState",
    "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'base_link'}, "
    "name: ['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6','joint_7',"
    "'robotiq_85_left_knuckle_joint','robotiq_85_right_knuckle_joint'], "
    "position: [0.0,0.2618,3.14159,-2.2689,0.0,0.9599,1.5708,0.0,0.0], velocity: [], effort: []}",
]


def restart_stack(timeout_ready_s: int = 25) -> subprocess.Popen:
    """Kill any stale ROS2 nodes, relaunch the fake-execution stack, wait for
    skill_server readiness, then prime /joint_states. Mirrors start.sh steps 1-3
    exactly (single source of truth for "how to cleanly restart" stays start.sh;
    this just re-runs the same commands so RQ5 gets the same guarantee)."""
    print("  [restart] killing stale ROS2 nodes...")
    for pattern in RESTART_KILL_PATTERNS:
        subprocess.run(["pkill", "-9", "-f", pattern], stderr=subprocess.DEVNULL)
    subprocess.run(["ros2", "daemon", "stop"], stderr=subprocess.DEVNULL)
    time.sleep(2)
    # Clear the fast-dds shared-memory segments the killed nodes left behind.
    # Across the ~50 restarts this run performs they otherwise accumulate and
    # eventually corrupt the DDS transport (service calls time out, IK returns
    # bad solutions), producing execution failures that look like robot-code
    # bugs but are really a degraded environment. Wiping them keeps every
    # restart as clean as the first, so success/failure reflects the code.
    subprocess.run("rm -f /dev/shm/*fast* /dev/shm/sem.*fast* 2>/dev/null",
                   shell=True, stderr=subprocess.DEVNULL)
    time.sleep(1)

    print("  [restart] launching fresh stack...")
    log = open("/tmp/rq5_ros2_launch.log", "w")
    proc = subprocess.Popen(LAUNCH_CMD, stdout=log, stderr=subprocess.STDOUT,
                             cwd=str(DISSERTATION))

    ready = False
    for _ in range(timeout_ready_s):
        if proc.poll() is not None:
            raise RuntimeError(f"ROS2 launch exited early (rc={proc.returncode}); see /tmp/rq5_ros2_launch.log")
        r = subprocess.run(["ros2", "node", "list"], capture_output=True, text=True)
        if "skill_server" in (r.stdout or ""):
            ready = True
            break
        time.sleep(1)
    if not ready:
        raise RuntimeError("skill_server did not become ready within timeout; see /tmp/rq5_ros2_launch.log")
    print("  [restart] skill_server ready")

    subprocess.run(JOINT_INIT_CMD, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    print("  [restart] joint state primed")
    return proc


def load_config_plans(config: str) -> dict[str, list]:
    """command -> planned_skills, from the given RQ1 configuration, trial_id=1.

    config="FS" (Full System: RAG + validation) vs config="LO" (LLM Only, no
    RAG, no validation) -- comparing these two under REAL execution is the
    point: does the validation/RAG layer actually prevent real robot
    failures, or does it only look better on paper (RQ2's text-only recall)?
    """
    # The current rq1.jsonl (produced by run_fast) keys plans by
    # `validation_level` (NV/SV/RV/FV), not the legacy `configuration`=FS/LO
    # field this script was originally written against. Without this mapping
    # load_config_plans matched nothing, so the llm/llm_lo execution sources
    # silently never ran (only `reference` did). Map the two execution configs
    # onto the validation dimension that IS present:
    #   FS (full system, validated)  -> FV (full validation applied)
    #   LO (LLM only, unvalidated)    -> NV (no validation)
    # NOTE: rq1 varies only the validation layer, not RAG, so this reproduces
    # the "validated vs unvalidated" comparison but not the original FS/LO
    # RAG-on/off contrast (RAG is varied in RQ3, not RQ1).
    config_to_level = {"FS": "FV", "LO": "NV"}
    want = config_to_level.get(config, config)
    out = {}
    with open(LLM_PLANS_SOURCE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            level = r.get("validation_level", r.get("configuration"))
            if level == want and r.get("trial_id") == 1:
                out[r["command"]] = r["planned_skills"]
    return out


def load_reference_plans() -> dict[str, list]:
    refs = json.loads(REFERENCE_PLANS.read_text(encoding="utf-8"))["commands"]
    out = {}
    for cmd, info in refs.items():
        if info["safety_label"] != "should_pass":
            continue
        plans = info.get("acceptable_reference_plans", [])
        out[cmd] = plans[0] if plans else []
    return out


def load_should_block_hallucinated_plans() -> dict[str, list]:
    """Defense-in-depth probe: for should_block commands (the LLM should have
    refused), pull the RAW unvalidated plan (RQ2's NV level -- no schema/rule
    validation applied) wherever it's non-empty, i.e. the LLM hallucinated an
    attempt instead of correctly refusing. RQ1/RQ2's text-level validation
    already measures whether the validator WOULD flag these -- and the answer
    is: almost never. Safety recall is 2/80 (2.5%) at the Schema and Full
    levels and 0/80 at No-Validation and Rule (see "RQ2 is reported as two
    views" in README.md; an earlier revision of this comment cited ~33% from
    the superseded metric that conflated "plan is non-canonical" with "command
    is unsafe"). That near-total miss rate is exactly why this probe matters:
    it asks the harder, real-world question -- if
    an unflagged bad plan actually reaches the real skill_server, does
    dispatch-level structural checking (schema/target existence) catch it
    independently, or does it just execute a semantically-wrong-but-
    structurally-valid action on the real robot?
    """
    refs = json.loads(REFERENCE_PLANS.read_text(encoding="utf-8"))["commands"]
    should_block = {c for c, v in refs.items() if v["safety_label"] == "should_block"}
    out = {}
    with open(VALIDATION_PLANS_SOURCE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if (r.get("validation_level") == "NV" and r.get("trial_id") == 1
                    and r.get("command") in should_block and r.get("planned_skills")):
                out[r["command"]] = r["planned_skills"]
    return out


def run_one(cmd: str, plan_source: str, plan: list, sink: JsonlSink) -> dict:
    key = f"rq5|{plan_source}|{cmd}"
    if sink.has(key):
        print(f"  [skip] already done: {plan_source} / {cmd}")
        return None

    print(f"\n=== {plan_source.upper()}: {cmd!r} ===")
    print(f"  plan: {plan}")

    restart_stack()

    executor = Executor(use_ros=True)
    t0 = time.time()
    try:
        result = executor.execute({"plan": plan}, timeout=60.0)
    finally:
        executor.shutdown()
    dt = time.time() - t0

    row = {
        "_key": key,
        "command": cmd,
        "plan_source": plan_source,
        "plan": plan,
        "success": result["success"],
        "executed": result["executed"],
        "failed": result["failed"],
        "total_steps": len(plan),
        "wall_time_s": round(dt, 2),
        "log": result["log"],
        "timestamp": datetime.now().isoformat(),
    }
    sink.write(row)
    print(f"  -> success={row['success']} executed={row['executed']}/{row['total_steps']} time={dt:.1f}s")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commands", nargs="*", default=None,
                    help="Subset of commands to run (default: all commands for the selected sources).")
    ap.add_argument("--sources", nargs="*", default=["llm", "llm_lo", "reference"],
                    choices=["llm", "llm_lo", "reference", "defense"],
                    help="Which plan sources to execute. 'llm' = FS (RAG+validation), "
                         "'llm_lo' = LO (no RAG, no validation) -- comparing the two "
                         "under real execution is the point of the RQ2-vs-reality check. "
                         "'defense' = should_block commands where the unvalidated LLM "
                         "hallucinated a non-empty plan instead of refusing -- probes "
                         "whether the real skill_server catches what text-level "
                         "validation (RQ1+RQ2 factorial) missed. Default runs the "
                         "should_pass comparison (llm/llm_lo/reference); pass "
                         "--sources defense to run the defense-in-depth probe instead.")
    args = ap.parse_args()

    fs_plans = load_config_plans("FS")
    lo_plans = load_config_plans("LO")
    ref_plans = load_reference_plans()
    defense_plans = load_should_block_hallucinated_plans()
    plans_by_source = {"llm": fs_plans, "llm_lo": lo_plans, "reference": ref_plans,
                        "defense": defense_plans}

    default_universe = ref_plans if any(s in args.sources for s in ("llm", "llm_lo", "reference")) else defense_plans
    commands = args.commands if args.commands else sorted(default_universe.keys())

    sink = JsonlSink(RESULTS / "rq5.jsonl")

    for cmd in commands:
        for source in args.sources:
            plans = plans_by_source[source]
            if cmd in plans:
                run_one(cmd, source, plans[cmd], sink)
            else:
                print(f"  [skip] {cmd!r}: no {source} plan found")

    sink.close()

    # summary
    rows = [json.loads(l) for l in open(RESULTS / "rq5.jsonl", encoding="utf-8")]
    for source in ("llm", "llm_lo", "reference"):
        rs = [r for r in rows if r["plan_source"] == source]
        if not rs:
            continue
        n_ok = sum(1 for r in rs if r["success"])
        ci = wilson_ci(n_ok, len(rs))
        print(f"\n{source.upper()}: {n_ok}/{len(rs)} plans executed successfully "
              f"({100*ci['p']:.1f}% [{100*ci['lo']:.1f}, {100*ci['hi']:.1f}])")

    defense_rows = [r for r in rows if r["plan_source"] == "defense"]
    if defense_rows:
        n_caught = sum(1 for r in defense_rows if not r["success"])  # failure here = caught = good
        ci = wilson_ci(n_caught, len(defense_rows))
        print(f"\nDEFENSE-IN-DEPTH (should_block commands, unvalidated hallucinated plan sent to real robot):")
        print(f"  Real execution layer independently caught {n_caught}/{len(defense_rows)} "
              f"({100*ci['p']:.1f}% [{100*ci['lo']:.1f}, {100*ci['hi']:.1f}]) "
              f"-- 'caught' means the plan FAILED to execute (good: dispatch-level check worked).")
        for r in defense_rows:
            tag = "caught (failed to execute)" if not r["success"] else "[WARN] NOT caught (executed anyway)"
            print(f"    {r['command']:45} {tag}")


if __name__ == "__main__":
    main()
