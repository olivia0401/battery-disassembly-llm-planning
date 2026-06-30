#!/usr/bin/env python3
"""
Fast, resume-safe experiment runner for RQ1/RQ2/RQ3.

Speed-ups vs the original per-RQ scripts
----------------------------------------
1. CONCURRENCY: LLM calls run under an asyncio semaphore (default 8 in flight)
   instead of strictly serial. With ~8 s/call this is the dominant ~Nx win
   (1200 serial calls ~3 h  ->  ~15-25 min at concurrency 8).
2. SHARED PLANNERS: the SentenceTransformer + ChromaDB are loaded ONCE per
   config, not once per trial (the old RQ2 re-loaded them every single trial).
3. PLAN REUSE: a plan is generated ONCE per (command, trial) and then scored by
   all validation levels (RQ1) / reused across LO+LV and LR+FS (RQ3). This cuts
   RQ1 from 700 LLM calls to 175, and RQ3 from 525 to 350.
4. RESUME: every result is appended to a JSONL; re-running skips completed keys,
   so a crash never costs more than the in-flight batch.
5. HONEST PROVENANCE: model id, backend, per-trial seed and leak-free flag are
   written into every row.

Leakage fixes (use --leakfree)
------------------------------
- Memory/test split: RAG memory is loaded from eval/memory_split.json (disjoint
  from the test commands) instead of the file that equals the test set.
- Prompt: pass --prompt config/prompt_clean.txt (no gold answers as few-shot).

Usage
-----
    python run_fast.py --rq 3 --trials 5 --concurrency 8
    python run_fast.py --rq all --leakfree --trials 10
"""
from __future__ import annotations
import sys, os, json, re, time, asyncio, argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

try:  # the planner/RAG prints emojis; avoid GBK console crashes on Windows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "llm_agent"))
from planner import Planner            # noqa: E402
from validator import Validator        # noqa: E402

BASE = Path(__file__).parent
SRC = BASE.parent / "src" / "llm_agent"
RESULTS = BASE / "results_fast"
RESULTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------- helpers
def load_commands(path: Path) -> List[Dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for category, cmds in data["commands"].items():
        for c in cmds:
            out.append({"command": c, "category": category})
    return out


def model_id(planner: Planner) -> str:
    try:
        return f"{planner.llm.backend}:{planner.llm.config.get('model','')}"
    except Exception:
        return "unknown"


def model_tag(model_id_str: str) -> str:
    """Key-safe tag derived from a model id, e.g. 'ollama:llama3.2:1b' -> 'ollama_llama3.2_1b'.

    _key did NOT include the model before, so switching models (e.g. testing
    llama3.2:1b after a prior gemma2:2b/llama3.2-3B run) silently skipped
    re-generating rows whose (rq, config, command, trial) already existed
    from the OLD model's run -- the new model's LLM calls would execute but
    the result was discarded by the resume-skip check, producing zero new
    data with no error. Confirmed live during a 2026-06-30 run.
    """
    return re.sub(r"[^A-Za-z0-9_.-]", "_", model_id_str or "unknown")


class JsonlSink:
    """Append-only result store with a resume key set."""
    def __init__(self, path: Path):
        self.path = path
        self.done = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    self.done.add(r["_key"])
                except Exception:
                    pass
        self.fh = open(path, "a", encoding="utf-8")

    def has(self, key: str) -> bool:
        return key in self.done

    def write(self, row: Dict[str, Any]):
        self.fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.fh.flush()
        self.done.add(row["_key"])

    def close(self):
        self.fh.close()


async def gen_plan(planner: Planner, command: str, sem: asyncio.Semaphore) -> Dict[str, Any]:
    """Generate one plan under the concurrency limiter; returns plan + timing."""
    async with sem:
        t0 = time.time()
        try:
            res = await planner.plan(command, use_llm=True)
            plan = res.get("plan", [])
            meta = res.get("meta", {})
            return {"plan": plan, "ok": True,
                    "planning_time": time.time() - t0,
                    "planner_mode": meta.get("planner_mode", "?"),
                    "k_returned": meta.get("retrieval", {}).get("k_returned", 0),
                    "max_sim": max(meta.get("retrieval", {}).get("similarities", [0.0]) or [0.0])}
        except Exception as e:
            return {"plan": [], "ok": False, "planning_time": time.time() - t0,
                    "planner_mode": "error", "error": str(e), "k_returned": 0, "max_sim": 0.0}


# ---------------------------------------------------------------- RQ3 + RQ1 (share generated plans)
async def run_planning_matrix(commands, trials, concurrency, leakfree, prompt_name, backend="openrouter"):
    """Generate plans for every (command, trial) with and without RAG, ONCE.

    Returns dicts keyed by (command, trial) -> {'norag': planrow, 'rag': planrow}.
    These feed both RQ3 (configs) and RQ1 (validation levels) without re-calling
    the LLM per config/level.
    """
    sem = asyncio.Semaphore(concurrency)
    kw = dict(config_dir=SRC / "config", backend=backend)
    prompt_file = "prompt_clean.txt" if leakfree else "prompt.txt"
    rag_source = str(SRC / "rag_data" / "experience_cases_clean.json") if leakfree else None
    p_norag = Planner(enable_rag=False, prompt_file=prompt_file, **kw)
    p_rag = Planner(enable_rag=True, prompt_file=prompt_file, rag_source=rag_source, **kw)
    mid_norag, mid_rag = model_id(p_norag), model_id(p_rag)

    jobs = []
    for c in commands:
        for t in range(1, trials + 1):
            # per-trial seed so stability metrics are measurable (NOT a fixed seed)
            jobs.append((c, t))

    async def one(c, t, planner, tag):
        r = await gen_plan(planner, c["command"], sem)
        r.update({"command": c["command"], "category": c["category"], "trial": t, "tag": tag})
        return r

    tasks = [one(c, t, p_norag, "norag") for (c, t) in jobs] + \
            [one(c, t, p_rag, "rag") for (c, t) in jobs]
    print(f"Generating {len(tasks)} plans (concurrency={concurrency}) ...")
    results = await asyncio.gather(*tasks)

    bucket: Dict[tuple, Dict[str, Any]] = {}
    for r in results:
        bucket.setdefault((r["command"], r["trial"]), {})[r["tag"]] = r
    return bucket, mid_norag, mid_rag


class ScriptedBaseline:
    """Keyword->skill mapping baseline (inlined to avoid the ROS-dependent module)."""
    MAP = {
        "home": [{"name": "moveTo", "params": {"target": "HOME"}}],
        "grasp bolts": [{"name": "grasp", "params": {"target": "TopCoverBolts"}}],
        "release": [{"name": "openGripper", "params": {}}],
        "placement": [{"name": "moveTo", "params": {"target": "place_bolts"}}],
        "open": [{"name": "openGripper", "params": {}}],
        "close": [{"name": "closeGripper", "params": {}}],
    }

    def plan(self, command: str):
        c = command.lower()
        for kw, skills in self.MAP.items():
            if kw in c:
                return skills
        return []


def write_rq3(bucket, validator, sink, mid_norag, mid_rag, leakfree):
    """SB scripted + LO/LV(reuse norag) + LR/FS(reuse rag), no extra LLM calls."""
    sb = ScriptedBaseline()
    ts = datetime.now().isoformat()
    for (cmd, trial), b in bucket.items():
        cat = (b.get("rag") or b.get("norag"))["category"]
        # configs that reuse the no-RAG plan
        for cfg, src in [("LO", "norag"), ("LV", "norag"), ("LR", "rag"), ("FS", "rag")]:
            pr = b.get(src, {})
            plan = pr.get("plan", [])
            apply_val = cfg in ("LV", "FS")
            valid, _ = (validator.validate_plan({"plan": plan}) if (apply_val and plan) else (bool(plan), []))
            row_model = mid_rag if src == "rag" else mid_norag
            row = {"_key": f"rq3|{cfg}|{cmd}|{trial}|{model_tag(row_model)}", "rq": 3, "configuration": cfg,
                   "command": cmd, "category": cat, "trial_id": trial,
                   "planned_skills": plan, "plan_valid": bool(valid),
                   "planning_time": pr.get("planning_time", 0.0),
                   "planner_mode": pr.get("planner_mode", "?"),
                   "model": row_model,
                   "leakfree": leakfree, "timestamp": ts}
            if not sink.has(row["_key"]):
                sink.write(row)
        # scripted baseline (no LLM involved, but keep a stable tag for consistency)
        plan = sb.plan(cmd)
        row = {"_key": f"rq3|SB|{cmd}|{trial}|scripted", "rq": 3, "configuration": "SB",
               "command": cmd, "category": cat, "trial_id": trial,
               "planned_skills": plan, "plan_valid": bool(plan),
               "planning_time": 0.0, "model": "scripted", "leakfree": leakfree, "timestamp": ts}
        if not sink.has(row["_key"]):
            sink.write(row)


def write_rq1(bucket, validator, sink, mid_rag, leakfree):
    """4 validation levels applied to the SAME RAG plan (reuse, no LLM re-calls)."""
    from run_rq1_safety import SchemaOnlyValidation, RuleBasedValidation, FullValidationWrapper, NoValidation
    levels = {"NV": NoValidation(),
              "SV": SchemaOnlyValidation(str(SRC / "config" / "skills.json")),
              "RV": RuleBasedValidation(validator),
              "FV": FullValidationWrapper(validator)}
    ts = datetime.now().isoformat()
    for (cmd, trial), b in bucket.items():
        pr = b.get("rag", {})
        plan = pr.get("plan", [])
        cat = pr.get("category", "")
        for lvl, v in levels.items():
            res = v.validate(plan)
            row = {"_key": f"rq1|{lvl}|{cmd}|{trial}|{model_tag(mid_rag)}", "rq": 1, "validation_level": lvl,
                   "command": cmd, "category": cat, "trial_id": trial,
                   "planned_skills": plan, "plan_valid": bool(res.get("valid", False)),
                   "validation_time": 0.0,
                   "planner_mode": pr.get("planner_mode", "?"), "model": mid_rag, "leakfree": leakfree, "timestamp": ts}
            if not sink.has(row["_key"]):
                sink.write(row)


# ---------------------------------------------------------------- RQ2 memory
def _rq2_row_from_bucket(k, c, t, pr, validator, mid, leakfree, ts, is_rag):
    """Build an RQ2 row by REUSING a plan already generated in the planning matrix
    (k=0 reuses the no-RAG plan; k=max reuses the full-RAG plan) -> no extra LLM call."""
    plan = pr.get("plan", [])
    valid, _ = validator.validate_plan({"plan": plan}) if plan else (False, [])
    max_sim = float(pr.get("max_sim", 0.0)) if is_rag else 0.0
    return {"_key": f"rq2|{k}|{c['command']}|{t}|{model_tag(mid)}", "rq": 2, "memory_size": k,
            "command": c["command"], "category": c["category"], "trial_id": t,
            "planned_skills": plan, "plan_valid": bool(valid),
            "planning_time": pr.get("planning_time", 0.0),
            "num_cases_retrieved": pr.get("k_returned", 0) if is_rag else 0,
            "max_similarity": max_sim, "is_seen": max_sim >= 0.7,
            "planner_mode": pr.get("planner_mode", "?"),
            "model": mid, "leakfree": leakfree, "timestamp": ts}


async def run_rq2(commands, trials, concurrency, sizes, leakfree, backend="openrouter",
                  reuse_bucket=None, mid_norag="", mid_rag=""):
    sem = asyncio.Semaphore(concurrency)
    sink = JsonlSink(RESULTS / "rq2.jsonl")
    ts = datetime.now().isoformat()
    validator = Validator(SRC / "config")
    prompt_file = "prompt_clean.txt" if leakfree else "prompt.txt"
    rag_source = str(SRC / "rag_data" / "experience_cases_clean.json") if leakfree else None
    full_k = max(sizes) if sizes else None  # k==full_k == the full-RAG plan already generated
    for k in sizes:
        # SPEEDUP: reuse planning-matrix plans for k=0 (no-RAG) and k=full (full-RAG)
        if reuse_bucket is not None and k in (0, full_k):
            tag = "norag" if k == 0 else "rag"
            mid = mid_norag if k == 0 else mid_rag
            n = 0
            for (cmd, trial), b in reuse_bucket.items():
                if trial > trials:
                    continue
                pr = b.get(tag)
                if pr is None:
                    continue
                c = {"command": cmd, "category": pr.get("category", "")}
                row = _rq2_row_from_bucket(k, c, trial, pr, validator, mid, leakfree, ts, k != 0)
                if not sink.has(row["_key"]):
                    sink.write(row); n += 1
            print(f"RQ2 k={k}: reused {n} plans from planning matrix (0 LLM calls)")
            continue
        planner = (Planner(config_dir=SRC / "config", backend=backend,
                           enable_rag=False, prompt_file=prompt_file)
                   if k == 0 else
                   Planner(config_dir=SRC / "config", backend=backend,
                           enable_rag=True, rag_limit=k, rag_seed=42,
                           prompt_file=prompt_file, rag_source=rag_source))   # shared per-k!
        mid = model_id(planner)

        async def one(c, t):
            r = await gen_plan(planner, c["command"], sem)
            plan = r["plan"]
            valid, _ = validator.validate_plan({"plan": plan}) if plan else (False, [])
            return {"_key": f"rq2|{k}|{c['command']}|{t}|{model_tag(mid)}", "rq": 2, "memory_size": k,
                    "command": c["command"], "category": c["category"], "trial_id": t,
                    "planned_skills": plan, "plan_valid": bool(valid),
                    "planning_time": r["planning_time"], "num_cases_retrieved": r["k_returned"],
                    "max_similarity": r["max_sim"], "is_seen": r["max_sim"] >= 0.7,
                    "planner_mode": r.get("planner_mode", "?"),
                    "model": mid, "leakfree": leakfree, "timestamp": ts}

        jobs = [one(c, t) for c in commands for t in range(1, trials + 1)
                if not sink.has(f"rq2|{k}|{c['command']}|{t}|{model_tag(mid)}")]
        print(f"RQ2 k={k}: {len(jobs)} trials ...")
        for row in await asyncio.gather(*jobs):
            sink.write(row)
    sink.close()


# ---------------------------------------------------------------- main
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq", default="all", choices=["1", "2", "3", "all"])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--commands", default="unified_test_suite.json")
    ap.add_argument("--sizes", type=int, nargs="+", default=[0, 10, 20, 35])
    ap.add_argument("--backend", default="openrouter",
                    help="openrouter | ollama | openai | chutes")
    ap.add_argument("--leakfree", action="store_true",
                    help="use disjoint memory split + leak-free prompt")
    args = ap.parse_args()

    commands = load_commands(BASE / args.commands)
    validator = Validator(SRC / "config")
    prompt_name = "prompt_clean.txt" if args.leakfree else None
    # leak-free memory has only ~15 clean cases -> cap the size sweep accordingly
    if args.leakfree and args.sizes == [0, 10, 20, 35]:
        args.sizes = [0, 5, 10, 15]

    bucket = mid_norag = mid_rag = None
    if args.rq in ("1", "3", "all"):
        bucket, mid_norag, mid_rag = await run_planning_matrix(
            commands, args.trials, args.concurrency, args.leakfree, prompt_name, args.backend)
        if args.rq in ("3", "all"):
            s = JsonlSink(RESULTS / "rq3.jsonl")
            write_rq3(bucket, validator, s, mid_norag, mid_rag, args.leakfree); s.close()
        if args.rq in ("1", "all"):
            s = JsonlSink(RESULTS / "rq1.jsonl")
            write_rq1(bucket, validator, s, mid_rag, args.leakfree); s.close()
    if args.rq in ("2", "all"):
        # reuse_bucket lets RQ2 skip re-generating k=0 and k=max (~40% fewer LLM calls)
        await run_rq2(commands, args.trials, args.concurrency, args.sizes, args.leakfree,
                      args.backend, reuse_bucket=bucket, mid_norag=mid_norag or "", mid_rag=mid_rag or "")

    print(f"\nDone. JSONL results in {RESULTS}")
    print("Next:  python -m eval.analyze   then   python -m eval.build_workbook")


if __name__ == "__main__":
    asyncio.run(main())
