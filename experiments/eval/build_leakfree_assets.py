"""
Build the leak-free experiment assets:

1. rag_data/experience_cases_clean.json
   = memory cases whose task is NOT verbatim-equal to any test command.
   (Removes the open-book leakage where memory == test set, while keeping
    realistic 'similar paraphrase' neighbours.)

2. config/prompt_clean.txt
   = a zero/one-shot prompt with NO test-command answers as few-shot examples,
   and a real {skills_list} placeholder (the old prompt hard-coded only 5 skills).

3. eval/memory_split.json
   = documentation of which commands are in memory vs held out for testing.
"""
import json
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent.parent / "src" / "llm_agent"
EXP = Path(__file__).resolve().parent.parent


def norm(s): return s.strip().lower()


def build_clean_memory():
    mem = json.loads((SRC / "rag_data" / "experience_cases.json").read_text(encoding="utf-8"))
    suite = json.loads((EXP / "unified_test_suite.json").read_text(encoding="utf-8"))
    tests = {norm(c) for cmds in suite["commands"].values() for c in cmds}
    clean = [c for c in mem if norm(c["task"]) not in tests]
    out = SRC / "rag_data" / "experience_cases_clean.json"
    out.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
    split = {
        "_meta": "Leak-free split: memory cases share no verbatim command with the test set.",
        "memory_commands": sorted(c["task"] for c in clean),
        "test_commands": sorted(c for cmds in suite["commands"].values() for c in cmds),
        "n_memory": len(clean), "n_test": len(tests),
    }
    (Path(__file__).parent / "memory_split.json").write_text(
        json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"clean memory: {len(clean)} cases -> {out.name}")
    return len(clean)


PROMPT_CLEAN = """You are a robot task planner for battery disassembly. Convert the command into a JSON
action plan, using ONLY the skills, positions, and objects listed below.

TASK: {task_description}

AVAILABLE SKILLS:
{skills_list}

POSITIONS (the only valid moveTo targets): {poses_list}
OBJECTS (the only valid grasp/release/inspect/unscrew/disconnect targets): {objects_list}

PARAMETER RULES (per skill type):
- moveTo                                   -> params {{"target": <POSITION>}}
- grasp / release / inspect / unscrew / disconnect -> params {{"target": <OBJECT>}}
- openGripper / closeGripper / waitForStabilization -> params {{}}   (no target)
- rotateGripper                            -> params {{"angle": <degrees>}}

OUTPUT RULES:
1. Use ONLY skill names from AVAILABLE SKILLS, and ONLY target values from the
   POSITIONS / OBJECTS lists above. Do not invent skills, positions, or objects.
2. A task may require several ordered steps. Emit them in execution order.
3. If the task needs a capability, position, or object that is NOT listed above
   (e.g. an action this robot cannot perform), return {{"plan": []}} — refuse
   rather than invent.
4. Return ONLY the JSON object — no prose, no markdown.

FORMAT EXAMPLES (illustrative only — these are NOT answers to the task):

Single step ("Move to the safe zone"):
{{"plan": [{{"step": 1, "name": "moveTo", "params": {{"target": "safe_zone"}}}}]}}

Multi-step pick-and-place ("Put the cooling plate in sorting area A"):
{{"plan": [
  {{"step": 1, "name": "grasp", "params": {{"target": "CoolingPlate"}}}},
  {{"step": 2, "name": "moveTo", "params": {{"target": "sorting_area_A"}}}},
  {{"step": 3, "name": "release", "params": {{"target": "CoolingPlate"}}}}
]}}

Refusal ("Paint the casing" — no such capability):
{{"plan": []}}

Now generate the plan for: {task_description}
"""


def build_clean_prompt():
    out = SRC / "config" / "prompt_clean.txt"
    out.write_text(PROMPT_CLEAN, encoding="utf-8")
    print(f"clean prompt -> {out.name} (zero gold answers, real skills list)")


if __name__ == "__main__":
    build_clean_memory()
    build_clean_prompt()
