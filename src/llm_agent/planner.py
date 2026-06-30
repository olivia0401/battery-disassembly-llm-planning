"""
Unified planning logic: Prompt + Parse + (Optional) RAG
Experiment-ready version:
- reproducible RAG memory-size control (limit/seed)
- structured retrieval metadata for Results
- safer JSON parsing
"""

import json
import asyncio
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from llm_client import LLMClient

try:
    from rag_engine import RAGEngine
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


class Planner:
    """Task planner"""

    def __init__(
        self,
        config_dir: Optional[Path] = None,
        backend: str = "ollama",
        enable_rag: bool = True,
        rag_topk: int = 3,
        rag_limit: Optional[int] = None,  # for memory-size experiments: 0/10/20/35
        rag_seed: int = 42,
        prompt_version: str = "v1",        # set to git hash or manual version
        temperature: Optional[float] = None,  # passed through to the LLM call
        prompt_file: str = "prompt.txt",   # use "prompt_clean.txt" for leak-free runs
        rag_source: Optional[Path] = None,  # explicit memory file (leak-free split)
    ):
        if config_dir is None:
            config_dir = Path(__file__).parent / "config"

        # Load configs
        with open(config_dir / "skills.json", "r", encoding="utf-8") as f:
            self.skills_config = json.load(f)

        with open(config_dir / prompt_file, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

        self.prompt_version = prompt_version
        self.rag_topk = int(rag_topk)

        # LLM client
        self.llm = LLMClient(backend=backend)
        # Temperature is forwarded to every generate() call (the old code set an
        # attribute that generate() ignored, so temperature was silently fixed).
        self.temperature = 0.3 if temperature is None else float(temperature)

        # RAG engine (optional)
        self.rag = None
        self.rag_limit = rag_limit
        self.rag_seed = rag_seed
        self.rag_source = Path(rag_source) if rag_source else None

        if enable_rag and RAG_AVAILABLE:
            try:
                self.rag = RAGEngine(enable_rag=True)
                # Load an explicit (leak-free) memory file and/or a limited snapshot.
                if self.rag_source is not None or rag_limit is not None:
                    loaded = self.rag.load_from_json(path=self.rag_source,
                                                     limit=rag_limit, seed=rag_seed)
                    print(f"✅ RAG enabled (source={self.rag_source.name if self.rag_source else 'default'}, "
                          f"limit={rag_limit}, seed={rag_seed}): loaded {loaded} cases")
                else:
                    stats = self.rag.get_stats()
                    print(f"✅ RAG enabled: {stats.get('total_cases', 0)} cases in KB")
            except Exception as e:
                print(f"⚠️  RAG init failed: {e}")
                self.rag = None

    async def plan(self, task: str, use_llm: bool = True) -> Dict[str, Any]:
        """
        Generate a plan. Returns a dict with:
          - plan: [...]
          - meta: { planner_mode, rag_used, retrieval_stats, timings, prompt_version ... }
          - rag_context: list of retrieved cases (optional, for UI/inspection)
        """
        t0 = time.time()

        # RAG retrieval (optional)
        rag_context = self._retrieve_similar_cases(task)

        # Fallback if LLM not available
        needs_key = self.llm.config.get("requires_key", True)
        if (not use_llm) or (needs_key and not getattr(self.llm, "api_key", None)):
            plan = self._demo_plan(task)
            return self._attach_meta(
                plan=plan,
                task=task,
                rag_context=rag_context,
                planner_mode="demo",
                t0=t0,
            )

        prompt = self._build_prompt(task, rag_context)

        print(f"🤖 Planning: {task}")
        print("📡 Calling LLM...")

        try:
            response = await self.llm.generate(prompt, temperature=self.temperature)
            plan = self._parse_response(response)

            return self._attach_meta(
                plan=plan,
                task=task,
                rag_context=rag_context,
                planner_mode="llm",
                t0=t0,
                llm_raw=response,
            )

        except Exception as e:
            # str(e) is empty for some exceptions (notably asyncio.TimeoutError),
            # which made every timeout look like a blank, unexplained failure.
            print(f"❌ LLM planning failed: {type(e).__name__}: {e}")
            print("🔄 Falling back to demo plan...")
            plan = self._demo_plan(task)

            return self._attach_meta(
                plan=plan,
                task=task,
                rag_context=rag_context,
                planner_mode="fallback_demo",
                t0=t0,
                error=str(e),
            )

    # -----------------------------
    # RAG
    # -----------------------------
    def _retrieve_similar_cases(self, task: str) -> List[Dict[str, Any]]:
        if not self.rag or not getattr(self.rag, "enabled", False):
            return []
        try:
            return self.rag.retrieve_similar_cases(task, n_results=self.rag_topk)
        except Exception as e:
            print(f"⚠️  RAG retrieval failed: {e}")
            return []

    # -----------------------------
    # Prompt
    # -----------------------------
    def _build_prompt(self, task: str, rag_context: Optional[List[Dict[str, Any]]] = None) -> str:
        skills_list = "\n".join([
            f"- {s['name']}: {s['description']}"
            for s in self.skills_config.get('available_skills', [])
        ])

        poses_list = ", ".join(self.skills_config.get('available_poses', []))
        objects_list = ", ".join(self.skills_config.get('available_objects', []))

        prompt = self.prompt_template.format(
            task_description=task,
            skills_list=skills_list,
            poses_list=poses_list,
            objects_list=objects_list
        )

        # RAG context injection (structured and controlled)
        if rag_context:
            prompt += "\n\nSIMILAR PAST SUCCESSFUL CASES (use as guidance, DO NOT invent new skills/targets):\n"
            for i, case in enumerate(rag_context, 1):
                sim = float(case.get("similarity_score", 0.0))
                prompt += f"\nCase {i} | sim={sim:.2f}\n"
                prompt += f"Task: {case.get('task','')}\n"
                # Ensure plan is compact JSON to reduce token bloat
                prompt += "Plan JSON:\n"
                prompt += json.dumps(case.get("plan", {}), ensure_ascii=False)
                prompt += "\n"

        # Strong constraints (helps reduce hallucinations)
        prompt += (
            "\n\nIMPORTANT CONSTRAINTS:\n"
            "1) Output MUST be a single valid JSON object.\n"
            "2) JSON MUST contain key 'plan' with a list of steps.\n"
            "3) Each step MUST have: step (int), name (skill), params (dict).\n"
            "4) Skill name MUST be one of the available skills.\n"
            "5) params.target MUST be one of the available poses/objects where applicable.\n"
            "6) Do NOT include any extra text outside the JSON.\n"
        )

        return prompt

    # -----------------------------
    # Parsing
    # -----------------------------
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Robust JSON extraction:
        - Strip ```json fences
        - Extract first JSON object by bracket matching
        """
        # Remove code fences
        cleaned = re.sub(r"```json\s*", "", response, flags=re.IGNORECASE)
        cleaned = re.sub(r"```\s*", "", cleaned)

        json_str = self._extract_first_json_object(cleaned)
        if not json_str:
            raise ValueError("No JSON object found in LLM response")

        try:
            plan = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from LLM: {e}")

        if "plan" not in plan or not isinstance(plan["plan"], list):
            raise ValueError("Invalid plan format: missing 'plan' list")

        return plan

    def _extract_first_json_object(self, text: str) -> str:
        """Return the first top-level {...} JSON object substring."""
        start = text.find("{")
        if start == -1:
            return ""

        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
        return ""

    # -----------------------------
    # Demo plan
    # -----------------------------
    def _demo_plan(self, task: str = "") -> Dict[str, Any]:
        task_lower = task.lower()

        if "home" in task_lower and ("go" in task_lower or "move" in task_lower):
            return {"plan": [{"step": 1, "name": "moveTo", "params": {"target": "HOME"}}]}

        if "open" in task_lower and "gripper" in task_lower:
            return {"plan": [{"step": 1, "name": "openGripper", "params": {}}]}

        if "close" in task_lower and "gripper" in task_lower:
            return {"plan": [{"step": 1, "name": "closeGripper", "params": {}}]}

        if "remove" in task_lower and "bolt" in task_lower:
            return {
                "plan": [
                    {"step": 1, "name": "grasp", "params": {"target": "TopCoverBolts"}},
                    {"step": 2, "name": "moveTo", "params": {"target": "place_bolts"}},
                    {"step": 3, "name": "release", "params": {"target": "TopCoverBolts"}},
                ]
            }

        grasp_keywords = ["pick", "grasp", "grab", "take"]
        if any(kw in task_lower for kw in grasp_keywords) and "bolt" in task_lower:
            return {"plan": [{"step": 1, "name": "grasp", "params": {"target": "TopCoverBolts"}}]}

        release_keywords = ["release", "drop", "place", "put"]
        if any(kw in task_lower for kw in release_keywords) and "bolt" in task_lower:
            return {"plan": [{"step": 1, "name": "release", "params": {"target": "TopCoverBolts"}}]}

        disassemble_keywords = ["disassemble", "dismantle", "拆解", "拆卸", "complete", "full"]
        if any(kw in task_lower for kw in disassemble_keywords):
            return {
                "plan": [
                    {"step": 1, "name": "grasp", "params": {"target": "TopCoverBolts"}},
                    {"step": 2, "name": "release", "params": {"target": "TopCoverBolts"}},
                    {"step": 3, "name": "moveTo", "params": {"target": "HOME"}},
                    {"step": 4, "name": "grasp", "params": {"target": "BatteryBox_0"}},
                    {"step": 5, "name": "release", "params": {"target": "BatteryBox_0"}},
                    {"step": 6, "name": "moveTo", "params": {"target": "HOME"}},
                ]
            }

        print(f"⚠️  Unrecognized command '{task}', returning safe default (moveTo HOME)")
        return {"plan": [{"step": 1, "name": "moveTo", "params": {"target": "HOME"}}]}

    # -----------------------------
    # Metadata attachment (for experiments)
    # -----------------------------
    def _attach_meta(
        self,
        plan: Dict[str, Any],
        task: str,
        rag_context: List[Dict[str, Any]],
        planner_mode: str,
        t0: float,
        llm_raw: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        t1 = time.time()

        retrieval_ids = [c.get("id", "") for c in rag_context]
        retrieval_dists = [float(c.get("distance", 0.0)) for c in rag_context]
        retrieval_sims = [float(c.get("similarity_score", 0.0)) for c in rag_context]

        meta = {
            "task": task,
            "planner_mode": planner_mode,   # llm / demo / fallback_demo
            "prompt_version": self.prompt_version,
            "backend": getattr(self.llm, "backend", None) or self.llm.config.get("backend", ""),
            "rag_used": bool(rag_context),
            "rag_topk": self.rag_topk,
            "rag_limit": self.rag_limit,
            "rag_seed": self.rag_seed,
            "retrieval": {
                "ids": retrieval_ids,
                "distances": retrieval_dists,
                "similarities": retrieval_sims,
                "k_returned": len(rag_context),
            },
            "timing": {
                "planning_wall_s": round(t1 - t0, 4),
            },
        }
        if error:
            meta["error"] = error

        # Keep raw LLM response for debugging only (optional, can be huge)
        # You can disable this in experiments to avoid writing large logs.
        if llm_raw is not None:
            meta["llm_raw_len"] = len(llm_raw)

        plan = dict(plan)  # shallow copy
        plan["rag_context"] = rag_context
        plan["meta"] = meta
        return plan

    def print_plan(self, plan: Dict[str, Any]) -> None:
        print("\n✅ Generated Plan:")
        for action in plan.get("plan", []):
            print(f"  Step {action.get('step')}: {action.get('name')}({action.get('params')})")
        meta = plan.get("meta", {})
        if meta:
            print(f"\nℹ️  Meta: mode={meta.get('planner_mode')} rag_used={meta.get('rag_used')} "
                  f"planning_s={meta.get('timing',{}).get('planning_wall_s')}")


# quick test
async def test():
    planner = Planner(enable_rag=True, rag_limit=10, rag_seed=42)
    plan = await planner.plan("remove the bolt and place it in the tray", use_llm=False)
    planner.print_plan(plan)


if __name__ == "__main__":
    asyncio.run(test())
