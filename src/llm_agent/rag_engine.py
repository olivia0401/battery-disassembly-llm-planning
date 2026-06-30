"""
RAG retrieval engine (experiment-ready version)
- Deterministic embeddings (explicit embedding function)
- Safe loading (no duplicate inserts)
- Experiment controls: reset / load limit / seeded sampling
- Throttled disk persistence to avoid latency pollution
"""

from __future__ import annotations

import json
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optional deps
try:
    import chromadb
    from chromadb.api.types import EmbeddingFunction
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    chromadb = None
    SentenceTransformer = None
    EmbeddingFunction = object


@dataclass
class RetrievedCase:
    """
    Data structure representing a retrieved RAG case from the experience database.

    Attributes:
        task (str): Natural language description of the task
        plan (Dict[str, Any]): Planned skill sequence that was executed
        execution_time (float): Time taken to execute the plan (seconds)
        distance (float): Vector distance from query embedding
        similarity (float): Semantic similarity score (1 - distance)
        case_id (str): Unique identifier for the case
    """
    task: str
    plan: Dict[str, Any]
    execution_time: float
    distance: float
    similarity: float
    case_id: str


class _STEmbeddingFn(EmbeddingFunction):
    """Chroma-compatible embedding function using SentenceTransformer."""
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def __call__(self, texts: List[str]) -> List[List[float]]:
        # Normalize embeddings so cosine distance is well-behaved
        embs = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embs.tolist()


class RAGEngine:
    """
    Retrieval-Augmented Generation (RAG) engine for robot task planning.

    Stores and retrieves successful task execution cases to improve planning
    quality through experience. Uses ChromaDB for vector storage and
    SentenceTransformer for semantic embeddings.

    Features:
        - Deterministic embeddings for reproducible experiments
        - Automatic disk persistence with throttling
        - Experiment controls (reset, load limits, seeded sampling)
        - Safe loading to avoid duplicate entries

    Args:
        data_dir: Directory for storing ChromaDB and JSON backups
        enable_rag: Enable/disable RAG functionality
        collection_name: Name of the ChromaDB collection
        embedding_model_name: HuggingFace model name for embeddings
        distance_space: Distance metric (cosine/l2/ip)
        autosave_every_n_adds: Save to disk every N new cases
        autosave_min_interval_s: Minimum seconds between autosaves
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        enable_rag: bool = True,
        collection_name: str = "experience_cases",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        distance_space: str = "cosine",  # for documentation / expectations
        autosave_every_n_adds: int = 10,
        autosave_min_interval_s: float = 20.0,
    ):
        self.enabled = bool(enable_rag and RAG_AVAILABLE)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.distance_space = distance_space

        if data_dir is None:
            data_dir = Path(__file__).parent / "rag_data"
        data_dir.mkdir(exist_ok=True)
        self.data_dir = data_dir

        # autosave controls
        self._autosave_every_n_adds = max(1, int(autosave_every_n_adds))
        self._autosave_min_interval_s = float(autosave_min_interval_s)
        self._adds_since_save = 0
        self._last_save_ts = 0.0

        if not self.enabled:
            if not RAG_AVAILABLE:
                print("ℹ️  RAG disabled: missing dependencies (chromadb, sentence-transformers)")
            else:
                print("ℹ️  RAG disabled by configuration")
            return

        try:
            print("🔧 Initializing ChromaDB...")
            self.client = chromadb.PersistentClient(path=str(self.data_dir / "chromadb"))

            print(f"🤖 Loading embedding model ({self.embedding_model_name})...")
            self.embedder = SentenceTransformer(self.embedding_model_name)
            self.embedding_fn = _STEmbeddingFn(self.embedder)

            # Create collection with explicit embedding function
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={
                    "description": "Successful execution cases",
                    "embedding_model": self.embedding_model_name,
                    "distance_space_expected": self.distance_space,
                },
            )

            # Load JSON cases only if collection is empty (avoid duplicates)
            self._load_initial_cases_if_needed()

            print("✅ RAG engine ready")

        except Exception as e:
            print(f"⚠️  RAG init failed: {e}")
            print("   Running without RAG support")
            self.enabled = False

    # -----------------------------
    # Loading / persistence
    # -----------------------------
    def _experience_file(self) -> Path:
        return self.data_dir / "experience_cases.json"

    def _load_initial_cases_if_needed(self) -> None:
        """Load cases from disk only if the collection is empty."""
        if self.collection.count() > 0:
            # Already has data; don't re-import JSON (prevents duplicates).
            return

        experience_file = self._experience_file()
        if not experience_file.exists():
            print("ℹ️  No existing experience cases found")
            return

        try:
            with open(experience_file, "r", encoding="utf-8") as f:
                cases = json.load(f)

            if not cases:
                return

            # Use stable IDs if provided; otherwise derive from index
            ids, docs, metas = [], [], []
            for i, case in enumerate(cases):
                cid = str(case.get("id") or f"case_{i}")
                ids.append(cid)
                docs.append(case["task"])
                metas.append({
                    "plan": json.dumps(case["plan"], ensure_ascii=False),
                    "execution_time": float(case.get("execution_time", 0.0)),
                    "timestamp": float(case.get("timestamp", time.time())),
                    # extensible metadata slots:
                    "model_backend": case.get("model_backend", ""),
                    "prompt_version": case.get("prompt_version", ""),
                    "validator": case.get("validator", ""),
                    "result": case.get("result", "success"),
                })

            self.collection.add(documents=docs, metadatas=metas, ids=ids)
            print(f"✅ Imported {len(cases)} cases from {experience_file.name}")

        except Exception as e:
            print(f"⚠️  Failed to import experience cases: {e}")

    def export_to_json(self, path: Optional[Path] = None) -> Path:
        """Export all cases to JSON (full snapshot)."""
        if not self.enabled:
            raise RuntimeError("RAG engine is disabled")

        if path is None:
            path = self._experience_file()

        all_cases = self.collection.get(include=["documents", "metadatas"])
        docs = all_cases.get("documents") or []
        ids = all_cases.get("ids") or []
        metas = all_cases.get("metadatas") or []

        cases_data = []
        for i in range(len(docs)):
            meta = metas[i] or {}
            cases_data.append({
                "id": ids[i],
                "task": docs[i],
                "plan": json.loads(meta.get("plan", "{}")),
                "execution_time": float(meta.get("execution_time", 0.0)),
                "timestamp": float(meta.get("timestamp", 0.0)),
                # optional fields
                "model_backend": meta.get("model_backend", ""),
                "prompt_version": meta.get("prompt_version", ""),
                "validator": meta.get("validator", ""),
                "result": meta.get("result", "success"),
            })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(cases_data, f, indent=2, ensure_ascii=False)

        return path

    def _maybe_autosave(self) -> None:
        """Throttle disk exports to avoid affecting latency metrics."""
        now = time.time()
        if self._adds_since_save < self._autosave_every_n_adds:
            return
        if (now - self._last_save_ts) < self._autosave_min_interval_s:
            return

        try:
            self.export_to_json()
            self._adds_since_save = 0
            self._last_save_ts = now
        except Exception as e:
            print(f"⚠️  Autosave failed: {e}")

    # -----------------------------
    # Experiment helpers
    # -----------------------------
    def reset(self) -> None:
        """Delete and recreate the collection (EXPERIMENT ONLY)."""
        if not self.enabled:
            return

        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={
                "description": "Successful execution cases",
                "embedding_model": self.embedding_model_name,
                "distance_space_expected": self.distance_space,
            },
        )

    def load_from_json(self, path: Optional[Path] = None, limit: Optional[int] = None, seed: int = 0) -> int:
        """
        Load cases from a JSON snapshot, with optional limit and deterministic sampling.
        Designed for memory-size experiments (0/10/20/35).
        """
        if not self.enabled:
            return 0

        if path is None:
            path = self._experience_file()
        if not path.exists():
            return 0

        with open(path, "r", encoding="utf-8") as f:
            cases = json.load(f) or []

        if limit is not None:
            rnd = random.Random(seed)
            if limit <= 0:
                cases = []
            elif len(cases) > limit:
                cases = rnd.sample(cases, k=limit)

        # Clear current collection and load fresh to avoid contamination
        self.reset()
        if not cases:
            return 0

        ids, docs, metas = [], [], []
        for i, case in enumerate(cases):
            cid = str(case.get("id") or f"case_{i}")
            ids.append(cid)
            docs.append(case["task"])
            metas.append({
                "plan": json.dumps(case["plan"], ensure_ascii=False),
                "execution_time": float(case.get("execution_time", 0.0)),
                "timestamp": float(case.get("timestamp", time.time())),
                "model_backend": case.get("model_backend", ""),
                "prompt_version": case.get("prompt_version", ""),
                "validator": case.get("validator", ""),
                "result": case.get("result", "success"),
            })

        self.collection.add(documents=docs, metadatas=metas, ids=ids)
        return len(cases)

    # -----------------------------
    # Core API
    # -----------------------------
    def retrieve_similar_cases(self, task: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Retrieve similar successful cases for the given task."""
        if not self.enabled:
            return []
        if self.collection.count() == 0:
            return []

        try:
            results = self.collection.query(
                query_texts=[task],
                n_results=min(int(n_results), self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            docs = results.get("documents", [[]])[0] or []
            metas = results.get("metadatas", [[]])[0] or []
            dists = results.get("distances", [[]])[0] or []
            ids = results.get("ids", [[]])[0] if "ids" in results else [""] * len(docs)

            out: List[Dict[str, Any]] = []
            for i in range(len(docs)):
                dist = float(dists[i]) if i < len(dists) else 0.0

                # For normalized embeddings with cosine distance,
                # distance is typically in [0, 2]. Convert to a bounded similarity:
                sim = max(0.0, min(1.0, 1.0 - (dist / 2.0)))

                meta = metas[i] or {}
                out.append({
                    "id": ids[i] if i < len(ids) else "",
                    "task": docs[i],
                    "plan": json.loads(meta.get("plan", "{}")),
                    "execution_time": float(meta.get("execution_time", 0.0)),
                    "distance": dist,
                    "similarity_score": sim,
                })

            return out

        except Exception as e:
            print(f"⚠️  RAG retrieval failed: {e}")
            return []

    def add_successful_case(
        self,
        task: str,
        plan: Dict[str, Any],
        execution_time: float = 0.0,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a successful case to memory.

        extra_meta: you can attach experiment info:
          - model_backend, prompt_version, validator, result, etc.
        """
        if not self.enabled:
            return ""

        case_id = f"case_{int(time.time() * 1000)}"
        meta = {
            "plan": json.dumps(plan, ensure_ascii=False),
            "execution_time": float(execution_time),
            "timestamp": time.time(),
        }
        if extra_meta:
            # only store JSON-serializable primitives
            for k, v in extra_meta.items():
                try:
                    json.dumps(v)
                    meta[k] = v
                except TypeError:
                    meta[k] = str(v)

        try:
            self.collection.add(documents=[task], metadatas=[meta], ids=[case_id])
            self._adds_since_save += 1
            self._maybe_autosave()
            return case_id
        except Exception as e:
            print(f"⚠️  Failed to save case: {e}")
            return ""

    def get_stats(self) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "total_cases": 0,
                "reason": "Dependencies not installed or disabled",
                "data_dir": str(self.data_dir) if hasattr(self, "data_dir") else "N/A",
            }
        return {
            "enabled": True,
            "total_cases": int(self.collection.count()),
            "data_dir": str(self.data_dir),
            "collection": self.collection_name,
            "embedding_model": self.embedding_model_name,
            "distance_space": self.distance_space,
        }

    def format_retrieval_log(self, similar_cases: List[Dict[str, Any]]) -> List[str]:
        log: List[str] = []
        if not similar_cases:
            log.append("  └─ No similar cases found in knowledge base")
            return log

        log.append(f"  ├─ Found {len(similar_cases)} similar case(s) in knowledge base:")
        for i, case in enumerate(similar_cases, 1):
            score = float(case.get("similarity_score", 0.0))
            dist = float(case.get("distance", 0.0))
            task = case.get("task", "")
            task_preview = task[:40] + "..." if len(task) > 40 else task
            plan = case.get("plan", {}) or {}
            num_steps = len(plan.get("plan", []))
            exec_time = float(case.get("execution_time", 0.0))

            log.append(f"  │  [{i}] sim={score:.2f} (dist={dist:.3f}) | Task: \"{task_preview}\"")
            log.append(f"  │      Steps: {num_steps} | Exec time: {exec_time:.1f}s")

        return log
