"""Controlled and learned memory backends for the pinned CROSSEP-EMB evaluator.

All modes implement the official BaseMemory interface. The frozen worker receives
only memories derived from episodes that have already completed. The evolutionary
method is deliberately implemented by the outer validation-only search in
scripts/run_experiment.py, not by peeking at test rewards inside this adapter.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .base import BaseMemory, MemoryCallStats
except ImportError:
    class BaseMemory:
        def __init__(self, memory_type: str, read_only: bool = False):
            self.memory_type = memory_type
            self.read_only = read_only

    class MemoryCallStats:
        def __init__(self, input_tokens=0, output_tokens=0, embedding_tokens=0,
                     cached_tokens=0, latency=0.0):
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens
            self.embedding_tokens = embedding_tokens
            self.cached_tokens = cached_tokens
            self.latency = latency

        def __add__(self, other):
            return MemoryCallStats(
                self.input_tokens + other.input_tokens,
                self.output_tokens + other.output_tokens,
                self.embedding_tokens + other.embedding_tokens,
                self.cached_tokens + other.cached_tokens,
                self.latency + other.latency,
            )

try:
    from .bm25_adapter import _chunk_text, _format_trajectory, _tokenize
except ImportError:
    def _tokenize(text: str) -> list[str]:
        return text.split()

    def _chunk_text(text: str, chunk_size: int = 1024) -> list[str]:
        return [text]

    def _format_trajectory(conversation: list[dict]) -> str:
        parts = []
        for message in conversation[2:]:
            content = message.get("content") or ""
            if content and "--- Retrieved Memories ---" not in content:
                parts.append(f"{message.get('role', '?')}: {content}")
        return "\n".join(parts)


CATEGORIES = (
    "pick_and_place_simple",
    "pick_two_obj_and_place",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "look_at_obj_in_light",
)

PLAYBOOK = {
    "pick_and_place_simple":
        "Identify the requested object and destination. Locate and take the object, "
        "move to the destination receptacle, place it, and verify completion.",
    "pick_two_obj_and_place":
        "Identify both requested objects and the destination. Locate and transfer "
        "each object separately, then verify that both are placed.",
    "pick_clean_then_place_in_recep":
        "Take the target object, clean it using the appropriate sink or faucet, "
        "retrieve it, then place it in the requested receptacle.",
    "pick_cool_then_place_in_recep":
        "Take the target object, cool it in the refrigerator until the environment "
        "confirms the state, retrieve it, then place it at the destination.",
    "pick_heat_then_place_in_recep":
        "Take the target object, heat it using the appropriate appliance, retrieve "
        "it, then place it in the requested receptacle.",
    "look_at_obj_in_light":
        "Find the target object and an active light source, position the object "
        "appropriately, examine it, and verify the requested result.",
}

MODES = {
    "playbook",
    "random",
    "recent",
    "full_history",
    "bm25",
    "success_bm25",
    "outcome_bm25",
    "bandit",
    "contextual_bandit",
    "supervised_writer",
    "reflexion",
    "expel",
    "letta_consolidation",
    "bayesian_state",
}


def category_from_observation(observation: str) -> str:
    text = observation.lower()
    rules = (
        (r"\bclean\b", "pick_clean_then_place_in_recep"),
        (r"\bcool\b", "pick_cool_then_place_in_recep"),
        (r"\bheat(?:ed|ing)?\b", "pick_heat_then_place_in_recep"),
        (r"\b(?:two|both)\b", "pick_two_obj_and_place"),
        (r"\b(?:look at|examine|light source|under the light)\b", "look_at_obj_in_light"),
    )
    for pattern, category in rules:
        if re.search(pattern, text):
            return category
    return "pick_and_place_simple"


def count_tokens(text: str) -> int:
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, math.ceil(len(text) / 4))


def capped(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return text
    try:
        import tiktoken
        encoder = tiktoken.get_encoding("cl100k_base")
        tokens = encoder.encode(text)
        return text if len(tokens) <= max_tokens else encoder.decode(tokens[:max_tokens])
    except Exception:
        return text if len(text) <= max_tokens * 4 else text[:max_tokens * 4]


class LocalMemoryWriter:
    """OpenAI-compatible local writer; no provider-specific paid API is required."""

    def __init__(self, model: str, base_url: str, api_key: str = "local"):
        if not model or not base_url:
            raise RuntimeError(
                "This memory method needs EVO_MEM_LLM_MODEL and "
                "EVO_MEM_LLM_BASE_URL (normally the local vLLM endpoint)."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the eval extra: pip install -e '.[eval,test]'") from exc
        self.model = model
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or "local",
            timeout=240.0,
            max_retries=2,
        )

    def complete(self, system: str, user: str, max_tokens: int = 256) -> tuple[str, MemoryCallStats]:
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        text = response.choices[0].message.content or ""
        if usage is None:
            input_tokens = count_tokens(system + "\n" + user)
            output_tokens = count_tokens(text)
        return text.strip(), MemoryCallStats(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency=time.perf_counter() - started,
        )


class ControlledMemory(BaseMemory):
    """Ablations and explicit learning mechanisms for cross-episode ALFWorld."""

    ARMS = ("bm25", "recent", "random")
    CONTEXT_DIM = 13

    def __init__(
        self,
        read_only: bool = False,
        mode: str = "random",
        memory_dir: str = "./storage",
        top_k: int = 5,
        chunk_size: int = 1024,
        max_context_tokens: int = 2048,
        random_seed: int = 0,
        exploration: float = 1.0,
        llm_model: str | None = None,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        consolidate_every: int = 5,
        **unused: Any,
    ):
        if mode not in MODES:
            raise ValueError(f"Unknown controlled memory mode {mode!r}; available={sorted(MODES)}")
        super().__init__(memory_type=f"controlled_{mode}", read_only=read_only)
        if top_k < 1 or chunk_size < 1 or max_context_tokens < 0:
            raise ValueError("top_k/chunk_size must be positive and context cap nonnegative")
        if consolidate_every < 1:
            raise ValueError("consolidate_every must be at least one")

        self.mode = mode
        self.memory_dir = Path(memory_dir)
        self.bank_path = self.memory_dir / "experience.jsonl"
        self.policy_path = self.memory_dir / "bandit_policy.json"
        self.bayes_path = self.memory_dir / "bayesian_state.json"
        self.semantic_path = self.memory_dir / "semantic_memory.json"
        self.principles_path = self.memory_dir / "expel_principles.json"
        self.top_k = int(top_k)
        self.chunk_size = int(chunk_size)
        self.max_context_tokens = int(max_context_tokens)
        self.random_seed = int(random_seed)
        self.exploration = float(exploration)
        self.consolidate_every = int(consolidate_every)
        self.llm_model = (
            llm_model or os.getenv("EVO_MEM_LLM_MODEL")
            or os.getenv("MEMORY_MODEL") or os.getenv("WORKER_MODEL")
        )
        self.llm_base_url = (
            llm_base_url or os.getenv("EVO_MEM_LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
        )
        self.llm_api_key = (
            llm_api_key or os.getenv("EVO_MEM_LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY") or "local"
        )
        self._writer: LocalMemoryWriter | None = None
        self.entries: list[dict] = []
        self.semantic_notes: list[str] = []
        self.principles: list[str] = []
        self.bayes: dict[str, dict] = {}
        self.counts = {arm: 0 for arm in self.ARMS}
        self.values = {arm: 0.0 for arm in self.ARMS}
        self.lin_a = {arm: np.eye(self.CONTEXT_DIM) for arm in self.ARMS}
        self.lin_b = {arm: np.zeros(self.CONTEXT_DIM) for arm in self.ARMS}
        self.last_arm: str | None = None
        self.last_context: np.ndarray | None = None
        self.lock = threading.RLock()
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.load_from_disk()

    def _get_writer(self) -> LocalMemoryWriter:
        if self._writer is None:
            self._writer = LocalMemoryWriter(
                self.llm_model or "",
                self.llm_base_url or "",
                self.llm_api_key or "local",
            )
        return self._writer

    @staticmethod
    def _task(conversation: list[dict]) -> str:
        if len(conversation) >= 3:
            return str(conversation[2].get("content") or "")
        return ""

    @staticmethod
    def _strategy(conversation: list[dict]) -> str:
        actions = []
        for message in conversation[3:]:
            if message.get("role") != "assistant":
                continue
            content = str(message.get("content") or "").strip()
            if content:
                actions.append(content[:220])
            if len(actions) == 6:
                break
        return "\n".join(actions)

    @staticmethod
    def _trajectory(conversation: list[dict]) -> str:
        clean = deepcopy(conversation)
        for message in clean:
            content = str(message.get("content") or "")
            marker = "\n\n--- Retrieved Memories ---"
            if marker in content:
                message["content"] = content.split(marker, 1)[0]
        try:
            return _format_trajectory(clean)
        except Exception:
            return "\n".join(
                f"{m.get('role', '?')}: {m.get('content', '')}"
                for m in clean[2:] if m.get("content")
            )

    def _append(self, record: dict) -> None:
        with self.bank_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.entries.append(record)

    def _write_json(self, path: Path, payload: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)

    def load_from_disk(self, memory_dir: str | None = None, **unused: Any) -> None:
        with self.lock:
            if memory_dir is not None:
                self.memory_dir = Path(memory_dir)
                self.bank_path = self.memory_dir / "experience.jsonl"
                self.policy_path = self.memory_dir / "bandit_policy.json"
                self.bayes_path = self.memory_dir / "bayesian_state.json"
                self.semantic_path = self.memory_dir / "semantic_memory.json"
                self.principles_path = self.memory_dir / "expel_principles.json"
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            self.entries = []
            if self.bank_path.exists():
                with self.bank_path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            self.entries.append(json.loads(line))
            self.semantic_notes = (
                json.loads(self.semantic_path.read_text(encoding="utf-8"))
                if self.semantic_path.exists() else []
            )
            self.principles = (
                json.loads(self.principles_path.read_text(encoding="utf-8"))
                if self.principles_path.exists() else []
            )
            self.bayes = (
                json.loads(self.bayes_path.read_text(encoding="utf-8"))
                if self.bayes_path.exists() else {}
            )
            self.counts = {arm: 0 for arm in self.ARMS}
            self.values = {arm: 0.0 for arm in self.ARMS}
            self.lin_a = {arm: np.eye(self.CONTEXT_DIM) for arm in self.ARMS}
            self.lin_b = {arm: np.zeros(self.CONTEXT_DIM) for arm in self.ARMS}
            if self.policy_path.exists():
                state = json.loads(self.policy_path.read_text(encoding="utf-8"))
                self.counts.update({k: int(v) for k, v in state.get("counts", {}).items() if k in self.ARMS})
                self.values.update({k: float(v) for k, v in state.get("values", {}).items() if k in self.ARMS})
                for arm in self.ARMS:
                    if arm in state.get("lin_a", {}):
                        self.lin_a[arm] = np.asarray(state["lin_a"][arm], dtype=float)
                    if arm in state.get("lin_b", {}):
                        self.lin_b[arm] = np.asarray(state["lin_b"][arm], dtype=float)

    def _save_policy(self) -> None:
        self._write_json(self.policy_path, {
            "counts": self.counts,
            "values": self.values,
            "lin_a": {k: v.tolist() for k, v in self.lin_a.items()},
            "lin_b": {k: v.tolist() for k, v in self.lin_b.items()},
        })

    def _random_order(self, query: str, records: list[dict]) -> list[dict]:
        digest = hashlib.sha256(query.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big") + self.random_seed
        candidates = list(records)
        random.Random(seed).shuffle(candidates)
        return candidates

    def _bm25(self, query: str, records: list[dict]) -> list[dict]:
        if not records:
            return []
        from rank_bm25 import BM25Okapi
        texts = [str(row.get("memory_text") or row.get("chunk") or "") for row in records]
        corpus = [_tokenize(text) for text in texts]
        if not any(corpus):
            return list(reversed(records))[:self.top_k]
        scores = BM25Okapi(corpus).get_scores(_tokenize(query))
        indices = sorted(
            range(len(records)),
            key=lambda i: (float(scores[i]), i),
            reverse=True,
        )
        return [records[i] for i in indices[:self.top_k]]

    def _context(self, query: str) -> np.ndarray:
        category = category_from_observation(query)
        category_bits = [float(category == name) for name in CATEGORIES]
        task = query.lower()
        keyword_bits = [
            float(bool(re.search(pattern, task)))
            for pattern in (
                r"\btwo\b|\bboth\b",
                r"\bclean\b",
                r"\bcool\b",
                r"\bheat(?:ed|ing)?\b",
                r"\blook at\b|\bexamine\b",
                r"\bput\b|\bplace\b",
            )
        ]
        return np.asarray([1.0] + category_bits + keyword_bits, dtype=float)

    def _choose_ucb_arm(self) -> str:
        for arm in self.ARMS:
            if self.counts[arm] == 0:
                return arm
        total = sum(self.counts.values())
        return max(
            self.ARMS,
            key=lambda arm: self.values[arm]
            + self.exploration * math.sqrt(math.log(total + 1) / self.counts[arm]),
        )

    def _choose_contextual_arm(self, context: np.ndarray) -> str:
        for arm in self.ARMS:
            if self.counts[arm] == 0:
                return arm
        scored = {}
        for arm in self.ARMS:
            a = self.lin_a[arm]
            theta = np.linalg.solve(a, self.lin_b[arm])
            uncertainty = math.sqrt(max(0.0, float(context @ np.linalg.solve(a, context))))
            scored[arm] = float(context @ theta) + self.exploration * uncertainty
        return max(self.ARMS, key=lambda arm: scored[arm])

    def _select(self, query: str) -> tuple[list[dict], str]:
        if self.mode == "random":
            return self._random_order(query, self.entries)[:self.top_k], "Random prior episodes"
        if self.mode == "recent":
            return list(reversed(self.entries))[:self.top_k], "Most recent prior episodes"
        if self.mode == "full_history":
            return list(reversed(self.entries)), "Full prior history (bounded)"
        if self.mode == "success_bm25":
            eligible = [x for x in self.entries if x.get("success")]
            return self._bm25(query, eligible), "BM25 over successful episodes"
        if self.mode == "outcome_bm25":
            return self._bm25(query, self.entries), "Outcome-labelled BM25"
        if self.mode == "bm25":
            return self._bm25(query, self.entries), "BM25 over all prior episodes"
        if self.mode == "bandit":
            self.last_arm = self._choose_ucb_arm()
            arm = self.last_arm
        elif self.mode == "contextual_bandit":
            self.last_context = self._context(query)
            self.last_arm = self._choose_contextual_arm(self.last_context)
            arm = self.last_arm
        elif self.mode in {"supervised_writer", "reflexion", "expel"}:
            candidates = [x for x in self.entries if x.get("memory_text")]
            return self._bm25(query, candidates), self.mode.replace("_", " ").title()
        elif self.mode == "letta_consolidation":
            candidates = [{"memory_text": x} for x in self.semantic_notes]
            return self._bm25(query, candidates), "Sleep-time consolidated semantic memory"
        else:
            return [], self.mode
        if arm == "bm25":
            return self._bm25(query, self.entries), "Adaptive policy selected BM25"
        if arm == "recent":
            return list(reversed(self.entries))[:self.top_k], "Adaptive policy selected recent"
        return self._random_order(query, self.entries)[:self.top_k], "Adaptive policy selected random"

    def _render(self, selected: list[dict], *, labels: bool = False) -> str:
        parts = []
        for row in selected:
            text = str(row.get("memory_text") or row.get("chunk") or "").strip()
            if not text:
                continue
            if labels:
                text = f"[{str(row.get('outcome', 'experience')).upper()}] {text}"
            parts.append(text)
        return capped("\n\n".join(parts), self.max_context_tokens)

    def _inject_bayesian(self, query: str) -> str:
        category = category_from_observation(query)
        state = self.bayes.get(category)
        if not state:
            return ""
        alpha = float(state.get("alpha", 1.0))
        beta = float(state.get("beta", 1.0))
        n = int(state.get("n", 0))
        if n == 0:
            return ""
        posterior_mean = alpha / (alpha + beta)
        uncertainty = math.sqrt(alpha * beta / (((alpha + beta) ** 2) * (alpha + beta + 1)))
        procedure = str(state.get("best_successful_procedure") or "").strip()
        text = (
            f"Bayesian task-family state for {category}: "
            f"success posterior Beta({alpha:.1f}, {beta:.1f}), "
            f"mean success probability {posterior_mean:.2f}, "
            f"posterior standard deviation {uncertainty:.2f}, based on {n} prior episodes."
        )
        if procedure:
            text += "\nA previously successful procedure in this task family:\n" + procedure
        return capped(text, self.max_context_tokens)

    def inject(self, conversation: list[dict]) -> tuple[list[dict], MemoryCallStats]:
        started = time.perf_counter()
        if len(conversation) < 3:
            return conversation, MemoryCallStats()
        query = self._task(conversation)
        if self.mode == "playbook":
            memory_text = PLAYBOOK[category_from_observation(query)]
            label = "Static category playbook (procedural positive control; not oracle memory)"
        elif self.mode == "bayesian_state":
            memory_text = self._inject_bayesian(query)
            label = "Structured Bayesian strategy state"
        else:
            selected, label = self._select(query)
            memory_text = self._render(
                selected,
                labels=self.mode in {"outcome_bm25", "bandit", "contextual_bandit"},
            )
        if not memory_text:
            return conversation, MemoryCallStats(latency=time.perf_counter() - started)
        output = deepcopy(conversation)
        output[0] = dict(output[0])
        output[0]["content"] = (
            str(output[0].get("content", ""))
            + f"\n\n--- Retrieved Memories ---\n[{label}]\n{memory_text}\n---"
        )
        return output, MemoryCallStats(latency=time.perf_counter() - started)

    def _record(self, conversation: list[dict], data_idx: int | None, reward: float | None) -> dict:
        task = self._task(conversation)
        trajectory = self._trajectory(conversation)
        chunks = _chunk_text(trajectory, self.chunk_size)
        success = reward is not None and float(reward) >= 1.0
        return {
            "item_id": None if data_idx is None else str(data_idx),
            "task": task,
            "category": category_from_observation(task),
            "reward": None if reward is None else float(reward),
            "success": bool(success),
            "outcome": "success" if success else "failure",
            "trajectory": trajectory,
            "chunk": "\n\n".join(chunks),
            "memory_text": "",
        }

    def _update_bayesian(self, record: dict, conversation: list[dict]) -> None:
        category = record["category"]
        state = self.bayes.setdefault(category, {
            "alpha": 1.0, "beta": 1.0, "n": 0, "best_reward": -1.0,
            "best_successful_procedure": "",
        })
        success = bool(record["success"])
        state["alpha"] = float(state["alpha"]) + (1.0 if success else 0.0)
        state["beta"] = float(state["beta"]) + (0.0 if success else 1.0)
        state["n"] = int(state["n"]) + 1
        if success and float(record["reward"] or 0.0) >= float(state["best_reward"]):
            state["best_reward"] = float(record["reward"] or 0.0)
            state["best_successful_procedure"] = self._strategy(conversation)
        self._write_json(self.bayes_path, self.bayes)

    def _update_bandit(self, reward: float | None) -> None:
        if reward is None or self.last_arm is None:
            return
        value = 1.0 if float(reward) >= 1.0 else 0.0
        arm = self.last_arm
        self.counts[arm] += 1
        self.values[arm] += (value - self.values[arm]) / self.counts[arm]
        if self.mode == "contextual_bandit" and self.last_context is not None:
            x = self.last_context
            self.lin_a[arm] += np.outer(x, x)
            self.lin_b[arm] += value * x
        self._save_policy()
        self.last_arm = None
        self.last_context = None

    def _write_lesson(self, record: dict, kind: str) -> tuple[str, MemoryCallStats]:
        writer = self._get_writer()
        success = bool(record["success"])
        if kind == "supervised_writer":
            system = (
                "You are a reward-supervised memory writer for an embodied household agent. "
                "Use the environment reward label as supervision. Extract only a concise, "
                "reusable procedure demonstrated by a successful trajectory. Never invent "
                "objects, actions, or facts. Output at most three actionable bullet points."
            )
            user = (
                f"Outcome label: {'SUCCESS' if success else 'FAILURE'}\n"
                f"Environment reward: {record['reward']}\nTask:\n{record['task']}\n"
                f"Trajectory:\n{record['trajectory']}\n"
                "If the trajectory failed, return exactly NO_MEMORY. If successful, "
                "extract only steps supported by the trajectory."
            )
        elif kind == "reflexion":
            system = (
                "You are a Reflexion-style agent reflecting on one completed ALFWorld task. "
                "Write a concise lesson for a future attempt. Distinguish successful actions "
                "from failed or invalid actions. Do not claim an unobserved action worked. "
                "Return at most three actionable sentences."
            )
            user = (
                f"Outcome: {'SUCCESS' if success else 'FAILURE'}; reward={record['reward']}\n"
                f"Task:\n{record['task']}\nTrajectory:\n{record['trajectory']}"
            )
        else:
            system = (
                "You are an ExpeL-style experience analyst. Extract a generalizable principle "
                "from this completed task, including a useful correction from failure when "
                "appropriate. Merge it conceptually with the existing principles; do not "
                "repeat them or invent facts. Return one short, standalone principle."
            )
            user = (
                "Existing principles:\n"
                + "\n".join(f"- {x}" for x in self.principles[-20:])
                + f"\n\nOutcome: {'SUCCESS' if success else 'FAILURE'}; reward={record['reward']}"
                + f"\nTask:\n{record['task']}\nTrajectory:\n{record['trajectory']}"
            )
        return writer.complete(system, user, max_tokens=220)

    def _consolidate(self) -> MemoryCallStats:
        pending = [row for row in self.entries if row.get("consolidated") is not True]
        if not pending:
            return MemoryCallStats()
        writer = self._get_writer()
        selected = pending[-self.consolidate_every:]
        system = (
            "You are a sleep-time memory consolidator inspired by agent memory systems. "
            "Consolidate episodic records into a compact, durable semantic/procedural memory. "
            "Merge duplicates, preserve important conditions and exceptions, and separate "
            "verified successful procedures from failed approaches. Never invent facts. "
            "Return only the updated memory as concise bullet points."
        )
        user = (
            "Current long-term memory:\n"
            + ("\n".join(f"- {x}" for x in self.semantic_notes) or "(empty)")
            + "\n\nNew completed episodes (all precede future tasks):\n"
            + "\n\n".join(
                f"Task: {x['task']}\nOutcome: {x['outcome']}\nTrajectory:\n{x['trajectory'][:3500]}"
                for x in selected
            )
        )
        text, stats = writer.complete(system, user, max_tokens=700)
        notes = [line.strip(" -*\t") for line in text.splitlines() if line.strip(" -*\t")]
        if notes:
            self.semantic_notes = notes[:40]
            self._write_json(self.semantic_path, self.semantic_notes)
            for row in selected:
                row["consolidated"] = True
            self._rewrite_bank()
        return stats

    def _rewrite_bank(self) -> None:
        temporary = self.bank_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in self.entries:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(self.bank_path)

    def update(
        self,
        conversation: list[dict],
        data_idx: int | None = None,
        reward: float | None = None,
    ) -> MemoryCallStats:
        started = time.perf_counter()
        if self.read_only or self.mode == "playbook":
            return MemoryCallStats()
        with self.lock:
            stats = MemoryCallStats()
            if self.mode in {"bandit", "contextual_bandit"}:
                self._update_bandit(reward)
            record = self._record(conversation, data_idx, reward)
            if self.mode == "bayesian_state":
                self._update_bayesian(record, conversation)
                return MemoryCallStats(latency=time.perf_counter() - started)
            if self.mode == "success_bm25" and not record["success"]:
                return MemoryCallStats(latency=time.perf_counter() - started)
            if self.mode == "supervised_writer":
                if record["success"]:
                    lesson, writer_stats = self._write_lesson(record, self.mode)
                    stats = stats + writer_stats
                    if lesson and lesson.upper() != "NO_MEMORY":
                        record["memory_text"] = lesson
                else:
                    return MemoryCallStats(latency=time.perf_counter() - started)
            elif self.mode in {"reflexion", "expel"}:
                lesson, writer_stats = self._write_lesson(record, self.mode)
                stats = stats + writer_stats
                if lesson:
                    record["memory_text"] = lesson
                    if self.mode == "expel":
                        self.principles.append(lesson)
                        self.principles = self.principles[-100:]
                        self._write_json(self.principles_path, self.principles)
            self._append(record)
            if self.mode == "letta_consolidation":
                pending = sum(1 for row in self.entries if row.get("consolidated") is not True)
                if pending >= self.consolidate_every:
                    stats = stats + self._consolidate()
            stats.latency = time.perf_counter() - started
            return stats

    def save(self) -> None:
        """Explicit persistence hook for tests and experiment shutdown."""
        with self.lock:
            self._save_policy()
            self._write_json(self.bayes_path, self.bayes)
            self._write_json(self.semantic_path, self.semantic_notes)
            self._write_json(self.principles_path, self.principles)
