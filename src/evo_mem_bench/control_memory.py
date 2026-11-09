"""Controlled memory ablations for the *official* CROSSEP-EMB BaseMemory interface.

The module is copied into upstream scripts/memory/controlled_adapter.py by the
strict patcher. It reuses upstream BM25's trajectory formatting/chunking when
available, so success filtering is a meaningful ablation of the same data.
"""
from __future__ import annotations
import hashlib
import json
import math
import random
import re
import threading
import time
from copy import deepcopy
from pathlib import Path

try:
    from .base import BaseMemory, MemoryCallStats
except ImportError:  # Permits isolated unit tests before bootstrapping upstream.
    class BaseMemory:
        def __init__(self, memory_type, read_only=False):
            self.memory_type, self.read_only = memory_type, read_only
    class MemoryCallStats:
        def __init__(self, input_tokens=0, output_tokens=0, embedding_tokens=0,
                     cached_tokens=0, latency=0.0):
            self.input_tokens, self.output_tokens = input_tokens, output_tokens
            self.embedding_tokens, self.cached_tokens = embedding_tokens, cached_tokens
            self.latency = latency

try:
    from .bm25_adapter import _chunk_text, _format_trajectory, _tokenize
except ImportError:
    def _tokenize(text):
        return text.split()
    def _chunk_text(text, chunk_size=1024):
        return [text]
    def _format_trajectory(conversation):
        if len(conversation) < 3:
            return ""
        parts = [f"Task: {conversation[2].get('content', '')}", "[Execution Steps]"]
        for message in conversation[2:]:
            if message.get("content"):
                parts.append(f"{message.get('role','?')}: {message['content']}")
        return "\n".join(parts)

PLAYBOOK = {
    "pick_and_place_simple":
        "Identify the object and destination. Locate and take the object, place it in the requested receptacle, and verify task completion.",
    "pick_two_obj_and_place":
        "Find both required objects. Transfer each to the destination receptacle and verify that both were placed correctly.",
    "pick_clean_then_place_in_recep":
        "Take the target object, clean it with the appropriate sink/faucet action, retrieve it, then place it at the target destination.",
    "pick_cool_then_place_in_recep":
        "Take the target object, put it in the refrigerator until the environment indicates it is cool, retrieve it, then place it at the target.",
    "pick_heat_then_place_in_recep":
        "Take the target object, heat it using the appropriate appliance, retrieve it, then place it at the requested destination.",
    "look_at_obj_in_light":
        "Find the object and an active lamp, position the object at the light source, examine it, and verify success.",
}

def category_from_observation(observation):
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

def capped(text, max_tokens):
    if max_tokens <= 0:
        return text
    import tiktoken
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)
    return text if len(tokens) <= max_tokens else encoder.decode(tokens[:max_tokens])


class ControlledMemory(BaseMemory):
    MODES = {"playbook", "random", "recent", "success_bm25",
             "outcome_bm25", "bandit"}
    ARMS = ("bm25", "recent", "random")

    def __init__(self, read_only=False, mode="random", memory_dir="./storage",
                 top_k=10, chunk_size=1024, max_context_tokens=2048,
                 random_seed=0, exploration=1.0, **unused):
        super().__init__(memory_type=f"controlled_{mode}", read_only=read_only)
        if mode not in self.MODES:
            raise ValueError(f"Unknown controlled memory mode: {mode}")
        self.mode = mode
        self.memory_dir = Path(memory_dir)
        self.bank_path = self.memory_dir / "experience.jsonl"
        self.policy_path = self.memory_dir / "bandit.json"
        self.top_k = int(top_k)
        self.chunk_size = int(chunk_size)
        self.max_context_tokens = int(max_context_tokens)
        self.random_seed = int(random_seed)
        self.exploration = float(exploration)
        self.entries = []
        self.counts = {arm: 0 for arm in self.ARMS}
        self.values = {arm: 0.0 for arm in self.ARMS}
        self.last_arm = None
        self.lock = threading.RLock()
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.load_from_disk()

    def load_from_disk(self, memory_dir=None, **unused):
        with self.lock:
            if memory_dir is not None:
                self.memory_dir = Path(memory_dir)
                self.bank_path = self.memory_dir / "experience.jsonl"
                self.policy_path = self.memory_dir / "bandit.json"
            self.entries = []
            if self.bank_path.exists():
                with self.bank_path.open(encoding="utf-8") as handle:
                    self.entries = [json.loads(line) for line in handle if line.strip()]
            self.counts = {arm: 0 for arm in self.ARMS}
            self.values = {arm: 0.0 for arm in self.ARMS}
            if self.policy_path.exists():
                state = json.loads(self.policy_path.read_text())
                self.counts.update({k: int(v) for k,v in state.get("counts", {}).items() if k in self.ARMS})
                self.values.update({k: float(v) for k,v in state.get("values", {}).items() if k in self.ARMS})

    def _random_order(self, query):
        digest = hashlib.sha256(query.encode()).digest()
        seed = int.from_bytes(digest[:8], "big") + self.random_seed
        candidates = self.entries[:]
        random.Random(seed).shuffle(candidates)
        return candidates

    def _bm25(self, query, candidates):
        if not candidates:
            return []
        from rank_bm25 import BM25Okapi
        tokens = [_tokenize(record["chunk"]) for record in candidates]
        scores = BM25Okapi(tokens).get_scores(_tokenize(query))
        indices = sorted(range(len(candidates)), key=lambda i: (float(scores[i]), -i), reverse=True)
        return [candidates[i] for i in indices[:self.top_k]]

    def _choose_arm(self):
        for arm in self.ARMS:
            if self.counts[arm] == 0:
                return arm
        total = sum(self.counts.values())
        return max(self.ARMS, key=lambda arm: (
            self.values[arm] + self.exploration * math.sqrt(math.log(total+1) / self.counts[arm])
        ))

    def _retrieve(self, query):
        if not self.entries:
            return []
        if self.mode == "random":
            return self._random_order(query)[:self.top_k]
        if self.mode == "recent":
            return self.entries[-self.top_k:][::-1]
        if self.mode == "success_bm25":
            return self._bm25(query, [x for x in self.entries if x["success"]])
        if self.mode == "outcome_bm25":
            return self._bm25(query, self.entries)
        if self.mode == "bandit":
            self.last_arm = self._choose_arm()
            if self.last_arm == "bm25":
                return self._bm25(query, self.entries)
            if self.last_arm == "recent":
                return self.entries[-self.top_k:][::-1]
            return self._random_order(query)[:self.top_k]
        return []

    def inject(self, conversation):
        started = time.perf_counter()
        if len(conversation) < 3:
            return conversation, MemoryCallStats()
        query = str(conversation[2].get("content", ""))
        self.last_arm = None
        if self.mode == "playbook":
            cat = category_from_observation(query)
            memory_text = PLAYBOOK[cat]
            label = f"Category playbook ({cat}; static positive control)"
        else:
            selected = self._retrieve(query)
            if not selected:
                return conversation, MemoryCallStats(latency=time.perf_counter()-started)
            if self.mode == "outcome_bm25":
                memory_text = "\n\n".join(f"[{row['outcome']}] {row['chunk']}" for row in selected)
            else:
                memory_text = "\n\n".join(row["chunk"] for row in selected)
            label = f"Controlled {self.mode}" + (f" arm={self.last_arm}" if self.last_arm else "")
        memory_text = capped(memory_text, self.max_context_tokens)
        if not memory_text:
            return conversation, MemoryCallStats(latency=time.perf_counter()-started)
        output = deepcopy(conversation)
        output[0]["content"] += f"\n\n--- Retrieved Memories ---\n[{label}]\n{memory_text}\n---"
        return output, MemoryCallStats(latency=time.perf_counter()-started)

    def update(self, conversation, data_idx=None, reward=None):
        started = time.perf_counter()
        if self.read_only or self.mode == "playbook":
            return MemoryCallStats()
        success = reward is not None and float(reward) >= 1.0
        with self.lock:
            if self.mode == "bandit" and self.last_arm is not None and reward is not None:
                arm = self.last_arm
                self.counts[arm] += 1
                count = self.counts[arm]
                self.values[arm] += (float(reward)-self.values[arm])/count
                self.policy_path.write_text(json.dumps({"counts":self.counts,"values":self.values})+"\n")
            self.last_arm = None
            if self.mode == "success_bm25" and not success:
                return MemoryCallStats(latency=time.perf_counter()-started)
            trajectory = _format_trajectory(conversation)
            if not trajectory.strip():
                return MemoryCallStats(latency=time.perf_counter()-started)
            chunks = _chunk_text(trajectory, self.chunk_size)
            with self.bank_path.open("a", encoding="utf-8") as handle:
                for index, chunk in enumerate(chunks):
                    entry = {
                        "item_id": str(data_idx), "chunk_index": index,
                        "reward": reward, "success": success,
                        "outcome": "success" if success else "failure", "chunk": chunk,
                    }
                    handle.write(json.dumps(entry, ensure_ascii=False)+"\n")
                    self.entries.append(entry)
        return MemoryCallStats(latency=time.perf_counter()-started)
