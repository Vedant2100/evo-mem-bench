"""Deterministic memory interventions and one local-LLM writer.

Every method receives the *same* discovery stream and frozen test questions.
The worker never receives previous test questions/results. A bandit can update
its retrieval policy from feedback, but not its memory documents.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

MODES = (
    "none", "oracle", "random", "full_history", "bm25",
    "success_only", "reward_labeled", "ucb", "writer",
)

def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Z0-9-]+", text.upper())

@dataclass
class Note:
    station: str
    text: str
    verified: bool

class Memory:
    def __init__(self, mode: str, *, seed: int, budget_chars: int = 1600, top_k: int = 2,
                 worker: Any = None):
        if mode not in MODES:
            raise ValueError(f"Unknown memory mode: {mode}")
        if budget_chars < 80:
            raise ValueError("Context budget too small.")
        self.mode = mode
        self.rng = random.Random(seed)
        self.notes: list[Note] = []
        self.budget = budget_chars
        self.top_k = top_k
        self.worker = worker
        self.arms = ("bm25", "recent", "random")
        self.arm_count = {arm: 0 for arm in self.arms}
        self.arm_mean = {arm: 0.0 for arm in self.arms}
        self.last_arm: str | None = None
        self.writer_usage: list[dict] = []

    def add(self, station: str, observation: str, verified: bool) -> None:
        if self.mode in ("none", "oracle"):
            return
        if self.mode == "success_only" and not verified:
            return
        if self.mode == "writer":
            if not verified:
                return
            if self.worker is None:
                raise RuntimeError("Writer condition requires an LLM.")
            text, usage = self.worker.generate(
                "You compress a verified past observation into one faithful reusable memory. "
                "Retain the exact station, tag and gate. Do not invent facts. "
                "Return a single short sentence with no explanation.",
                observation,
                max_tokens=100,
            )
            self.writer_usage.append(usage)
            if not text.strip():
                return
        else:
            text = observation
        self.notes.append(Note(station=station, text=text, verified=verified))

    def _rank(self, query: str, notes: list[Note]) -> list[Note]:
        if not notes:
            return []
        corpus = [tokens(n.text) for n in notes]
        scores = BM25Okapi(corpus).get_scores(tokens(query))
        ordering = sorted(range(len(notes)), key=lambda i: (float(scores[i]), -i), reverse=True)
        return [notes[i] for i in ordering[:self.top_k]]

    def _choose_arm(self) -> str:
        for arm in self.arms:
            if self.arm_count[arm] == 0:
                return arm
        total = sum(self.arm_count.values())
        return max(
            self.arms,
            key=lambda arm: self.arm_mean[arm] + math.sqrt(2 * math.log(total + 1) / self.arm_count[arm]),
        )

    def _render(self, selected: list[Note], *, label: bool) -> str:
        items = [
            (f"[{'VERIFIED' if n.verified else 'REJECTED'}] " if label else "") + n.text
            for n in selected
        ]
        return "\n".join(items)[:self.budget]

    def inject(self, *, station: str, query: str, oracle: str) -> str:
        self.last_arm = None
        if self.mode == "none":
            return ""
        if self.mode == "oracle":
            return oracle[:self.budget]
        if not self.notes:
            return ""
        if self.mode == "full_history":
            return self._render(list(reversed(self.notes)), label=False)
        if self.mode == "random":
            irrelevant = [n for n in self.notes if n.station != station]
            return self._render(self.rng.sample(irrelevant, min(self.top_k, len(irrelevant))), label=False)
        if self.mode == "ucb":
            arm = self._choose_arm()
            self.last_arm = arm
        elif self.mode in ("bm25", "success_only", "reward_labeled", "writer"):
            arm = "bm25"
        else:
            raise AssertionError(self.mode)
        if arm == "bm25":
            selected = self._rank(query, self.notes)
        elif arm == "recent":
            selected = list(reversed(self.notes))[:self.top_k]
        else:
            selected = self.rng.sample(self.notes, min(self.top_k, len(self.notes)))
        return self._render(selected, label=self.mode in ("reward_labeled", "ucb"))

    def feedback(self, success: bool) -> None:
        if self.mode != "ucb" or self.last_arm is None:
            return
        arm = self.last_arm
        self.arm_count[arm] += 1
        count = self.arm_count[arm]
        self.arm_mean[arm] += (int(success) - self.arm_mean[arm]) / count
        self.last_arm = None
