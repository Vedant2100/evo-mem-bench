"""Deterministic memory-necessary preflight; not a replacement for EvoMemBench.

World secrets are randomly assigned under a seed. Current test questions never
contain the answer. Training observations reveal a rule that applies to new
parcels at the same station, so exact test-answer memorization is unnecessary.
"""
from __future__ import annotations

import hashlib
import random
import string
from dataclasses import dataclass

GATES = ("RILO", "TAVU", "MENO", "KASA", "PIVO", "DURA")
CHARS = string.ascii_uppercase


@dataclass(frozen=True)
class World:
    station: str
    tag: str
    gate: str
    train_parcel: str
    test_parcel: str

    @property
    def key(self) -> str:
        return self.station

    @property
    def discovery(self) -> str:
        return (
            f"FIELD LOG: At station {self.station}, a shipment tagged {self.tag} "
            f"was inspected. This station's permanent routing rule is: all "
            f"shipments tagged {self.tag} must pass through gate {self.gate}. "
            f"Observed parcel: {self.train_parcel}. This rule persists for later parcels."
        )

    @property
    def task(self) -> str:
        choices = ", ".join(GATES)
        return (
            f"At station {self.station}, parcel {self.test_parcel} with tag "
            f"{self.tag} requires routing. Select the correct gate from "
            f"[{choices}]. Answer with the FOUR-LETTER GATE CODE ONLY."
        )

    @property
    def oracle(self) -> str:
        return f"Remembered from an earlier inspection: at station {self.station}, tag {self.tag} goes to {self.gate}."

def _code(rng: random.Random, length: int) -> str:
    return "".join(rng.choices(CHARS, k=length))

def generate(seed: int, count: int) -> list[World]:
    if count < 6:
        raise ValueError("Need at least six independent worlds.")
    rng = random.Random(seed)
    worlds: list[World] = []
    used: set[str] = set()
    while len(worlds) < count:
        station = "ST-" + _code(rng, 8)
        if station in used:
            continue
        used.add(station)
        gate = GATES[len(worlds) % len(GATES)]
        worlds.append(World(
            station=station, tag="TAG-" + _code(rng, 5),
            gate=gate, train_parcel="TR-" + _code(rng, 7),
            test_parcel="TE-" + _code(rng, 7),
        ))
    rng.shuffle(worlds)
    return worlds

def manifest_hash(worlds: list[World]) -> str:
    payload = "|".join(f"{x.station}:{x.tag}:{x.gate}:{x.test_parcel}" for x in worlds)
    return hashlib.sha256(payload.encode()).hexdigest()

def check_disjoint(worlds: list[World]) -> None:
    assert len({w.station for w in worlds}) == len(worlds)
    assert all(w.train_parcel != w.test_parcel for w in worlds)
    assert len({w.test_parcel for w in worlds}) == len(worlds)
    assert set(w.gate for w in worlds) == set(GATES)
