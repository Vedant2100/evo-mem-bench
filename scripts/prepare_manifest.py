#!/usr/bin/env python3
"""Validate official ALFWorld maps and create deterministic disjoint streams."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = (
    "pick_and_place_simple",
    "pick_two_obj_and_place",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "look_at_obj_in_light",
)
TEST_COUNTS = (46, 45, 37, 28, 25, 19)
TRAIN_COUNTS = (672, 389, 448, 394, 321, 196)
COMPONENT = ROOT / "vendor/EvoMemBench/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB"


def category(task_type: str) -> str:
    for name in CATEGORIES:
        if task_type.startswith(name + "-"):
            return name
    raise ValueError(f"Unrecognized official ALFWorld task_type: {task_type}")


def parse_mapping(path: Path, expected_counts: tuple[int, ...], expected_start: int):
    data = path.read_bytes()
    rows = json.loads(data)
    if len(rows) != sum(expected_counts):
        raise ValueError(f"{path.name}: expected {sum(expected_counts)} rows, got {len(rows)}")
    ids = [int(row["item_id"]) for row in rows]
    if set(ids) != set(range(expected_start, expected_start + len(rows))):
        raise ValueError(f"{path.name}: IDs are not the expected contiguous range")
    task_ids = [row["task_id"] for row in rows]
    if len(set(task_ids)) != len(rows):
        raise ValueError(f"{path.name}: duplicated task_id")
    groups = defaultdict(list)
    for row in rows:
        groups[category(row["task_type"])].append(int(row["item_id"]))
    counts = [len(groups[name]) for name in CATEGORIES]
    if counts != list(expected_counts):
        raise ValueError(f"{path.name}: counts changed: {counts}")
    return rows, dict(groups), hashlib.sha256(data).hexdigest()


def load_maps():
    base = COMPONENT / "agentenv-alfworld/configs"
    train_rows, train_groups, train_hash = parse_mapping(
        base / "mappings_train.json", TRAIN_COUNTS, 0
    )
    test_rows, test_groups, test_hash = parse_mapping(
        base / "mappings_test.json", TEST_COUNTS, 2420
    )
    train_task_ids = {row["task_id"] for row in train_rows}
    test_task_ids = {row["task_id"] for row in test_rows}
    if train_task_ids & test_task_ids:
        raise ValueError("Leakage: official train/test task IDs overlap")
    return {
        "train": (train_rows, train_groups, train_hash),
        "test": (test_rows, test_groups, test_hash),
    }


def _ordered_ids(groups: dict, seed: int, kind: str) -> dict[str, list[int]]:
    result = {}
    for i, name in enumerate(CATEGORIES):
        ids = sorted(groups[name])
        if seed != 0:
            offset = 100000 if kind == "train" else 0
            random.Random(seed * 1009 + i * 17 + offset).shuffle(ids)
        result[name] = ids
    return result


def select(groups, seed: int, limit: int | None, kind: str) -> dict[str, list[int]]:
    if limit is not None and limit < 1:
        raise ValueError("Per-category selection count must be at least one")
    ordered = _ordered_ids(groups, seed, kind)
    return {
        name: ids if limit is None else ids[:limit]
        for name, ids in ordered.items()
    }


def split_train_validation(
    groups: dict,
    seed: int,
    train_limit: int | None,
    validation_limit: int | None,
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    if validation_limit is not None and validation_limit < 1:
        raise ValueError("validation_per_category must be at least one when supplied")
    if validation_limit is not None and train_limit is None:
        raise ValueError("A validation split requires a finite train_per_category")
    ordered = _ordered_ids(groups, seed, "train")
    train, validation = {}, {}
    for name, ids in ordered.items():
        n_train = len(ids) if train_limit is None else train_limit
        n_validation = 0 if validation_limit is None else validation_limit
        if n_train + n_validation > len(ids):
            raise ValueError(
                f"{name}: requested {n_train} train + {n_validation} validation "
                f"items, only {len(ids)} official train items exist"
            )
        train[name] = ids[:n_train]
        validation[name] = ids[n_train:n_train + n_validation]
    return train, validation


def create_manifest(
    seed: int,
    train_n: int | None = None,
    test_n: int | None = None,
    validation_n: int | None = None,
) -> dict:
    mappings = load_maps()
    train, validation = split_train_validation(
        mappings["train"][1], seed, train_n, validation_n
    )
    test = select(mappings["test"][1], seed, test_n, "test")
    train_ids = set(sum(train.values(), []))
    validation_ids = set(sum(validation.values(), []))
    test_ids = set(sum(test.values(), []))
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise ValueError("Leakage: train, validation, and test item IDs must be disjoint")
    return {
        "seed": seed,
        "train": train,
        "validation": validation,
        "test": test,
        "train_sha256": mappings["train"][2],
        "test_sha256": mappings["test"][2],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-per-category", type=int)
    parser.add_argument("--validation-per-category", type=int)
    parser.add_argument("--test-per-category", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    manifest = create_manifest(
        args.seed,
        args.train_per_category,
        args.test_per_category,
        args.validation_per_category,
    )
    print(json.dumps({
        "train_total": sum(map(len, manifest["train"].values())),
        "validation_total": sum(map(len, manifest["validation"].values())),
        "test_total": sum(map(len, manifest["test"].values())),
        "train_sha256": manifest["train_sha256"],
        "test_sha256": manifest["test_sha256"],
        "train_counts": {k: len(v) for k, v in manifest["train"].items()},
        "validation_counts": {k: len(v) for k, v in manifest["validation"].items()},
        "test_counts": {k: len(v) for k, v in manifest["test"].items()},
        "disjoint": (
            set(sum(manifest["train"].values(), [])).isdisjoint(
                set(sum(manifest["validation"].values(), []))
            )
            and set(sum(manifest["train"].values(), [])).isdisjoint(
                set(sum(manifest["test"].values(), []))
            )
            and set(sum(manifest["validation"].values(), [])).isdisjoint(
                set(sum(manifest["test"].values(), []))
            )
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
