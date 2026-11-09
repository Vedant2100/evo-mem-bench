#!/usr/bin/env python3
"""Paired per-task comparison of base and LoRA no-memory worker runs."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean

from prepare_manifest import CATEGORIES


def load_run(run_dir: Path):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if not manifest.get("finished"):
        raise ValueError(f"Incomplete run: {run_dir}")
    if manifest.get("protocol") != "online" or manifest.get("methods") != ["none"]:
        raise ValueError("Both runs must be original/online protocol with methods=['none']")
    if manifest.get("seeds") != [0]:
        raise ValueError("For this comparison, run both conditions with exactly seed 0")
    rows = {}
    for category in CATEGORIES:
        directory = run_dir / "seed_0" / "none" / "online" / category
        expected = manifest["manifests"]["0"]["test"][category]
        for item_id in expected:
            path = directory / f"alfworld_{item_id}.json"
            if not path.exists():
                raise FileNotFoundError(path)
            row = json.loads(path.read_text())
            if row.get("item_id") != f"alfworld_{item_id}" or "error" in row:
                raise ValueError(f"Invalid episode result: {path}")
            rows[int(item_id)] = {
                "category": category,
                "success": int(row["success"]),
                "reward": float(row["reward"]),
            }
    return manifest, rows


def bootstrap_ci(values, draws=5000, seed=7):
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        sample = [values[rng.randrange(len(values))] for _ in values]
        samples.append(mean(sample))
    samples.sort()
    return [samples[int(0.025 * draws)], samples[int(0.975 * draws)]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_run", type=Path)
    parser.add_argument("lora_run", type=Path)
    parser.add_argument("--output", type=Path, default=Path("lora_paired_comparison.json"))
    args = parser.parse_args()

    base_manifest, base = load_run(args.base_run.resolve())
    lora_manifest, lora = load_run(args.lora_run.resolve())
    if set(base) != set(lora):
        raise ValueError("Base and LoRA runs do not contain identical test item IDs")
    if base_manifest["manifests"]["0"]["test"] != lora_manifest["manifests"]["0"]["test"]:
        raise ValueError("Task manifest mismatch; do not compare different test selections")

    paired = []
    for item_id in sorted(base):
        paired.append({
            "item_id": item_id,
            "category": base[item_id]["category"],
            "base_success": base[item_id]["success"],
            "lora_success": lora[item_id]["success"],
            "success_delta": lora[item_id]["success"] - base[item_id]["success"],
            "base_reward": base[item_id]["reward"],
            "lora_reward": lora[item_id]["reward"],
            "reward_delta": lora[item_id]["reward"] - base[item_id]["reward"],
        })
    success_deltas = [row["success_delta"] for row in paired]
    reward_deltas = [row["reward_delta"] for row in paired]
    summary = {
        "base_model": base_manifest["worker"]["model"],
        "lora_model": lora_manifest["worker"]["model"],
        "n_paired_test_tasks": len(paired),
        "base_success_rate": mean(row["base_success"] for row in paired),
        "lora_success_rate": mean(row["lora_success"] for row in paired),
        "paired_success_delta": mean(success_deltas),
        "paired_success_delta_bootstrap_95_ci": bootstrap_ci(success_deltas),
        "base_mean_reward": mean(row["base_reward"] for row in paired),
        "lora_mean_reward": mean(row["lora_reward"] for row in paired),
        "paired_reward_delta": mean(reward_deltas),
        "paired_reward_delta_bootstrap_95_ci": bootstrap_ci(reward_deltas),
        "by_category": {},
        "paired_rows": paired,
        "same_test_items": True,
        "both_conditions_no_external_memory": True,
    }
    for category in CATEGORIES:
        group = [row for row in paired if row["category"] == category]
        summary["by_category"][category] = {
            "n": len(group),
            "base_success_rate": mean(row["base_success"] for row in group),
            "lora_success_rate": mean(row["lora_success"] for row in group),
            "paired_success_delta": mean(row["success_delta"] for row in group),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "paired_rows"}, indent=2))


if __name__ == "__main__":
    main()
