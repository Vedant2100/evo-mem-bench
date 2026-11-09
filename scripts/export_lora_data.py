#!/usr/bin/env python3
"""Export successful no-memory ALFWorld train trajectories for separate LoRA SFT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collect_run", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/lora_train.jsonl"))
    args = parser.parse_args()

    run_dir = args.collect_run.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("protocol") != "collect":
        raise ValueError("Input run must have protocol=collect; do not train on online/test episodes")
    if manifest.get("methods") != ["none"]:
        raise ValueError("LoRA data collection must use only the no-memory worker")
    if not manifest.get("finished"):
        raise ValueError("Collection run is incomplete")
    if manifest.get("worker", {}).get("memory_cap_tokens", 0) < 0:
        raise ValueError("Invalid collection manifest")

    rows = []
    used_ids = set()
    skipped = {"unsuccessful": 0, "missing_conversation": 0}
    for seed, split in manifest["manifests"].items():
        train_ids = set(sum(split["train"].values(), []))
        validation_ids = set(sum(split.get("validation", {}).values(), []))
        test_ids = set(sum(split["test"].values(), []))
        if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
            raise ValueError(f"Manifest leakage detected for seed {seed}")
        for category, indices in split["train"].items():
            out_dir = run_dir / f"seed_{seed}" / "none" / "train" / category
            for index in indices:
                path = out_dir / f"alfworld_{index}.json"
                if not path.exists():
                    raise FileNotFoundError(f"Missing collected train trajectory: {path}")
                record = json.loads(path.read_text())
                if record.get("item_id") != f"alfworld_{index}":
                    raise ValueError(f"Item ID mismatch in {path}")
                if int(record.get("success", 0)) != 1:
                    skipped["unsuccessful"] += 1
                    continue
                conversation = record.get("conversations")
                if not isinstance(conversation, list) or not conversation:
                    skipped["missing_conversation"] += 1
                    continue
                messages = []
                for message in conversation:
                    role = message.get("role")
                    content = message.get("content")
                    if role not in {"system", "user", "assistant"} or content is None:
                        continue
                    if not isinstance(content, str):
                        content = str(content)
                    if content.strip():
                        messages.append({"role": role, "content": content})
                if not any(m["role"] == "assistant" for m in messages):
                    skipped["missing_conversation"] += 1
                    continue
                if index not in train_ids or index in validation_ids or index in test_ids:
                    raise ValueError(f"Non-training item {index} found in SFT source")
                rows.append({
                    "messages": messages,
                    "item_id": int(index),
                    "seed": int(seed),
                    "category": category,
                    "reward": float(record["reward"]),
                    "source_run": run_dir.name,
                    "source_phase": "official_train_no_memory",
                })
                used_ids.add(int(index))

    if len(rows) < 10:
        raise RuntimeError(
            f"Only {len(rows)} successful trajectories; collect more official train tasks before SFT"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source_run": str(run_dir),
        "protocol": "official_train_no_memory_only",
        "successful_episodes": len(rows),
        "unique_item_ids": len(used_ids),
        "item_ids": sorted(used_ids),
        "skipped": skipped,
        "output": str(args.output.resolve()),
        "test_items_used": False,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
