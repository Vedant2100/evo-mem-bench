"""Run no-cost-provider-required memory calibration with local Qwen.

Example:
  python -m evomembench.experiment --base-url http://127.0.0.1:8000/v1 \
      --model Qwen/Qwen3-8B --worlds 36 --seeds 11,22,33
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from .hidden_rules import GATES, check_disjoint, generate, manifest_hash
from .local_model import LocalWorker, normalize_answer
from .memory import MODES, Memory

SYSTEM = (
    "You route parcels in a simulated warehouse. Each station has a private, "
    "persistent routing rule learned only through earlier inspections. "
    "Respond with exactly one four-letter gate code from the current question. "
    "Use remembered rules only when they match the current station and tag. "
    "Ignore previously rejected routing attempts."
)

def discovery_stream(worlds, seed):
    events = []
    for world in worlds:
        wrong = next(g for g in GATES if g != world.gate)
        failed = (
            f"REJECTED LOG: At station {world.station}, a parcel tagged {world.tag} "
            f"was sent to gate {wrong}. The validator explicitly REJECTED that route. "
            f"This is not the station routing rule."
        )
        events.append((world.station, failed, False))
        events.append((world.station, world.discovery, True))
    # Keep reject-before-verify within each world; shuffle world blocks only.
    blocks = [events[i:i+2] for i in range(0, len(events), 2)]
    random.Random(seed).shuffle(blocks)
    return [x for block in blocks for x in block]

def evaluate_condition(
    worker, *, mode, worlds, events, seed, budget, top_k, out_dir: Path,
):
    memory = Memory(mode, seed=seed, budget_chars=budget, top_k=top_k, worker=worker)
    for station, observation, verified in events:
        memory.add(station, observation, verified)
    results = []
    for i, world in enumerate(worlds):
        instruction = memory.inject(
            station=world.station,
            query=f"station {world.station} tag {world.tag}",
            oracle=world.oracle,
        )
        prompt = world.task
        if instruction:
            prompt = "PRIOR EXPERIENCE (not the current task):\n" + instruction + "\n\nCURRENT TASK:\n" + world.task
        try:
            answer, usage = worker.generate(SYSTEM, prompt, max_tokens=128)
        except Exception as exc:
            (out_dir / f"{mode}.error.json").write_text(json.dumps({
                "error": str(exc), "world_index": i, "mode": mode,
            }, indent=2))
            raise RuntimeError(f"{mode} failed at world {i}; entire condition invalid") from exc
        prediction = normalize_answer(answer)
        success = prediction == world.gate
        memory.feedback(success)
        results.append({
            "world_index": i, "station": world.station,
            "expected": world.gate, "predicted": prediction,
            "raw_response": answer, "success": success,
            "injected_memory": instruction, "injected_chars": len(instruction),
            "usage": usage,
        })
    output = {
        "mode": mode, "seed": seed, "n": len(results),
        "success_rate": mean(int(r["success"]) for r in results),
        "mean_injected_chars": mean(r["injected_chars"] for r in results),
        "writer_calls": len(memory.writer_usage),
        "writer_usage": memory.writer_usage,
        "ucb_counts": memory.arm_count, "rows": results,
    }
    (out_dir / f"{mode}.json").write_text(json.dumps(output, indent=2) + "\n")
    return output

def competence_summary(results: dict[str, dict]) -> dict:
    n = results["none"]["n"]
    no_memory = results["none"]["success_rate"]
    oracle = results["oracle"]["success_rate"]
    return {
        "worlds": n, "chance": 1 / len(GATES),
        "no_memory": no_memory, "oracle": oracle,
        "oracle_delta": oracle - no_memory,
        "worker_can_use_visible_rule": oracle >= .80,
        "memory_is_needed": no_memory <= .40,
        "memory_headroom_confirmed": oracle >= .80 and no_memory <= .40 and oracle - no_memory >= .40,
    }

def run(args):
    modes = list(dict.fromkeys(args.methods.split(",")))
    if not modes or any(mode not in MODES for mode in modes):
        raise ValueError(f"Unknown method in {modes}; options={MODES}")
    if "none" not in modes or "oracle" not in modes:
        raise ValueError("Include none and oracle in every run for competence/necessity gates.")
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate seeds.")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Refusing to reuse run directory; different conditions must never inherit state.")
    if args.dry_run:
        print(json.dumps({
            "worlds_per_seed": args.worlds, "seeds": seeds,
            "methods": modes, "worker_calls": args.worlds * len(seeds) * len(modes),
            "writer_calls": args.worlds * len(seeds) if "writer" in modes else 0,
        }, indent=2))
        return
    worker = LocalWorker(args.model, args.base_url, args.api_key, args.temperature)
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps({
        "model": args.model, "base_url": args.base_url, "temperature": args.temperature,
        "worlds": args.worlds, "seeds": seeds, "methods": modes,
        "budget_chars": args.budget_chars, "top_k": args.top_k,
        "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evaluation": "synthetic hidden-rule preflight, not EvoMemBench score",
    }, indent=2) + "\n")
    all_summaries = []
    for seed in seeds:
        worlds = generate(seed, args.worlds)
        check_disjoint(worlds)
        order = list(worlds)
        random.Random(seed + 1234).shuffle(order)
        events = discovery_stream(worlds, seed + 984)
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir()
        (seed_dir / "manifest.json").write_text(json.dumps({
            "seed": seed, "sha256": manifest_hash(worlds),
            "worlds": [asdict(world) for world in worlds],
            "events": [{"station": s, "observation": t, "verified": v} for s, t, v in events],
            "evaluation_order": [w.station for w in order],
        }, indent=2) + "\n")
        # Always run the two gates first. Stop before costly learned methods if
        # the worker cannot use a known rule or no-memory already knows it.
        condition_order = ["none", "oracle"] + [x for x in modes if x not in ("none", "oracle")]
        records = {}
        for mode in condition_order:
            if args.gate and mode not in ("none", "oracle") and not competence_summary(records)["memory_headroom_confirmed"]:
                break
            records[mode] = evaluate_condition(
                worker, mode=mode, worlds=order, events=events,
                seed=seed, budget=args.budget_chars, top_k=args.top_k, out_dir=seed_dir,
            )
        gate = competence_summary(records)
        summary = {"seed": seed, "competence_gate": gate,
                   "results": {k: {"success_rate": v["success_rate"], "n": v["n"]} for k, v in records.items()}}
        all_summaries.append(summary)
        (seed_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    (output / "summary.json").write_text(json.dumps(all_summaries, indent=2) + "\n")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("MODEL", "Qwen/Qwen3-8B"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "local"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--worlds", type=int, default=36)
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--methods", default="none,oracle,random,full_history,bm25,success_only,reward_labeled,ucb,writer")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--budget-chars", type=int, default=1600)
    parser.add_argument("--output", default="runs/hidden_rules_v1")
    parser.add_argument("--gate", action="store_true", default=False,
                        help="Stop after no-memory and oracle if the competence/memory-need gate fails.")
    parser.add_argument("--dry-run", action="store_true")
    run(parser.parse_args())

if __name__ == "__main__":
    main()
