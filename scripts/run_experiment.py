#!/usr/bin/env python3
"""Staged, paired experiments around the pinned official ALFWorld evaluator."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import os
import random
import re
import socket
import subprocess
import sys
import uuid
from pathlib import Path
from statistics import mean
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_manifest import CATEGORIES, COMPONENT, create_manifest


PACKAGE_IMPORTS = {
    "amem": ("agentic_memory",),
    "agent_kb": ("sklearn", "dotenv"),
    "awm": ("sklearn", "dotenv"),
    "mem0": ("mem0",),
}


def load_env() -> None:
    path = ROOT / ".env"
    if path.exists():
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def read_json(path: Path):
    return json.loads(path.read_text())


def substitution(value, variables):
    if isinstance(value, dict):
        return {key: substitution(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [substitution(item, variables) for item in value]
    if isinstance(value, str):
        def expand(match):
            name = match.group(1)
            if name not in variables:
                raise ValueError(f"Missing config variable {name}")
            return str(variables[name])
        return re.sub(r"@@([A-Z0-9_]+)@@", expand, value)
    return value


def sanitized(obj):
    if isinstance(obj, dict):
        return {
            key: ("***" if value else "") if (
                "api_key" in key.lower() or "secret" in key.lower()
            ) else sanitized(value)
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [sanitized(value) for value in obj]
    return obj


def validate_outputs(outdir: Path, indices: list[int]) -> None:
    missing, bad = [], []
    for index in indices:
        path = outdir / f"alfworld_{index}.json"
        if not path.exists():
            missing.append(index)
            continue
        try:
            result = read_json(path)
            assert result.get("item_id") == f"alfworld_{index}", "item_id mismatch"
            reward = float(result["reward"])
            success = int(result["success"])
            assert math.isfinite(reward) and success in (0, 1)
            assert success == int(reward >= 1.0)
            assert "error" not in result
        except Exception as exc:
            bad.append((index, str(exc)))
    if missing or bad:
        raise RuntimeError(
            f"INVALID condition {outdir}: missing={missing[:8]}, malformed={bad[:5]}"
        )
    summary = outdir / "summary.json"
    if not summary.exists() or int(read_json(summary)["num_samples"]) != len(indices):
        raise RuntimeError(f"INVALID upstream summary {summary}")


def config_file(
    run_dir: Path,
    seed: int,
    method: str,
    category: str,
    spec: dict,
    memory_dir: Path,
    variables: dict,
    config_key: str | None = None,
) -> Path | None:
    if spec["kind"] == "none":
        return None
    suffix = f"{config_key}_{category}" if config_key else category
    path = run_dir / "configs" / f"seed_{seed}" / method / f"{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if spec["kind"] in ("controlled", "evolutionary"):
        params = {
            "mode": spec["mode"],
            "memory_dir": str(memory_dir),
            "task_category": category,
            "top_k": 10,
            "chunk_size": 1024,
            "max_context_tokens": variables["MEMORY_CAP"],
            "random_seed": seed,
            "llm_model": variables["EVO_MEM_LLM_MODEL"],
            "llm_base_url": variables["EVO_MEM_LLM_BASE_URL"],
            "llm_api_key": variables["EVO_MEM_LLM_API_KEY"],
        }
        params.update(copy.deepcopy(spec.get("params", {})))
    else:
        params = substitution(
            copy.deepcopy(spec["params"]),
            {**variables, "MEMORY_DIR": str(memory_dir)},
        )
    path.write_text(json.dumps(params, indent=2) + "\n")
    return path


def condition(
    *,
    run_dir: Path,
    seed: int,
    method: str,
    spec: dict,
    phase: str,
    category: str,
    indices: list[int],
    model: str,
    mode: str,
    rounds: int,
    cap: int,
    port: int,
    env: dict,
    memory_dir: Path | None = None,
    cfg: Path | None = None,
    readonly: bool = False,
    source: str | None = None,
) -> Path:
    label = category if source is None else f"{source}__to__{category}"
    out = run_dir / f"seed_{seed}" / method / phase / label
    if out.exists():
        raise FileExistsError(f"Refusing to reuse condition output {out}")
    out.mkdir(parents=True)
    script = COMPONENT / "scripts/eval_alfworld_with_memory.py"
    cmd = [
        sys.executable, str(script),
        "--port", str(port),
        "--indices", json.dumps(indices),
        "--max_rounds", str(rounds),
        "--parallel", "1",
        "--api_mode", mode,
        "--output_dir", str(out),
    ]
    if mode == "standard":
        cmd.extend(["--model", model])
    if spec["kind"] == "none":
        cmd.append("--no_memory")
    else:
        memory_type = (
            "controlled" if spec["kind"] in ("controlled", "evolutionary")
            else spec["backend"]
        )
        cmd.extend(["--memory_type", memory_type, "--memory_config", str(cfg)])
        if readonly:
            cmd.append("--readonly_memory")
    meta = {
        "seed": seed,
        "method": method,
        "phase": phase,
        "category": category,
        "source": source,
        "read_only": readonly,
        "item_ids": indices,
        "model": model,
        "api_mode": mode,
        "max_rounds": rounds,
        "memory_cap_tokens": cap,
        "memory_store": str(memory_dir) if memory_dir else None,
        "memory_config": sanitized(read_json(cfg)) if cfg else None,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log = run_dir / "logs" / f"seed_{seed}" / method / phase / f"{label}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"RUN {phase} {method} {label} [{len(indices)} episodes]", flush=True)
    with log.open("w") as handle:
        completed = subprocess.run(
            cmd, cwd=COMPONENT, env=env, stdout=handle, stderr=subprocess.STDOUT
        )
    if completed.returncode:
        raise RuntimeError(f"Upstream evaluator returned {completed.returncode}; inspect {log}")
    validate_outputs(out, indices)
    return out


def context_tokens(record: dict) -> int:
    conversations = record.get("conversations") or []
    if not conversations:
        return 0
    system = str(conversations[0].get("content") or "")
    marker = "--- Retrieved Memories ---"
    if marker not in system:
        return 0
    suffix = system.split(marker, 1)[1]
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(suffix))
    except Exception:
        return max(1, math.ceil(len(suffix) / 4))


def evolutionary_candidates(search: dict, seed: int) -> list[list[dict]]:
    modes = list(search.get("modes", ["bm25", "success_bm25", "outcome_bm25", "recent", "full_history"]))
    top_ks = list(search.get("top_k_values", [1, 3, 5]))
    population_size = int(search.get("population_size", 4))
    generations = int(search.get("generations", 2))
    if not modes or not top_ks or population_size < 2 or generations < 1:
        raise ValueError("Invalid evolutionary search configuration")
    grid = [{"mode": mode, "top_k": int(k)} for mode in modes for k in top_ks]
    if population_size > len(grid):
        population_size = len(grid)
    rng = random.Random(seed + 20261109)
    population = rng.sample(grid, population_size)
    result = []
    for generation in range(generations):
        result.append(population)
        if generation == generations - 1:
            break
        next_population = [dict(x) for x in population[: min(2, len(population))]]
        attempts = 0
        while len(next_population) < population_size and attempts < 1000:
            attempts += 1
            parent = dict(rng.choice(next_population))
            if rng.random() < 0.5:
                parent["mode"] = rng.choice(modes)
            else:
                parent["top_k"] = int(rng.choice(top_ks))
            if parent not in next_population:
                next_population.append(parent)
        for candidate in grid:
            if len(next_population) >= population_size:
                break
            if candidate not in next_population:
                next_population.append(dict(candidate))
        population = next_population
    return result


def evolutionary_search(
    *,
    run_dir: Path,
    seed: int,
    base_spec: dict,
    memory_dir: dict[str, Path],
    validation: dict[str, list[int]],
    baseline_validation: dict[str, Path],
    variables: dict,
    worker: dict,
    env: dict,
    search: dict,
) -> tuple[dict, dict]:
    generations = evolutionary_candidates(search, seed)
    baseline_rows = []
    for category, indices in validation.items():
        baseline_dir = baseline_validation[category]
        for index in indices:
            baseline_rows.append(read_json(baseline_dir / f"alfworld_{index}.json"))
    baseline_success = mean(float(row["success"]) for row in baseline_rows) if baseline_rows else 0.0

    evaluated = []
    for generation_index, population in enumerate(generations):
        generation_results = []
        for candidate_index, candidate in enumerate(population):
            candidate_spec = {
                "kind": "controlled",
                "mode": candidate["mode"],
                "params": {
                    "top_k": int(candidate["top_k"]),
                    "chunk_size": 1024,
                    "max_context_tokens": int(variables["MEMORY_CAP"]),
                    "random_seed": seed,
                    "llm_model": variables["EVO_MEM_LLM_MODEL"],
                    "llm_base_url": variables["EVO_MEM_LLM_BASE_URL"],
                    "llm_api_key": variables["EVO_MEM_LLM_API_KEY"],
                },
            }
            candidate_method = f"evolutionary/search/gen_{generation_index}/candidate_{candidate_index}"
            rows = []
            injected_tokens = []
            for category, indices in validation.items():
                cfg = config_file(
                    run_dir, seed, candidate_method, category, candidate_spec,
                    memory_dir[category], variables,
                )
                out = condition(
                    run_dir=run_dir,
                    seed=seed,
                    method=candidate_method,
                    spec=candidate_spec,
                    phase="validation",
                    category=category,
                    indices=indices,
                    model=worker["model"],
                    mode=worker["api_mode"],
                    rounds=worker["max_rounds"],
                    cap=worker["memory_cap_tokens"],
                    port=worker["port"],
                    env=env,
                    memory_dir=memory_dir[category],
                    cfg=cfg,
                    readonly=True,
                )
                for index in indices:
                    record = read_json(out / f"alfworld_{index}.json")
                    rows.append(record)
                    injected_tokens.append(context_tokens(record))
            success_rate = mean(float(row["success"]) for row in rows) if rows else 0.0
            mean_context_tokens = mean(injected_tokens) if injected_tokens else 0.0
            result = {
                **candidate,
                "generation": generation_index,
                "candidate_index": candidate_index,
                "n": len(rows),
                "success_rate": success_rate,
                "paired_delta_vs_no_memory": success_rate - baseline_success,
                "mean_injected_context_tokens": mean_context_tokens,
            }
            generation_results.append(result)
            evaluated.append(result)
        generation_results.sort(
            key=lambda row: (row["success_rate"], -row["mean_injected_context_tokens"]),
            reverse=True,
        )
        # The next generation is mutated from the two best candidates.
        if generation_index + 1 < len(generations):
            elite = generation_results[: min(2, len(generation_results))]
            next_population = [dict(x) for x in elite]
            rng = random.Random(seed + 991 * (generation_index + 1))
            modes = list(search.get("modes", []))
            top_ks = list(search.get("top_k_values", []))
            while len(next_population) < len(generations[generation_index + 1]):
                parent = dict(rng.choice(elite))
                if rng.random() < 0.5:
                    parent["mode"] = rng.choice(modes)
                else:
                    parent["top_k"] = int(rng.choice(top_ks))
                candidate = {"mode": parent["mode"], "top_k": int(parent["top_k"])}
                if candidate not in next_population:
                    next_population.append(candidate)
                if len(next_population) >= len(generations[generation_index + 1]):
                    break
            generations[generation_index + 1] = next_population
    evaluated.sort(
        key=lambda row: (row["success_rate"], -row["mean_injected_context_tokens"]),
        reverse=True,
    )
    best = {"mode": evaluated[0]["mode"], "top_k": int(evaluated[0]["top_k"])}
    selection = {
        "seed": seed,
        "validation_item_ids": validation,
        "no_memory_validation_success": baseline_success,
        "candidate_results": evaluated,
        "selected": best,
        "selection_rule": "maximize validation success; tie-break by fewer injected context tokens",
        "test_items_used_for_selection": False,
    }
    path = run_dir / f"seed_{seed}" / "evolutionary" / "selection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection, indent=2) + "\n")
    return best, selection


def create_plan(args):
    load_env()
    config = read_json(ROOT / "configs/experiment.json")
    specs = read_json(ROOT / "configs/methods.json")
    plan = copy.deepcopy(config["plans"][args.plan])
    methods = [x.strip() for x in args.methods.split(",") if x.strip()] if args.methods else plan["methods"]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()] if args.seeds else plan["seeds"]
    if not methods or "none" not in methods or len(methods) != len(set(methods)):
        raise ValueError("Methods must include unique names and 'none' for paired baseline")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Need unique order seeds")
    unknown = set(methods) - set(specs)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")

    train_n = args.train_per_category if args.train_per_category is not None else plan.get("train_per_category")
    validation_n = (
        args.validation_per_category
        if args.validation_per_category is not None
        else plan.get("validation_per_category")
    )
    test_n = args.test_per_category if args.test_per_category is not None else plan.get("test_per_category")
    cap = (
        args.memory_cap_tokens if args.memory_cap_tokens is not None
        else plan.get("memory_cap_tokens", config["worker"]["memory_cap_tokens"])
    )
    if cap < 0:
        raise ValueError("Memory cap must be nonnegative")

    worker = copy.deepcopy(config["worker"])
    worker["api_mode"] = args.api_mode or os.environ.get("WORKER_API_MODE") or worker["api_mode"]
    worker["model"] = args.model or os.environ.get("WORKER_MODEL") or worker["model"]
    if worker["api_mode"] not in ("standard", "batch"):
        raise ValueError("Only upstream standard and ARK batch modes are supported")
    if worker["api_mode"] == "batch":
        worker["model"] = os.environ.get("BATCH_MODEL") or worker["model"]
    worker["port"] = int(os.environ.get("ALFWORLD_PORT") or worker["port"])
    worker["memory_cap_tokens"] = cap
    temperature = os.environ.get("WORKER_TEMPERATURE")
    worker["temperature"] = float(temperature) if temperature not in (None, "") else float(worker["temperature"])

    if "evolutionary" in methods and plan["protocol"] != "heldout":
        raise ValueError("Evolutionary search is validation-only; use a heldout plan with a validation split")
    if "evolutionary" in methods and validation_n is None:
        raise ValueError("Evolutionary search requires validation_per_category")

    manifests = {
        str(seed): create_manifest(seed, train_n, test_n, validation_n)
        for seed in seeds
    }
    protocol = plan["protocol"]
    writer_methods = [method for method in methods if method not in ("none", "playbook")]
    train_count = sum(sum(map(len, manifest["train"].values())) for manifest in manifests.values())
    validation_count = sum(sum(map(len, manifest["validation"].values())) for manifest in manifests.values())
    test_count = sum(sum(map(len, manifest["test"].values())) for manifest in manifests.values())
    if protocol == "heldout":
        train_episodes = train_count * len(writer_methods)
        test_episodes = test_count * len(methods)
        search_cfg = specs.get("evolutionary", {}).get("search", {})
        candidate_count = (
            int(search_cfg.get("population_size", 4)) * int(search_cfg.get("generations", 2))
            if "evolutionary" in methods else 0
        )
        validation_episodes = validation_count * (candidate_count + 1 if candidate_count else 0)
    elif protocol == "online":
        train_episodes = validation_episodes = 0
        test_episodes = test_count * len(methods)
    elif protocol == "transfer":
        train_episodes = validation_episodes = 0
        test_episodes = test_count * len(methods) + test_count * 5 * len(writer_methods)
    elif protocol == "collect":
        train_episodes = train_count * len(methods)
        validation_episodes = test_episodes = 0
    else:
        raise ValueError(f"Unknown protocol {protocol!r}")
    episodes = train_episodes + validation_episodes + test_episodes
    return (
        config, specs, plan, methods, seeds, manifests, worker, protocol,
        episodes, train_episodes, validation_episodes, test_episodes,
        train_n, validation_n, test_n,
    )


def assert_endpoint(url: str, label: str) -> None:
    parsed = urlparse(url)
    if not parsed.hostname:
        raise RuntimeError(f"{label} is not a valid URL: {url!r}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=5):
            pass
    except OSError as exc:
        raise RuntimeError(f"{label} is unreachable at {url}: {exc}") from exc


def check_method_dependencies(method: str, spec: dict) -> None:
    missing = []
    for package in spec.get("requires_packages", []):
        for module in PACKAGE_IMPORTS.get(package, ()):
            try:
                __import__(module)
            except ImportError:
                missing.append(module)
    if missing:
        raise RuntimeError(
            f"{method} is missing Python packages {sorted(set(missing))}. "
            f"Run: bash scripts/install_memory_methods.sh {method}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        choices=("smoke", "competence", "core", "systems", "full", "original", "learned", "transfer", "collect"),
        required=True,
    )
    parser.add_argument("--methods")
    parser.add_argument("--seeds")
    parser.add_argument("--train-per-category", type=int)
    parser.add_argument("--validation-per-category", type=int)
    parser.add_argument("--test-per-category", type=int)
    parser.add_argument("--memory-cap-tokens", type=int)
    parser.add_argument("--model")
    parser.add_argument("--api-mode", choices=("standard", "batch"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    try:
        (
            config, specs, plan, methods, seeds, manifests, worker, protocol,
            episodes, train_episodes, validation_episodes, test_episodes,
            train_n, validation_n, test_n,
        ) = create_plan(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    print(json.dumps({
        "plan": args.plan,
        "protocol": protocol,
        "methods": methods,
        "seeds": seeds,
        "worker_model": worker["model"],
        "api_mode": worker["api_mode"],
        "temperature": worker["temperature"],
        "max_rounds": worker["max_rounds"],
        "memory_cap_tokens": worker["memory_cap_tokens"],
        "train_per_category": train_n,
        "validation_per_category": validation_n,
        "test_per_category": test_n,
        "episodes": {
            "train": train_episodes,
            "validation_search_and_baseline": validation_episodes,
            "evaluation_and_transfer": test_episodes,
            "total": episodes,
        },
        "worker_call_upper_bound": episodes * worker["max_rounds"],
        "warning": (
            "Memory-writer and embedding calls are additional. Evolutionary candidates "
            "are selected only on the disjoint official-train validation split."
        ),
    }, indent=2))
    if args.dry_run:
        print("Dry-run only; no environment/inference calls or run directory created")
        return

    upstream_root = ROOT / "vendor/EvoMemBench"
    actual_upstream = subprocess.check_output(
        ["git", "-C", str(upstream_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_upstream != config["upstream"]["commit"]:
        raise RuntimeError("Pinned upstream commit mismatch")
    patch_marker = COMPONENT / ".controlled_patch.json"
    if not patch_marker.exists() or int(read_json(patch_marker).get("patch_version", 0)) < 2:
        raise RuntimeError("Run scripts/bootstrap_upstream.sh to install patch version 2")

    env = os.environ.copy()
    if worker["api_mode"] == "standard":
        if not env.get("OPENAI_BASE_URL") or not env.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_BASE_URL and OPENAI_API_KEY for the local OpenAI-compatible worker")
        assert_endpoint(env["OPENAI_BASE_URL"], "Worker inference endpoint")
    elif not env.get("DEEPSEEK_API_KEY") or not env.get("BATCH_MODEL"):
        raise RuntimeError("Set DEEPSEEK_API_KEY and BATCH_MODEL for ARK batch worker")

    env["WORKER_MODEL"] = worker["model"]
    env["EVO_MEM_LLM_BASE_URL"] = (
        env.get("EVO_MEM_LLM_BASE_URL") or env.get("OPENAI_BASE_URL", "")
    )
    env["EVO_MEM_LLM_MODEL"] = (
        env.get("EVO_MEM_LLM_MODEL") or env.get("MEMORY_MODEL") or worker["model"]
    )
    env["EVO_MEM_LLM_API_KEY"] = (
        env.get("EVO_MEM_LLM_API_KEY") or env.get("OPENAI_API_KEY") or "local"
    )
    env["EVO_MEM_EMBED_MODEL"] = (
        env.get("EVO_MEM_EMBED_MODEL") or "Qwen/Qwen3-Embedding-0.6B"
    )
    env["DASHSCOPE_BASE_URL"] = (
        env.get("DASHSCOPE_BASE_URL") or "http://127.0.0.1:8001/v1"
    )
    env["DASHSCOPE_API_KEY"] = env.get("DASHSCOPE_API_KEY") or "local"
    env["EVO_MEM_TEMPERATURE"] = str(worker["temperature"])
    env["EVO_MEM_MAX_INJECT_TOKENS"] = str(worker["memory_cap_tokens"])
    env["PYTHONPATH"] = str(COMPONENT / "agentenv") + os.pathsep + env.get("PYTHONPATH", "")

    variables = {
        **env,
        "MEMORY_DIR": "",
        "MEMORY_CAP": worker["memory_cap_tokens"],
        "EVO_MEM_LLM_BASE_URL": env["EVO_MEM_LLM_BASE_URL"],
        "EVO_MEM_LLM_MODEL": env["EVO_MEM_LLM_MODEL"],
        "EVO_MEM_LLM_API_KEY": env["EVO_MEM_LLM_API_KEY"],
        "EVO_MEM_EMBED_MODEL": env["EVO_MEM_EMBED_MODEL"],
    }
    for method in methods:
        spec = specs[method]
        for required in spec.get("requires", []):
            if not variables.get(required):
                raise RuntimeError(f"{method} requires environment variable {required}")
        check_method_dependencies(method, spec)

    with socket.create_connection(("127.0.0.1", worker["port"]), timeout=5):
        pass
    if any(specs[method].get("requires") for method in methods):
        if any("EVO_MEM_LLM_BASE_URL" in specs[method].get("requires", []) for method in methods):
            assert_endpoint(env["EVO_MEM_LLM_BASE_URL"], "Memory LLM endpoint")
        if any("DASHSCOPE_BASE_URL" in specs[method].get("requires", []) for method in methods):
            assert_endpoint(env["DASHSCOPE_BASE_URL"], "Memory embedding endpoint")

    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:7]
    run_dir = Path(args.output_dir).resolve() if args.output_dir else ROOT / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)

    allowed_env = (
        "BATCH_MODEL", "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL",
        "OPENAI_API_KEY", "OPENAI_BASE_URL", "EVO_MEM_LLM_BASE_URL",
        "EVO_MEM_LLM_MODEL", "EVO_MEM_LLM_API_KEY", "EVO_MEM_EMBED_MODEL",
        "DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL",
        "EVO_MEM_TEMPERATURE", "EVO_MEM_MAX_INJECT_TOKENS",
    )
    (COMPONENT / ".env").write_text(
        "\n".join(f"{key}={env[key]}" for key in allowed_env if key in env) + "\n"
    )

    repo_commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    meta = {
        "run_id": run_id,
        "plan": args.plan,
        "protocol": protocol,
        "worker": worker,
        "repository_commit": repo_commit,
        "upstream_commit": actual_upstream,
        "patch_version": 2,
        "methods": methods,
        "specs": specs,
        "seeds": seeds,
        "manifests": manifests,
        "evolutionary_choices": {},
        "finished": False,
    }
    (run_dir / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")

    for seed in seeds:
        selected = manifests[str(seed)]
        for method in methods:
            spec = specs[method]
            dirs, cfgs = {}, {}
            for category in CATEGORIES:
                if spec["kind"] != "none":
                    store = run_dir / f"seed_{seed}" / method / "memory" / category
                    store.mkdir(parents=True)
                    dirs[category] = store
                    cfgs[category] = config_file(
                        run_dir, seed, method, category, spec, store, variables
                    )

            if protocol == "collect":
                for category in CATEGORIES:
                    condition(
                        run_dir=run_dir, seed=seed, method=method, spec=spec,
                        phase="train", category=category, indices=selected["train"][category],
                        model=worker["model"], mode=worker["api_mode"],
                        rounds=worker["max_rounds"], cap=worker["memory_cap_tokens"],
                        port=worker["port"], env=env,
                        memory_dir=dirs.get(category), cfg=cfgs.get(category),
                    )
                continue

            if protocol == "heldout":
                if method not in ("none", "playbook"):
                    for category in CATEGORIES:
                        condition(
                            run_dir=run_dir, seed=seed, method=method, spec=spec,
                            phase="train", category=category, indices=selected["train"][category],
                            model=worker["model"], mode=worker["api_mode"],
                            rounds=worker["max_rounds"], cap=worker["memory_cap_tokens"],
                            port=worker["port"], env=env,
                            memory_dir=dirs[category], cfg=cfgs[category],
                        )

                eval_spec = spec
                eval_cfgs = cfgs
                if method == "evolutionary":
                    baseline_validation = {}
                    for category in CATEGORIES:
                        baseline_validation[category] = condition(
                            run_dir=run_dir, seed=seed, method="none", spec=specs["none"],
                            phase="validation", category=category,
                            indices=selected["validation"][category],
                            model=worker["model"], mode=worker["api_mode"],
                            rounds=worker["max_rounds"], cap=worker["memory_cap_tokens"],
                            port=worker["port"], env=env,
                        )
                    best, selection = evolutionary_search(
                        run_dir=run_dir,
                        seed=seed,
                        base_spec=spec,
                        memory_dir=dirs,
                        validation=selected["validation"],
                        baseline_validation=baseline_validation,
                        variables=variables,
                        worker=worker,
                        env=env,
                        search=spec["search"],
                    )
                    meta["evolutionary_choices"][str(seed)] = selection
                    (run_dir / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
                    eval_spec = {
                        "kind": "controlled",
                        "mode": best["mode"],
                        "params": {
                            "top_k": best["top_k"],
                            "chunk_size": 1024,
                            "max_context_tokens": worker["memory_cap_tokens"],
                            "random_seed": seed,
                            "llm_model": env["EVO_MEM_LLM_MODEL"],
                            "llm_base_url": env["EVO_MEM_LLM_BASE_URL"],
                            "llm_api_key": env["EVO_MEM_LLM_API_KEY"],
                        },
                    }
                    eval_cfgs = {
                        category: config_file(
                            run_dir, seed, method, category, eval_spec,
                            dirs[category], variables, config_key="selected",
                        )
                        for category in CATEGORIES
                    }

                for category in CATEGORIES:
                    condition(
                        run_dir=run_dir, seed=seed, method=method, spec=eval_spec,
                        phase="eval", category=category, indices=selected["test"][category],
                        model=worker["model"], mode=worker["api_mode"],
                        rounds=worker["max_rounds"], cap=worker["memory_cap_tokens"],
                        port=worker["port"], env=env,
                        memory_dir=dirs.get(category), cfg=eval_cfgs.get(category),
                        readonly=spec["kind"] != "none",
                    )
            elif protocol == "online":
                for category in CATEGORIES:
                    condition(
                        run_dir=run_dir, seed=seed, method=method, spec=spec,
                        phase="online", category=category, indices=selected["test"][category],
                        model=worker["model"], mode=worker["api_mode"],
                        rounds=worker["max_rounds"], cap=worker["memory_cap_tokens"],
                        port=worker["port"], env=env,
                        memory_dir=dirs.get(category), cfg=cfgs.get(category),
                    )
            elif protocol == "transfer":
                for category in CATEGORIES:
                    condition(
                        run_dir=run_dir, seed=seed, method=method, spec=spec,
                        phase="online", category=category, indices=selected["test"][category],
                        model=worker["model"], mode=worker["api_mode"],
                        rounds=worker["max_rounds"], cap=worker["memory_cap_tokens"],
                        port=worker["port"], env=env,
                        memory_dir=dirs.get(category), cfg=cfgs.get(category),
                    )
                if method not in ("none", "playbook"):
                    for source in CATEGORIES:
                        for target in CATEGORIES:
                            if source == target:
                                continue
                            condition(
                                run_dir=run_dir, seed=seed, method=method, spec=spec,
                                phase="transfer", category=target, source=source,
                                indices=selected["test"][target],
                                model=worker["model"], mode=worker["api_mode"],
                                rounds=worker["max_rounds"], cap=worker["memory_cap_tokens"],
                                port=worker["port"], env=env,
                                memory_dir=dirs[source], cfg=cfgs[source], readonly=True,
                            )

    meta["finished"] = True
    (run_dir / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Completed validated run at {run_dir}")
    if protocol != "collect":
        print(f"Analyze: python scripts/analyze_results.py {run_dir}")


if __name__ == "__main__":
    main()
