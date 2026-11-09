# EvoMemBench — controlled memory-learning experiments

**Research question:** Can a capable, *frozen* agent turn experience from earlier tasks into measurable improvements on later tasks? Which memory mechanisms help, and which merely add distracting context?

This project **reuses the official [EvoMemBench](https://github.com/DSAIL-Memory/EvoMemBench) CROSSEP-EMB ALFWorld evaluator, server, task maps, rewards, and memory adapters**. Upstream is pinned to `3efc221fd8d5f4de7c2dc2022357e600d26325c7`. No replacement simulator or alternative success metric.

## Two scientifically distinct protocols

1. **Original online protocol** (`--plan original`): 200 official test tasks in six categories, sequential memory updates, followed optionally by the original 30 directed source→target read-only transfer comparisons (`--plan transfer`). Seed 0 retains official item ordering. This is **not** an independent held-out test: earlier evaluation episodes become memory for later episodes.
2. **Held-out protocol** (`--plan heldout`): build memory exclusively on separate official **train** items (0–2419), freeze memory, evaluate on official **test** items (2420–2619). Train/test IDs and task IDs are disjoint. This is an **extension**, not a reproduction of published EvoMemBench results.

No experiment outcome is claimed until complete episode files exist and pass validation.

## Implemented methods

- **No memory** — matched worker baseline.
- **Static playbook** — hand-authored category procedure, **not** a true hidden-information oracle. A worker-utilization positive control.
- **Random / recent history** — equal-cap context controls.
- **Upstream BM25** — original chunking/retrieval adapter.
- **Success-only BM25 / outcome-labelled BM25** — custom ablations using the upstream BM25 formatting/chunking.
- **UCB retrieval bandit** — a lightweight reward-adaptive choice among BM25/recent/random; **not** actor–critic RL.
- **A-MEM, ReasoningBank, ACE** — official adapters with original configuration switches; optional dependencies and API keys needed.
- **Other official adapters** (Mem0, AgentKB, AWM, SkillWeaver, etc.) remain in upstream; this project does not falsely claim they have all been validated.

Further methods—evolutionary memory search, supervised memory-writer training, Bayesian belief-state agents, parameter-space LoRA, and actor–critic memory policies—require distinct training/evaluation protocols; see [methods and boundaries](docs/METHODS.md). Do not count them as completed experiments.

## Competence / causal interpretation gate

Before large runs, test **no memory + playbook** with a fixed worker. Flag overall no-memory success below 55%, category success below 30%, or ceiling above 95%. A playbook that fails to improve task performance does *not* demonstrate that learned memory is useless: it may be the wrong guidance or the benchmark may have no memory-sensitive headroom. **ALFWorld has no guaranteed hidden-information dependency**, unlike genuinely personalized or partially observable persistent worlds. If the positive control fails, stop and reconsider the testbed.

The original and held-out protocols must be reported separately. All methods in a comparison use the same worker, inference mode, max rounds, seeds/order, test IDs, and nominal memory-context cap. Upstream-fidelity runs explicitly disable the cap; compare them only to equally uncapped controls.

## Quick start

```bash
git clone https://github.com/Vedant2100/evo-mem-bench.git
cd evo-mem-bench
bash scripts/bootstrap_upstream.sh
python -m pip install -e '.[test]'
pytest -q
cp .env.example .env
```

Set up the ALFWorld server and evaluator according to [docs/SETUP.md](docs/SETUP.md). A local OpenAI-compatible endpoint (e.g. vLLM serving a competent Qwen model) is supported; or choose the upstream DeepSeek ARK batch route. Start the server separately.

```bash
python scripts/run_experiment.py --plan smoke --dry-run
python scripts/run_experiment.py --plan smoke
python scripts/run_experiment.py --plan heldout --dry-run
python scripts/run_experiment.py --plan original --dry-run
python scripts/run_experiment.py --plan transfer --dry-run
```

**No expensive experiment is started automatically.** Inspect the dry-run episode counts and worker-call bound before running. The recommended sequence is smoke → competence pilot → held-out core methods → original online reproduction → transfer → expensive adapters.

```bash
python scripts/analyze_results.py runs/RUN_ID
```

Every run records model/provider configuration (not secrets), pinned upstream commit, task-map hashes, seed, train/test IDs, raw per-episode JSON, method settings, and per-condition logs. Missing/error episodes invalidate results. An interrupted online memory run must be restarted with fresh storage, not resumed from the upstream JSON cache.

## Files

- `configs/`: worker/method/plan configuration.
- `src/evo_mem_bench/`: compact custom memory-control backend.
- `scripts/`: bootstrap, patcher, task manifest, runner, and analysis.
- `docs/`: scientific protocol, compatibility, limitations, setup.
- `tests/`: offline correctness checks. GitHub Actions also bootstraps/compiles upstream.

This is an **experiment platform**, not a claim that ALFWorld proves open-ended continual learning. Strong evidence requires held-out gains, competence calibration, proper cost accounting, and replications.

## Memory-necessary positive control (local Qwen)

Before interpreting ALFWorld memory gains or null results, run the companion
[hidden-rule memory-necessity diagnostic](docs/HIDDEN_RULES.md). It creates
new parcels requiring station-specific facts revealed only in earlier episodes,
with oracle, no-memory, irrelevant history, BM25, success filtering, outcome
labels, a UCB retrieval governor and a local-LLM compressed-memory writer.
It is intentionally a small single-turn positive control, **not** a replacement
for the original multi-step EvoMemBench evaluation.

~~~bash
pip install -e ".[eval,test]"
python -m evomembench.experiment --dry-run
python -m evomembench.experiment --model Qwen/Qwen3-32B \
  --base-url http://127.0.0.1:8000/v1 --worlds 36 --seeds 11,22,33 \
  --gate --output runs/hidden_rules_01
~~~

## Free Kaggle GPU runner (optional)

A GitHub Actions controller can launch the repository's **hidden-rule diagnostic** on a free Kaggle notebook GPU, monitor status every 30 minutes, and collect JSON results. It is **not yet the full ALFWorld evaluator**. GPU launches are explicitly opt-in and require Kaggle credentials stored as GitHub Actions secrets. See [Kaggle setup](docs/KAGGLE.md).
