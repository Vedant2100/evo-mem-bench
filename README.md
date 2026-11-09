# EvoMemBench: controlled ALFWorld memory experiments

This repository contains a reproducible, controlled extension of the **Cross-Episode Embodied AI (CROSSEP-EMB)** track from [DSAIL-Memory/EvoMemBench](https://github.com/DSAIL-Memory/EvoMemBench).

The project asks:

> When an agent encounters related ALFWorld tasks over time, does persistent memory improve later task success—and is the gain caused by useful experience rather than extra context, a stronger worker, or an evaluation artifact?

It reuses the upstream ALFWorld environment, task mapping, evaluator, memory adapters, and reward signal. It adds matched controls, sequential execution, explicit run manifests, validation, and paired analysis.

**Status:** experiment harness and protocol are being established. No experiment results are claimed by this repository until a run produces validated per-episode outputs.

## Experimental design

### Upstream benchmark retained

- Upstream repository: `DSAIL-Memory/EvoMemBench`
- Pinned upstream commit: `3efc221fd8d5f4de7c2dc2022357e600d26325c7`
- Component: `Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB`
- Official test mapping: `agentenv-alfworld/configs/mappings_test.json`
- 200 task instances, six ALFWorld task categories, 30 directed cross-category transfer pairs.
- Original success reward and task indices are retained.

The upstream source is fetched at the pinned commit by `scripts/bootstrap_upstream.sh`; it is not copied or silently forked into this repository.

### Controlled comparisons

The primary worker is fixed across conditions. The default worker configuration mirrors the upstream DeepSeek-V3.2 path; the actual model identifier, endpoint mode, temperature, prompt budget, and run timestamp are saved in each run manifest.

Conditions:

1. **No memory** — matched worker baseline.
2. **Random retrieval** — controls for receiving irrelevant prior context.
3. **Full history** — controls for retaining recent experience without selective retrieval.
4. **Oracle procedure** — positive control for whether the worker can use relevant procedural guidance.
5. **Upstream BM25** — original trajectory-retrieval baseline.
6. **Success-only BM25** — isolates whether filtering to successful experience helps.
7. **Upstream learned memory systems** — A-MEM, ReasoningBank, ACE, and other CROSSEP-EMB adapters when their dependencies and credentials are configured.

The custom controls are not presented as methods from the EvoMemBench paper. The upstream methods remain identifiable by their original adapter names.

### Two evaluation settings

- **In-environment learning:** each category's episodes are run sequentially; memory written after one episode is available to later episodes in that category.
- **Cross-environment transfer:** memory is built on a source category, then evaluated read-only on each of the other five target categories. The diagonal is excluded, giving 30 directed pairs.

For every target category, the no-memory baseline is evaluated on the same target task IDs, with the same worker configuration and episode order. Cross-environment comparisons are paired by target task ID.

## Soundness safeguards

- All conditions use the same worker model, API mode, action budget, and episode IDs within a comparison.
- Sequential execution (`--parallel 1`) prevents concurrent writes and order races in memory stores.
- Every condition/seed gets a fresh memory directory and output directory.
- Existing cached episode files are never reused for a new memory-learning run.
- The full 200-task mapping is checked for uniqueness, category counts, and cross-category task-ID disjointness.
- Source and target task IDs are audited; object overlap is reported rather than hidden.
- Optional memory-context truncation is applied uniformly in the controlled track. The unmodified upstream setting remains available for reproduction-oriented runs.
- Errors and missing episode files invalidate a run; they are not silently scored as failures or omitted.
- Analysis uses paired task-level comparisons and reports category-level results, learning curves, and uncertainty. The 30 transfer pairs are not treated as 30 independent datasets.

See [the protocol](docs/PROTOCOL.md), [upstream audit](docs/UPSTREAM_AUDIT.md), and [experiment design](docs/EXPERIMENT_DESIGN.md).

## Quick start

### 1. Bootstrap the pinned upstream code

```bash
bash scripts/bootstrap_upstream.sh
```

### 2. Create the two ALFWorld environments

The upstream setup uses separate server and evaluator environments. See [setup instructions](docs/PROTOCOL.md#environment-setup).

### 3. Configure credentials

Copy `.env.example` to `.env`. Never commit `.env`.

The default matched worker path uses the upstream ARK batch client. Memory systems may additionally require DashScope embeddings or their own optional packages.

### 4. Start the ALFWorld server

```bash
bash scripts/start_alfworld_server.sh
```

### 5. Validate and dry-run

```bash
python scripts/prepare_manifest.py --validate-only
python scripts/run_experiment.py --plan smoke --dry-run
```

### 6. Run the smoke test

```bash
python scripts/run_experiment.py --plan smoke
```

The smoke plan is deliberately small and does not produce paper-quality results.

### 7. Run the full primary comparison

```bash
python scripts/run_experiment.py --plan primary --dry-run
python scripts/run_experiment.py --plan primary
```

The primary plan is cost-bearing. Inspect the dry-run episode/API-call estimate and configure the method list before launching it. Full cross-category transfer is a separate stage.

## Repository layout

```text
configs/                 experiment and method configurations
docs/                    protocol, design, and upstream audit
scripts/                 bootstrap, manifest, run, validation, analysis
src/evo_mem_bench/       custom control memory adapter
tests/                   offline tests for manifest and analysis logic
vendor/                  pinned upstream checkout (created by bootstrap)
runs/                    raw outputs and run manifests (gitignored)
```

## Important interpretation limits

CROSSEP-EMB tests transfer of procedures across related ALFWorld task instances. It does **not** by itself demonstrate open-ended continual learning, model-weight improvement, or recursive self-improvement. A positive result supports a narrower claim: external memory can improve performance on later related embodied tasks under this protocol.

ALFWorld's 200-task subset is small and category-imbalanced (19–46 tasks per category). The same target episodes recur across the five source-to-target comparisons for a target category. The analysis therefore reports task-paired results and does not count transfer pairs as independent samples.

## References

- Wang et al., [EvoMemBench: Benchmarking Agent Memory from a Self-Evolving Perspective](https://arxiv.org/abs/2605.18421), 2026.
- [Official EvoMemBench repository](https://github.com/DSAIL-Memory/EvoMemBench).
- [ALFWorld](https://github.com/alfworld/alfworld).

