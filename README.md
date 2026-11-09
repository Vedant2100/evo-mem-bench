# EvoMemBench — full memory-arm experiment platform

**Research question:** When does experience from earlier tasks improve a capable agent's performance on later tasks, and which parts of the memory pipeline matter: writing, consolidation, retrieval, selection, or applying retrieved information?

This repository runs the **official EvoMemBench CROSSEP-EMB ALFWorld evaluator**, using its task mappings, environment, reward, agent loop, and upstream memory adapters. The upstream checkout is pinned to 3efc221fd8d5f4de7c2dc2022357e600d26325c7. The experiment adds controlled memory conditions and research extensions without replacing the benchmark.

## The two primary protocols

1. **Held-out memory learning** (core, systems, or full plan): build each method's memory on a disjoint subset of the official train mapping; optionally use another disjoint train subset for evolutionary policy selection; freeze memory before evaluating on the official 200-item test mapping. This is the cleanest test of whether prior experience transfers to later held-out tasks.
2. **Original online stream** (original plan): process the official 200 test items sequentially, updating memory after each episode. This matches the upstream online setting but is not an independent test stream: earlier evaluation episodes become later memory.

The transfer plan adds the 30 directed source-category → target-category comparisons with source memory frozen on target tasks. Report this separately from held-out and online results.

## Memory arms

### Controls and retrieval baselines
- none — matched worker, no memory.
- playbook — static hand-authored task-category guidance; a procedural positive control, **not an oracle**.
- random, recent, full_history — irrelevant/random, most-recent, and all-history context controls.
- bm25, success_bm25, outcome_bm25 — original BM25, successful episodes only, and success/failure-labelled episodes.

### Learned memory and policy mechanisms
- supervised_writer — reward-conditioned local LLM writer; writes only lessons supported by successful trajectories. This is not writer-weight fine-tuning.
- reflexion — prompt-based per-episode reflection over successes and failures.
- expel — prompt-based extraction of generalizable principles from experience.
- letta_consolidation — episodic buffer with bounded sleep-time consolidation into semantic notes. Letta-inspired, not a claim to reproduce the Letta product.
- bayesian_state — structured Beta-Bernoulli posterior over task-family success, paired with a prior successful procedure. It is not a full POMDP state estimator.
- bandit — non-contextual UCB selection among BM25/recent/random.
- contextual_bandit — LinUCB selection among BM25/recent/random using task-family and task-verb features.
- evolutionary — genetic search over retrieval mode and top-k using a disjoint validation split from official training data. The selected policy is frozen before test. It optimizes retrieval policy, not model weights or the contents of memory notes.

### Upstream memory systems
- amem, amem_evolving
- reasoning_bank
- ace
- agent_kb
- awm (Agent Workflow Memory)
- mem0
- skillweaver
- lightweight_memory

The upstream methods are invoked through their original adapters. Local OpenAI-compatible inference is supported for their memory LLM calls; a separate local embedding endpoint is used by embedding-backed systems. Install optional dependencies only for the systems you intend to run.

### Separate parameter-learning comparison
Continual LoRA/SFT changes the worker weights and is **not** an external-memory arm. It has a separate data-collection, training, and evaluation path in docs/LORA_TRACK.md. The synthetic hidden-rule oracle is also separate; ALFWorld has no true gold-memory oracle.

See docs/METHODS.md for the complete method definitions and boundaries, and docs/SCIENTIFIC_PROTOCOL.md for the registered protocol.

## Recommended execution order

    # 1. Clone and bootstrap the pinned official benchmark.
    git clone https://github.com/Vedant2100/evo-mem-bench.git
    cd evo-mem-bench
    bash scripts/bootstrap_upstream.sh

    # 2. Install the evaluator environment.
    python -m pip install -e '.[eval,test]'
    python -m pytest -q

    # 3. Install only the optional upstream systems you will compare.
    bash scripts/install_memory_methods.sh amem agent_kb awm mem0

    # 4. Check the small plan before spending GPU time.
    cp .env.example .env
    python scripts/prepare_manifest.py --validate-only
    python scripts/run_experiment.py --plan smoke --dry-run
    python scripts/run_experiment.py --plan competence --dry-run
    python scripts/run_experiment.py --plan core --dry-run

    # 5. Run the full custom-method and upstream-system matrix only after the gate.
    python scripts/run_experiment.py --plan full --dry-run

See docs/CLUSTER_RUN.md for local vLLM worker/embedding servers, Slurm/Apptainer notes, and a Codex run checklist.

The default full plan uses 20 memory-building tasks and 5 validation tasks per category, then the complete official test set, for three task-order seeds. Adjust the counts and methods explicitly if the compute budget is smaller. Dry-run prints phase counts and an upper bound on worker calls; memory-writer and embedding calls are additional.

## Soundness guarantees

- Each arm/seed/category gets a fresh memory store.
- Held-out train, validation, and test item IDs are disjoint.
- Test is read-only in held-out experiments.
- Evolutionary candidates are selected on validation only; no test outcomes select hyperparameters.
- Worker model, prompt, temperature, round limit, task IDs, and order seed are held fixed across paired conditions.
- Memory-only context is capped at the same configured token budget in controlled runs. Original-fidelity uncapped runs must use memory_cap_tokens=0 across every compared arm.
- The runner uses parallel=1 for sequential memory updates; completed ALFWorld environments are closed in finally.
- Every condition is validated against per-episode output files. Missing/error episodes invalidate that condition.
- The runner refuses to reuse output directories. Upstream cached JSON files are not a valid way to resume a memory-learning stream because skipped tasks do not replay memory updates.
- Raw conversations, exact manifests, method configs, upstream commit, logs, and per-episode metrics are retained. API keys are excluded from run manifests.

## Analyze

    python scripts/analyze_results.py runs/RUN_ID

Held-out, online, and transfer results are summarized separately. Paired comparisons use the same item IDs; online uncertainty is treated cautiously because earlier tasks affect later memory.

## Companion positive control

The local hidden-rule diagnostic in docs/HIDDEN_RULES.md checks whether a worker can exploit genuinely hidden information from previous episodes, using an actual oracle arm. It is a single-turn diagnostic, not a substitute for the multi-step ALFWorld benchmark.
