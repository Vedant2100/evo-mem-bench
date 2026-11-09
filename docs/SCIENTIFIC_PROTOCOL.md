# Registered experimental protocol

## Research question

Does persistent external memory improve later ALFWorld task success when the worker is competent enough to use it? Which intervention is responsible: storing raw experience, filtering failures, writing concise lessons, consolidating episodes, retrieving relevant items, selecting a retrieval policy, or adapting the worker weights?

## Official benchmark and splits

Use the pinned upstream CROSSEP-EMB ALFWorld task maps, environment, action parser, reward, worker prompt, and evaluation loop. The official train mapping contains 2,420 items (IDs 0–2419); the official test mapping contains 200 items (IDs 2420–2619) across six task categories. The mapping files are hash-checked at run creation.

The standard held-out plan creates three disjoint partitions:
- **Memory-build train:** default 20 tasks per category.
- **Policy-selection validation:** default 5 additional official-train tasks per category, used only by the evolutionary retrieval-policy arm.
- **Final test:** all 200 official test tasks, read-only for every memory arm.

For seed 0, the official order is retained within each category. Other seeds deterministically shuffle the task order and memory/validation samples. Train, validation, and test item IDs are checked for disjointness. All task IDs, seed-specific selections, and source mapping hashes are stored in the manifest.

The default full plan uses three order/memory seeds. A single seed is a pilot, not a robust replication.

## Protocol tracks

### Held-out memory learning
For each method, seed, and task category:
1. Create a fresh independent memory store.
2. Run the memory-build train subset sequentially, updating memory after each completed task.
3. For the evolutionary method only, evaluate candidate retrieval policies read-only on the separate validation subset. Select the best by validation success, breaking ties by lower injected-context token count.
4. Freeze the chosen memory store and policy.
5. Evaluate the complete official test set read-only.

No test episode can update memory or select a configuration. The baseline no-memory and static playbook conditions are evaluated on the same test IDs.

### Original online stream
Run the official 200 test tasks sequentially per category, updating memory after each task. Earlier test-stream outcomes may influence later tasks by design. This matches the online adaptation structure, but it is **not** an independent held-out test. Do not combine it with held-out estimates.

### Cross-category transfer
For each source category, build a memory bank from its sequential online stream. Then run each of the five other target categories with that source memory frozen. There are 6 × 5 = 30 directed source→target pairs. The same target test tasks recur across multiple source conditions; those comparisons are paired, not independent datasets.

### Continual LoRA/SFT
This is a separate parameter-learning intervention. Collect successful worker trajectories only from official train tasks, fine-tune a LoRA adapter, and evaluate that adapter on official test tasks with no external memory. Do not report it as a memory-system arm or mix its score into the memory-backend analysis.

## Methods

See docs/METHODS.md for each arm's exact implementation. The ALFWorld matrix includes no-memory, static playbook, random/recent/full history, BM25 and outcome-filtered retrieval, non-contextual UCB, contextual LinUCB, reward-supervised LLM writing, Reflexion-style reflection, ExpeL-style principle extraction, Letta-style sleep consolidation, Bayesian task-family state, validation-only evolutionary retrieval-policy search, and the selected official upstream adapters (A-MEM, ReasoningBank, ACE, AgentKB, AWM, Mem0, SkillWeaver, Lightweight Memory).

The true oracle is only available in the companion hidden-rule synthetic diagnostic. The ALFWorld playbook is a static procedural positive control, not gold memory.

## Controls and competence gate

Before full runs, compare no memory and static playbook using the same worker and decoding. Predeclared diagnostic flags:
- no-memory overall success below 55%;
- any task category below 30%;
- no-memory ceiling above 95%.

If the worker is weak, memory null results may reflect insufficient agent capability. If the playbook does not improve performance, this does not prove that memory is useless: the playbook may be inadequate, and ALFWorld does not guarantee that historical memory is necessary. Use the hidden-rule positive control to separately verify that the worker can apply genuinely hidden facts when exposed.

The pilot gate is a stop-and-review criterion, not a significance test. Report raw rates and uncertainty; do not silently drop a weak category.

## Outcomes and cost accounting

**Primary outcome:** binary success, defined by the official environment reward >= 1.0.

**Secondary outcomes:** mean official reward, success by category, success by task order/learning curve, worker prompt/completion tokens, memory-writer prompt/completion tokens, embedding tokens, memory-context tokens, action rounds, environment latency, LLM-only latency, memory-update/retrieval latency.

The evaluator reports upstream adapter usage fields. Some upstream adapters count tokens by estimates; others report API usage. Label estimates explicitly and do not treat them as provider-billed totals. For local vLLM, usage fields are model-reported tokens, not a dollar invoice.

## Fairness and leakage controls

- Identical worker checkpoint/model ID, system prompt, action parser, decoding temperature, max rounds, task IDs, and order seed within each paired comparison.
- Independent fresh memory store for every arm × seed × category.
- Held-out test is read-only. Evolutionary candidate policies are read-only on validation and selected without test results.
- The injected memory suffix is capped independently of the original task prompt. Controlled plans default to 2,048 memory tokens. For uncapped upstream-fidelity comparisons, set the cap to zero for **all** compared arms.
- The online track updates from prior test-stream episodes by design; its results are labeled online.
- Memory stores are category-specific in the primary track. Cross-category transfer is a separate experiment.
- The environment is sequential (parallelism one) so an episode's update is available only to later episodes. Each environment is explicitly closed in a finally block.
- The runner rejects an existing output directory. Upstream's cached per-episode JSON cannot safely resume a memory-learning run because a cached episode is skipped without replaying its memory update.
- Any missing, malformed, or errored episode invalidates that condition. Do not treat failed setup as zero success.

## Statistical analysis

Use paired differences against no memory on identical task IDs within each seed and category. Report raw numerators/denominators and category-level effects. For online streams, earlier outcomes alter later memory; ordinary independent-episode confidence intervals are overconfident. Use category/order-block resampling and multiple task-order seeds. With one seed, intervals are descriptive only.

Held-out runs should report variation across memory-building/order seeds. For transfer, report each of the 30 source→target pairs and account for repeated target items when summarizing uncertainty.

## Cost and stopping

Dry-run prints task episodes and an upper bound on worker calls (episodes × max rounds). Memory writer and embedding calls are additional and must be budgeted separately. Run smoke → competence → core → upstream systems → original online → transfer. Do not launch the full plan until the small gates and all selected adapter smoke tests pass.

## Reproducibility record

Every run stores repository and pinned-upstream commit hashes, worker/memory model identifiers, temperature, context cap, seeds, task IDs, mapping hashes, method configuration, selected evolutionary policy, raw per-episode JSON and conversations, logs, and completion validation. API keys are excluded. A run is complete only when every expected episode exists and passes validation.
