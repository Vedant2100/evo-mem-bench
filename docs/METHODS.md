# Method definitions, implementation status, and interpretation boundaries

This table describes the actual interventions implemented in this repository. Names containing “style” or “inspired” are deliberate adaptations, not claims of exact reproduction of a commercial product or every detail of a paper.

## ALFWorld arms

| Arm | Implementation | What it tests | Important boundary |
|---|---|---|---|
| none | Official upstream no-memory mode | Worker competence | Matched baseline |
| playbook | Fixed category-specific procedure | Whether generic procedural guidance helps | Not an oracle; no experience is learned |
| random | Randomly ordered prior episodes | Context quantity/noise control | Query-seeded ordering is deterministic |
| recent | Most recent prior episodes | Recency heuristic | Same memory-token cap |
| full_history | All prior episodes, newest first | Unfiltered history under a bounded context | Truncated only at the memory suffix |
| bm25 | Official upstream BM25 adapter | Sparse lexical retrieval | Original upstream chunking/retrieval |
| success_bm25 | Controlled adapter; only reward-successful episodes stored | Whether failed experiences harm retrieval | Shares upstream BM25 chunking/tokenization |
| outcome_bm25 | Controlled adapter; stores success and failure labels | Whether explicit outcome labels help | Same experience source as success-only |
| bandit | Non-contextual UCB over BM25/recent/random | Online retrieval-arm selection | Not contextual RL or actor-critic |
| contextual_bandit | LinUCB over BM25/recent/random; context is task-family one-hot plus task-verb features | Whether task context improves retrieval policy | Online policy update uses the completed episode reward |
| supervised_writer | Local LLM creates a short lesson only from reward-verified successful trajectories | Reward-conditioned memory writing | Writer is prompted, not fine-tuned; no lesson is written for failed episodes |
| reflexion | Local LLM reflects on each completed success/failure | Episodic self-reflection | Reflexion-inspired prompt adaptation, not a reproduction of every Reflexion paper detail |
| expel | Local LLM extracts a standalone general principle and carries prior principles forward | Experience-to-principle abstraction | ExpeL-inspired prompt adaptation |
| letta_consolidation | Episodic JSONL buffer; every N episodes a local LLM consolidates pending episodes into a bounded semantic note list | Sleep-time consolidation vs raw episodic retrieval | Letta-inspired; not the Letta service or its complete memory architecture |
| bayesian_state | Beta(1,1) prior per inferred ALFWorld task family; update alpha/beta from binary environment reward; retain a best successful action trace | Whether explicit uncertainty/state summaries help | A category-level Bayesian success model, not a full belief-state/POMDP tracker |
| evolutionary | Outer genetic search over retrieval mode and top-k; memory built on train subset, candidate policies evaluated read-only on a disjoint validation subset, best frozen on test | Validation-driven memory retrieval-policy optimization | Does not evolve note content, train worker weights, or inspect test outcomes |

### Custom writer accounting

For supervised_writer, reflexion, expel, and letta_consolidation, memory-writer calls use the configured local OpenAI-compatible memory endpoint, defaulting to the same model as the worker. The runner records memory LLM input/output token counts and latency when the endpoint reports usage. Retrieval itself uses lexical BM25 over stored notes for the first three methods; Letta-style consolidation retrieves from its semantic-note list using the same BM25 helper. This is a system comparison, not a pure writer-only or retriever-only ablation.

### Contextual bandit details

The contextual feature vector contains a bias, six inferred task-family indicators, and six task-verb indicators. LinUCB maintains one A matrix and b vector per retrieval arm. At task start it selects BM25, recent, or random; after the task completes it updates the chosen arm from binary success. In held-out evaluation the policy is frozen because the memory adapter is read-only. In the original online protocol it updates sequentially from earlier test-stream outcomes, matching the online nature of that protocol.

### Evolutionary search details

The default population is four candidate policies over BM25, success-BM25, outcome-BM25, recent, and full-history retrieval, with top-k in {1,3,5}; there are two generations. Fitness is validation success rate, with lower injected context tokens as a tie-break. All candidates read the same memory bank built only from the train subset. The validation item IDs are stored in the run manifest; the selected policy and all candidate results are saved under each seed's evolutionary folder. The final test phase loads the selected configuration read-only. This is evolutionary **policy** search, not evolution of memory content.

## Official upstream systems

| Arm | Upstream adapter | Configuration / runtime |
|---|---|---|
| amem | A-MEM | Local memory LLM plus local embedding endpoint; high evolution threshold (99999) |
| amem_evolving | A-MEM | Same, threshold 10; explicit lower-threshold extension |
| reasoning_bank | ReasoningBank | Local memory LLM for judge/extraction; local embedding endpoint |
| ace | ACE | Local memory LLM for reflection and curation |
| agent_kb | AgentKB | Local memory LLM and local embedding endpoint |
| awm | Agent Workflow Memory | Local memory LLM and local embedding endpoint |
| mem0 | Mem0 | Local OpenAI-compatible LLM and embedding endpoint; Chroma persistence |
| skillweaver | SkillWeaver | Local memory LLM; generated skills stored in the per-condition directory |
| lightweight_memory | Lightweight Memory | Local memory LLM; strategic/operational store per condition |

The official adapters are kept in the pinned upstream checkout. scripts/patch_upstream.py adds local OpenAI-compatible LLM support to adapters that otherwise require ARK batch calls, points AgentKB/AWM to the configured local embedding model, and instruments environment cleanup. Method-specific packages are installed with scripts/install_memory_methods.sh. Each optional adapter still needs a smoke test before its result is accepted.

## Separate tracks

- **Hidden-rule oracle:** available only in the companion synthetic diagnostic. Each station has an independently generated hidden rule revealed in an earlier discovery episode; oracle supplies the exact rule. ALFWorld has no equivalent gold-memory source.
- **Continual LoRA/SFT:** separate worker-weight intervention. It collects official train trajectories, fine-tunes a LoRA adapter, and evaluates the adapted worker with no external memory. Never merge its score into the memory-backend table.
- **Actor-critic memory RL:** not implemented. Contextual LinUCB is the current adaptive governor and must not be renamed actor-critic or policy-gradient RL.
- **GraphRAG / MemoryOS / MemOS / other upstream factory adapters:** upstream may expose these, but they are not in the registered full matrix until they have explicit configs, dependencies, and a successful smoke test.

## Fairness and cost

- The same worker checkpoint, system prompt, decoding temperature, action-round limit, task IDs, and seed/order are used across paired arms.
- The memory cap is applied only to the injected memory suffix, never the original task instructions. Controlled plans default to 2048 memory tokens. For original-fidelity uncapped comparisons set the cap to zero for every compared arm.
- Held-out training and validation tasks are disjoint from the official test set. Candidate selection is validation-only.
- The online protocol intentionally learns from earlier test-stream episodes; report it as online adaptation, not independent held-out generalization.
- Separate worker tokens, memory-writer tokens, embedding tokens, and latency. Some upstream adapters estimate token counts while others report provider usage; identify estimated values and do not treat them as directly comparable provider billing.
- A method that fails setup, API calls, or episode validation is a failed condition, not a zero-success method.
