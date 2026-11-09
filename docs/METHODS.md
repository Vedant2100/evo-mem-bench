# Method coverage and scientific boundaries

## Implemented in this repository

| Condition | Source | Intervention |
|---|---|---|
| none | Upstream `--no_memory` | Frozen worker, no state |
| playbook | Local control | Hand-written category-level instructions; **not oracle memory** |
| random | Local control | Random previous experience |
| recent | Local control | Most recent previous experience |
| bm25 | Upstream | Original BM25 chunk retrieval |
| success_bm25 | Local, shares upstream format/chunks | Only successful prior episodes |
| outcome_bm25 | Local, shares upstream format/chunks | Includes success/failure label |
| bandit | Local | Reward-adaptive UCB choice of BM25/recent/random, non-contextual |
| amem | Upstream | A-MEM note storage/extraction; original high threshold |
| amem_evolving | Upstream | A-MEM with threshold 10; extension, not published setting |
| reasoning_bank | Upstream | ReasoningBank extraction/retrieval |
| ace | Upstream | ACE reflection/playbook updates |

Native upstream factory additionally supports Mem0, Qwen embeddings, GraphRAG, AgentKB, AWM, LightweightMemory, SkillWeaver, MemoryOS, MemOS. The current runner does **not** claim these are validated; they require method-specific configs, dependencies, and smoke tests.

## Not implemented (must not be reported as results)

- **Supervised memory writer:** requires good-memory targets and a separate writer training split.
- **Evolutionary memory optimization:** needs candidate proposals and costly downstream training-task evaluations, never test-task selection.
- **Contextual bandit:** present UCB policy has no task-specific features; not equivalent.
- **Actor-critic / memory-native RL:** needs explicit state, action, reward and sufficient trajectories for training.
- **Continual LoRA/SFT:** changes worker weights and is a separate experimental arm, not an external memory backend.
- **Bayesian state memory:** requires a grounded probabilistic state schema; a text summary is not Bayesian inference.
- **Dream/sleep-time consolidation:** requires a bounded offline consolidation schedule and distinct compute accounting.
- **Procedural skills:** ACE and upstream SkillWeaver are relevant existing implementations; do not call BM25 a skill learner.

These extensions belong after the competence and memory-sensitivity gate. Adding unvalidated "RL" labels would be worse than a focused reproducible experiment.

## Fairness

The worker model/prompt/decoding/rounds and task IDs are held fixed. Different memory generators use different native model calls and embeddings; this is a **system-level comparison**, not a pure test of retrieval optimization. Report writer tokens, embedding tokens, worker tokens, and latency separately. Do not combine estimated adapter token counts with provider-billed totals.
