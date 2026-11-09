# Registered experimental protocol

**Question:** Does persistent external memory improve future ALFWorld task success conditional on sufficient worker competence?

**Official evaluation sets:** 2,420 train mapping items, 200 test mapping items, six task types. Official tasks, reward, evaluator, and action budget are not redefined.

**Tracks:**
- `heldout`: memory may update only on the *official train* mapping. It is frozen during all official test tasks. Train/test item IDs are disjoint. In this track, the first test task may already benefit from pretraining memory.
- `online`: sequential official test stream, with update after each episode. Matches upstream self-evolution setup, but the test stream is also the experience source.
- `transfer`: online source-category bank followed by five frozen cross-category test runs per source. The target task IDs recur under multiple source memories; they are not independent datasets.

**Primary outcome:** final reward >=1.0. Secondary: mean reward, worker API tokens, memory API tokens (where available), rounds, latency, category results, and learning curves.

**Competence flags (specified before full experiments):** under 55% no-memory overall, under 30% for a task category, or >95% no-memory ceiling by category. Playbook must be checked separately; absence of playbook gain is inconclusive about a true oracle. If a weak worker cannot solve tasks, do not attribute failures to memory.

**Paired inference:** same test IDs and seed/order per method, compare against that seed's no-memory control. For online streams, memory-induced dependence means independent-episode confidence intervals are overconfident; use category-block bootstrap and additional order seeds. With one seed intervals are descriptive only. For heldout, also repeat independent memory-building streams; seed label includes sampling/order.

**Leakage audit:** memory stores are unique to condition/seed/source. In heldout, test task IDs must never enter a writable memory store; in transfer, frozen source store must not be updated on the target. No full-test optimization of memory-generator prompts or retrieval settings. Train and test mapping game IDs are distinct, but underlying scene/object structures can recur; this is related-distribution transfer, not entirely unseen-world generalization.

**Oracle limitation:** static playbook is a positive control for procedural guidance, not a gold memory derived from previously hidden information. ALFWorld may reward procedural reuse without making historical memory necessary. Strong claims of "learning from experience" need heldout learning-curve improvements plus domain-aware causal diagnostics.

**Original comparability:** seed 0 preserves ordered official test IDs and upstream memory adapters. For published-method fidelity set `memory_cap_tokens=0` (no context cap) and matched worker across no-memory and memory. The controlled track has a 2048-token memory-only cap; it is **not** an exact published-condition reproduction. Different worker or provider is also a reproduction extension. We never present those as the authors' original results.

**Stopping rule:** after a small smoke and competence pilot, do not spend on A-MEM/ReasoningBank/ACE/full transfer if model competency is inadequate or memory-relevant headroom is unsubstantiated. Negative pilot result should be reported as limited evidence, not erased.

**Costs:** dry-run prints source and target episodes and an upper bound on agent calls; generator/embedding calls can exceed that and must be budgeted separately. All outputs include complete configurations, credentials excluded.
