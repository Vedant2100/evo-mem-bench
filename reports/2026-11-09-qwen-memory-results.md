# 2026-11-09: Verified Qwen GPU memory-necessity pilot

## Actual GPU experiment (completed)

- Inference: Qwen/Qwen2.5-7B-Instruct, 4-bit NF4, deterministic greedy decoding.
- Compute: one Modal T4, using existing GitHub Actions credentials. No pay-per-token inference API.
- Benchmark: synthetic hidden-rule transfer **positive control** from this repository, not original ALFWorld tasks.
- 3 independent seeds (11, 22, 33), 12 held-out downstream parcels per seed, no worker fine-tuning.
- [GitHub Actions run: success](https://github.com/Vedant2100/role-work-forge-artifacts/actions/runs/42586773349)
- [Corrected analysis and original raw predictions](https://github.com/Vedant2100/role-work-forge-artifacts/tree/main/hackathons/2026-11/results/evo-memory-modal-42586773349)

On all 36 downstream questions: no-memory 8/36 (22.2%), true prior-discovery oracle 36/36 (100%). 28 of 36 previously-unsolved cases were rescued by the oracle.

The preset gate required no-memory <=40% and oracle >=80%, with a >=40 percentage-point gap. Seed 33 had no-memory 5/12 (41.7%) and oracle 12/12. Its non-oracle methods were **not run**. The following methods were compared on the same 24 questions from seeds 11 and 22:

| Intervention | Success | Paired delta versus 3/24 no-memory |
|---|---:|---:|
| No memory | 3/24 = 12.5% | 0.0 pp |
| Unrelated random history | 4/24 = 16.7% | +4.2 pp |
| Bounded full recent history | 10/24 = 41.7% | +29.2 pp |
| Non-contextual UCB retrieval controller | 15/24 = 62.5% | +50.0 pp |
| Reward-labelled BM25 | 23/24 = 95.8% | +83.3 pp |
| BM25 over raw records | 24/24 = 100% | +87.5 pp |
| Success-only BM25 | 24/24 = 100% | +87.5 pp |
| Locally written concise memory + BM25 | 24/24 = 100% | +87.5 pp |

### Mechanistic observations

1. Retrieval relevance dominated context quantity. BM25 identified the matching station/attribute and recovered the correct hidden rule; a short latest-history window often omitted the relevant episode.
2. The local Qwen memory writer preserved necessary routing facts while reducing injected text from ~417 to ~160 characters per query relative to raw BM25 (approximately 62% fewer characters), with the same task success.
3. The UCB controller sometimes chose an irrelevant station's history or only invalid/rejected routes; non-contextual feedback was not enough to match BM25.
4. A reward-labelled memory response returned invalid code PIOV instead of PIVO despite containing the correct verified rule. Thus exposure and retrieval do not guarantee application; strict objective parsing revealed the failure.

### Statistical and reporting limits

- This is single-turn synthetic fact lookup/transfer. It establishes that the worker can exploit **earlier hidden information when retrieved**, not that memory improves multi-step agent performance, model weights, or general continual learning.
- The small 12-question-per-seed cutoff may be unstable; seed 33 was excluded from the extended methods by the predefined threshold even though oracle rescued 7/12 cases.
- Original Modal-generated per-method aggregate delta incorrectly used all 36 baseline questions rather than the 24 matched ones. The linked corrected GitHub summary/findings fixed this without modifying raw predictions or rerunning GPU inference.
- Differences between 24/24 methods are not statistically resolvable from these few trials. Evaluate harder memories before ranking those methods.

## Official original EvoMemBench infrastructure (completed)

Pinned DSAIL-Memory/EvoMemBench commit 3efc221fd8d5f4de7c2dc2022357e600d26325c7, original ALFWorld server and game data.

The first free GitHub Actions setup trial failed because the original setup.sh concealed a Python dependency install problem. The follow-up installer uses fail-fast Python 3.10 setup and succeeded.

- [Official CPU environment preflight: success](https://github.com/Vedant2100/evo-mem-bench/actions/runs/42587847627)
- All 2,420 original train game files found in json_2.1.1/train.
- All 200 original test game files found in json_2.1.1/valid_train.
- Official test task item 2420 reset successfully, exposing 26 admissible actions.
- This preflight **did not run model inference**.

## Multi-step ALFWorld model pilot

A separate bounded experimental workflow connects the original GitHub Actions CPU ALFWorld environment with a locally served Qwen2.5-7B on Modal T4 via remote method calls, using the original eval_one evaluator and original reward. Planned comparison: no-memory, fixed playbook, and BM25; 3 categories x 2 test tasks x 3 methods, max 12 action rounds.

- [Multi-step ALFWorld experiment workflow](https://github.com/Vedant2100/role-work-forge-artifacts/actions/runs/42589124605)

Scores remain **unverified here** pending the run's persisted results. This is a capability pilot and must not be represented as the full 200-task, 20-round, 30-transfer-pair original-paper comparison.

## Remaining scientific work

Strong enough ALFWorld worker, full held-out and online evaluation, multiple memory-building seeds, all 30 transfer pairs, original A-MEM/ACE/ReasoningBank writer backends, and genuinely learned contextual/RL/evolutionary memory methods remain to be tested or implemented. Expanding from this pilot without gates risks spending compute on base-model incompetence rather than a memory hypothesis.
