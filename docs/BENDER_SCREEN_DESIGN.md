# Bender ALFWorld memory screen: design (written before any GPU run)

Status: exploratory screen, not a strict-official replication. Goal: find where memory helps; do not promise it does.

## Setup (bender, 2x A30 24GB)
- Worker: frozen local model behind vLLM, thinking OFF, temperature 0. Start with Qwen3-8B. 20-task no-memory pilot; keep only if success is 20-70%, else Qwen3-32B-AWQ on one A30. Competence gate below still applies.
- Own dirs only: /var/tmp/vbork001/evomem (venvs, ALFWorld data, pip cache). Ports: vLLM 8100, ALFWorld 36105 (not defaults).
- Never touch /var/tmp/vbork001/fm-bender or ~/fm-bender (second-machine study).

## Stages (stop and report after each)
1. smoke (36 episodes): validates stack, measures s/episode.
2. competence gate: none + playbook, 216 episodes. Stop and rethink if no-memory success <55% or >95%, or any category <30%.
3. screen: 1 seed, 10 train + 10 test tasks per category (~1,100 episodes). Methods: none, playbook, recent, bm25, outcome_bm25, reflexion, expel, letta_consolidation, supervised_writer, bayesian_state.
4. confirm: top 2-3 methods, more seeds and test tasks.

## Pre-registered bar for "clear effect" (fixed before the screen)
- >= +15 points over none AND >= +10 points over the static-tips control, paired McNemar p < 0.01, in >= 4 of 6 task types.
- 60 test tasks per method detects only ~10-point gaps: screen results are "promising, to be confirmed", not a ranking.

## Design-gap resolutions (user decision: fix them, then start)
- Splits: use the repo's own per-category train/test splits (not valid_seen/valid_unseen). Stated deviation; results are not comparable to published valid_unseen numbers.
- Static tips control: the repo's `playbook` arm (fixed per-category procedure, no learning) is the static-tips control. Its length is fixed, not tuned to the 2048-token memory cap; report its token count next to each memory arm's.
- Thinking off via vLLM `--default-chat-template-kwargs '{"enable_thinking": false}'` (the client has no thinking switch).
- Writer arms are prompted, not trained; claims limited accordingly.

## Environment status
Eval venv (34 tests pass), ALFWorld server venv + 5,234 games, vLLM 0.31.0 (cu130, driver supports CUDA 13.3) installed. Disk is tight: / ~54 GB free and shared.

## Amendment 1 (2026-11-10, user decision): max_rounds 20 -> 40
Reason: Qwen3-8B and Qwen3-32B-AWQ both scored ~33-35% no-memory in the competence gate; pick_heat/pick_cool/pick_clean hit the 20-round cap while still searching receptacles one by one (confirmed by reading a pick_heat trajectory). Raising the cap changes the protocol: results are not comparable to the upstream 20-round setting. vLLM max-model-len raised 16384 -> 32768 to fit longer episodes. The 55-95% gate is unchanged; if it fails again the chain stops. 20-round gate results (8B, 32B) are kept as records.

## Amendment 2 (2026-11-10 12:15, written BEFORE gate/screen results): deadline-sized screen
User needs a verdict by 15:30 on whether RL-trained memory writers are worth pursuing. The 10-method screen cannot finish by then, so:
- Reduced screen, 1 seed, 6 train + 6 test tasks per category. Stream A: none, playbook, bm25, reflexion. Stream B: none, letta_consolidation, supervised_writer. (none is in both streams as a paired baseline.)
- Competence floor for this run lowered from 55% to 40% no-memory overall (ceiling stays 95%), because the question is headroom for memory, not worker competence. The 55% floor is NOT met if no-memory is 40-55%; the write-up must say so.
- Power: 36 test tasks per method detects only very large gaps (~25+ points). Output is a headroom/sensitivity assessment, not a ranking, and does NOT test RL-trained memory directly.
- Pre-registered bar from the design (+15 over none, +10 over playbook, McNemar p<0.01, 4 of 6 types) cannot be met at this sample size; it is retained for the later confirm stage.

## Amendment 3 (2026-11-10 12:45, before results): add RL-style arms
User pointed out the reduced screen (Amendment 2) omitted the repo's reinforcement-learning-style methods. Added Stream C: none, bandit (UCB over bm25/recent/random), contextual_bandit (LinUCB), same 6 train + 6 test per category, seed 0, so none is paired with the same tasks. Not included: `evolutionary` (needs a validation search; too slow for the 15:30 deadline). Caveat: with only 36 training episodes a bandit gets few updates, so a null result is weak evidence against bandits; it is not a test of RL-trained weights (the LoRA track is behavioral cloning, also not run).

## Amendment 4 (2026-11-10 16:10, before results)
User asked to run the original memory matrix. Running the original stage-3 screen as designed: 10 train + 10 test per category, seed 0, 40 rounds, Qwen3-32B-AWQ. Three streams, `none` paired in each: A = playbook, recent, bm25; B = outcome_bm25, reflexion, expel; C = letta_consolidation, supervised_writer, bayesian_state. Not included: bandit (already run at 6+6, could be added later), evolutionary (needs validation search), upstream adapters (A-MEM, ReasoningBank, ACE, AgentKB, AWM, Mem0, SkillWeaver, Lightweight Memory). Chain: `chain/chain7.sh`.

## Amendment 5 (2026-11-11)
After the full 10-method screen, ran short smoke checks (1 task per category) of the remaining registered methods: evolutionary, ace, skillweaver, lightweight_memory (chain8), then reasoning_bank, agent_kb, awm, amem, amem_evolving, mem0 (chain10/11). Embedding-backed methods used a CPU-only OpenAI-compatible embedding service (`embed_server.py`, Qwen3-Embedding-0.6B, port 8001, no GPU). amem/amem_evolving/mem0 first failed because the system sqlite3 is older than ChromaDB's 3.35 minimum; fixed inside `.venv-eval` with `pysqlite3-binary` plus a `sitecustomize.py` shim; no system changes. `scripts/install_memory_methods.sh` has an unexpanded `\${BASH_SOURCE[0]}` escape bug, so the same pip commands were run by hand. All ten smokes pass; they are too small for success-rate comparisons.
