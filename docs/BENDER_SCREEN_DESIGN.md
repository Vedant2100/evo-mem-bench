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

## Known gaps vs the requested design (need a decision, not silently changed)
- The repo's runner uses its own per-category train/test splits, not valid_seen / valid_unseen (134 tasks).
- A static per-task-type tips control at equal token budget is not in the repo; needs implementing (playbook may be close, unverified).
- Writer arms are prompted (not RL-trained); claims limited accordingly.

## Environment status
Eval venv (34 tests pass), ALFWorld server venv + 5,234 games, vLLM 0.31.0 (cu130, driver supports CUDA 13.3) installed. Disk is tight: / ~54 GB free and shared.
