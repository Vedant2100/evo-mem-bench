# Handoff: evo-mem-bench ALFWorld memory screen

## Goal and why
- The user wants to show where memory models actually make a difference. Memory gave no gain in the earlier coding experiments, so they switched to ALFWorld, a setting where recurring task types should let experience transfer.
- They also want to test the memory methods from Letta's memory-models blog: consolidation, writers, reflection and similar.
- The deadline is the FORGE paper on Nov 15, 2026. Today is Nov 9.
- The user's latest steer is that the full matrix is too comprehensive. They want promising methods identified within about 12 hours.

## State (as of Nov 9)
- **Repo:** Vedant2100/evo-mem-bench, branch `codex/full-memory-arm-matrix`, tip `d039f63`.
  - It is cloned at `~/evo-mem-bench-eval/evo-mem-bench`.
  - Upstream is bootstrapped at pin `3efc221` and the patches applied.
  - Dry-runs for every plan work.
  - I did not run `pytest` because it isn't installed in the default Python. The reported CI was green with 33 tests.
- **Not done:** no GPU job has run. No environments, vLLM or ALFWorld server have been set up. Nothing has been sent to `bender-sonnet-agent`.
- **Plan sizes (from dry-run):**

| Plan       | Episodes | Worker-call upper bound |
| ---------- | -------- | ----------------------- |
| smoke      | 36       | 720                     |
| competence | 216      | 4,320                   |
| core       | 15,450   | 309,000                 |
| systems    | 10,200   | 204,000                 |
| full       | 24,090   | 481,800                 |
| original   | 13,800   | —                       |
| transfer   | 6,200    | —                       |

  Writer and embedding calls are extra in every plan.

## GPU situation
- **Cluster:** my job `fm-serial` (305865) holds the 2-GPU cap on nodevcggpu02 until about Nov 17. Two jobs are held (`fm-held`, 305010 and 305011).
- **Bender (2×A30, 24 GB each):** I saw both GPUs about 80% full with an Ollama `llama-server` run as user vbork001, 14 minutes old at the time. It is probably `bender-sonnet-agent`'s study. I did not touch it.
- **Bender gaps found:**
  - Bender has no conda or docker on PATH. Apptainer is available.
  - 40 CPUs and 251 GB RAM, `/scratch` 414 GB free.
  - Whether vLLM is installed there is unknown.
  - Bender's home is a separate mount, so repo, weights and assets must be copied over with `~/.cache/fm-bender-ssh.sh ssh|scp`.

## Recommended design (awaiting the user's go-ahead)
1. **Smoke:** validate the stack and measure seconds per episode.
2. **Competence gate:** `none` and `playbook`, 216 episodes. Stop and rethink if no-memory success is below 55% or above 95%, or if any category is below 30%.
3. **Screen:** 1 seed, 10 train and 10 test tasks per category (about 1,100 episodes in total).
   - Methods: `none`, `playbook`, `recent`, `bm25`, `outcome_bm25`, `reflexion`, `expel`, `letta_consolidation`, `supervised_writer`, `bayesian_state`.
   - Run 2–3 streams against one vLLM server to cut wall time.
   - The time estimate is a guess of 20–40 seconds per episode, to be replaced by the smoke measurement.
4. **Confirm later:** re-run the top 2–3 methods with more seeds and test tasks.
- **Dropped for now:** `evolutionary`, the bandits, all upstream adapters (A-MEM, Mem0, ACE, AgentKB, AWM and others), and the `original`, `transfer` and `full` plans.
- **Power caveat:** 60 test tasks per method only detects gaps of about 10 points or more. Report "promising, to be confirmed" and not a ranking.

## Open questions for the user
1. **GPUs:** the user leans toward bender, which means pausing or releasing the Ollama job there. Confirm that, and that I should message `bender-sonnet-agent` (via SendMessage) to ask it to free the A30s. Don't stop its job unilaterally.
2. **Worker model:**
   - `.env.example` defaults to Qwen3-32B, which needs about 64 GB in bf16.
   - It would need 4-bit AWQ across both A30s, with no room left for embeddings.
   - Choose a smaller model such as 14B or 8B. This is a methodology decision, so the user must sign off, and the competence gate tests whether it is capable enough.
3. **Qwen3 thinking mode:** on by default and not addressed in the docs. Decide on a setting and record it.
4. **Does the strict-official rule apply?** The earlier FrontierMemory rule says to match official setups except for a local model. This is a different study. Confirm whether that rule applies here or this is exploratory.
5. **Framing:** the paper should ask when memory helps. Don't promise that it does.

## Constraints from memory notes
- Never use more than 2 GPUs at once across all jobs, and check `squeue` before any GPU request. On the cluster, serialize with `--job-name=fm-serial --dependency=singleton`. Keep disk, CPU and memory modest.
- Bender is a separate machine. Coordinate with `bender-sonnet-agent` only through SendMessage, and never compare bender numbers with cluster numbers. For ssh and scp use `~/.cache/fm-bender-ssh.sh`, and never print the password.
- Fix Ollama stalls in exploratory runs yourself and report afterwards. Ask first only for counted or irreversible actions.
- The user sends outward-facing emails and public PRs personally. Don't add attribution lines to git commits or PRs.
- Run `docs/CLUSTER_RUN.md` stages in order. Do not reuse an old run directory. A setup failure or missing episode is never a zero-success result.

## Next steps for the new agent
1. Get the user's answers to the questions above, mainly GPUs and worker model.
2. Message `bender-sonnet-agent` and check vLLM and the Python environments on bender.
3. Copy the repo and weights to bender, then set up the evaluator environment, the ALFWorld server (Apptainer or conda) and vLLM.
4. Run smoke, then the competence gate, and report the numbers before launching the screen.