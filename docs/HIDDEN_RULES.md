# Hidden-rule memory-necessity diagnostic

The full ALFWorld benchmark does not guarantee that history is *necessary* for success. Many ALFWorld tasks can be solved from ordinary knowledge and the current observation. A null memory effect is hard to interpret when the worker is weak or history is irrelevant.

This small companion diagnostic creates many independent fictional warehouse stations, each with a randomly assigned persistent routing rule. The agent sees the rule in an earlier inspection; a new parcel at that station later requires applying it. The present question contains the station and tag, but not the station's hidden gate. There are six balanced gate possibilities, so an unaided guess has approximately 1/6 chance.

Each station's discovery log includes a rejected wrong attempt followed by a verified correct rule. Different stations use independently generated identifiers and different train/test parcels. No future test prompt or answer is shown to any memory writer. All conditions use one unchanged worker prompt template, output parser, model, decoding and evaluation order.

## Methods

- none: worker sees only current question
- oracle: precisely the correct previously observed rule (positive control)
- random: only unrelated stations' experience
- full_history: latest history, truncated to a fixed character budget
- bm25: top similar prior records (including rejected attempts)
- success_only: BM25 over verified records only
- reward_labeled: BM25 over all records with VERIFIED/REJECTED labels
- ucb: online policy choosing BM25/recent/random retrieval, updated from test outcome
- writer: same local LLM distills verified discoveries into concise memory, then BM25 retrieves the result

UCB updates its *retrieval policy* from previous evaluation feedback; it does not add held-out test outcomes to the memory bank. Do not compare its adaptive online test stream as though it were a frozen policy.

This is a **single-turn diagnostic positive control**, not a scientific replacement for EvoMemBench or a general tool-use/embodied benchmark. It isolates memory-necessity, memory extraction/retrieval and worker utilization. Full-scale claims must still come from a capable agent in an objective multi-step environment.

## Run using local Qwen

Install optional OpenAI client and offline tests:

~~~bash
pip install -e ".[eval,test]"
pytest -q
~~~

Serve a sufficiently capable Qwen model through an OpenAI-compatible local endpoint. The model ID must match your server.

~~~bash
python -m evomembench.experiment --dry-run
python -m evomembench.experiment \
  --model Qwen/Qwen3-32B \
  --base-url http://127.0.0.1:8000/v1 \
  --worlds 36 --seeds 11,22,33 --gate \
  --output runs/hidden_rules_01
~~~

With --gate, no-memory and oracle run first. If oracle is below 80%, no-memory above 40%, or their difference below 40 percentage points, the remaining memory methods are skipped for that seed. Those thresholds are fixed diagnostics, not preregistered hypothesis-test significance thresholds.

To test all nine conditions, omit --gate.

Every run stores raw responses, exact discovery/test manifests, memory injected, model token usage and outcomes. It refuses to reuse an output directory, avoiding the state contamination found in the earlier BFCL pilot.

## Interpreting the outcome

- Oracle fails: model or prompting is not reliably capable of using the right remembered rule. Do not blame retrieval or the memory writer.
- Oracle succeeds, raw history fails: storage/injection noise or retrieval selection is plausible.
- BM25 succeeds, writer fails: suspect extraction/compression.
- Success-only beats outcome-labelled memory: learning from rejected trajectories is adding confusion.
- No-memory succeeds near oracle: the task doesn't establish memory necessity, so do not use it as a memory benchmark.
- All retrieval methods succeed: the controlled positive test is healthy; proceed to the harder original ALFWorld/held-out experiments.

The model is frozen. There is no actor-critic, RL fine-tuning, or parameter update in this diagnostic.
