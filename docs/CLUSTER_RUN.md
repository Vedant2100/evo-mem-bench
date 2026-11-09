# Run the experiment on a GPU cluster with Codex

This is a runbook, not a submitted job. It assumes a Slurm-style cluster and local GPU inference. Do not guess a partition name or GPU resource flag: use the cluster's current documentation and allocation policy.

## 1. Allocate one GPU node

For an interactive allocation, request one GPU, enough CPU/RAM for the ALFWorld server, and a long enough walltime. Example shape only:

    salloc --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=12:00:00

If the cluster requires a partition, account, or a different GPU flag, add the values from its current docs. Keep the worker, embedding service, ALFWorld server, and evaluator on the same node unless the cluster provides a reliable shared network route.

## 2. Clone and create environments

Keep the repository and ALFWorld downloaded assets on persistent storage, not a short-lived scratch directory.

    git clone https://github.com/Vedant2100/evo-mem-bench.git
    cd evo-mem-bench
    bash scripts/bootstrap_upstream.sh

Create/activate the evaluator environment and install the repository:

    python -m venv .venv-eval
    source .venv-eval/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -e '.[eval,test]'
    python -m pip install -e vendor/EvoMemBench/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB/agentenv
    python -m pytest -q

Create the separate upstream ALFWorld server environment following docs/SETUP.md. Download the official game assets once to persistent storage.

If the cluster requires Apptainer rather than Docker, use the cluster-approved Apptainer image/overlay for these environments. No Docker or Podman daemon is required by this repository.

## 3. Start local GPU inference

The worker endpoint is OpenAI-compatible. The example below is for vLLM; select a model/quantization that actually fits the allocated GPU memory. Do not assume a 32B checkpoint fits unquantized.

    export WORKER_MODEL=YOUR_LOCAL_CHAT_MODEL_ID
    export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
    export OPENAI_API_KEY=local
    export EVO_MEM_LLM_BASE_URL=http://127.0.0.1:8000/v1
    export EVO_MEM_LLM_MODEL=$WORKER_MODEL
    export EVO_MEM_LLM_API_KEY=local
    export WORKER_TEMPERATURE=0

Start the chat model in a separate terminal within the same allocation:

    vllm serve "$WORKER_MODEL" \
      --host 127.0.0.1 --port 8000 \
      --dtype auto --max-model-len 32768 \
      --gpu-memory-utilization 0.78

For the systems plan, start a separate embedding model service. Qwen3-Embedding is the default; use an embedding checkpoint supported by your installed vLLM version:

    export EVO_MEM_EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B
    export DASHSCOPE_BASE_URL=http://127.0.0.1:8001/v1
    export DASHSCOPE_API_KEY=local

    vllm serve "$EVO_MEM_EMBED_MODEL" \
      --task embed --host 127.0.0.1 --port 8001 \
      --gpu-memory-utilization 0.15

The two GPU memory utilization values must fit the actual device and KV-cache needs. If the embedding service cannot coexist with the chat model, run the core plan first (it does not require embeddings), then allocate a separate run/node for embedding-backed systems. Do not silently switch embedding providers between arms.

For an external OpenAI-compatible embedding endpoint, set DASHSCOPE_BASE_URL, DASHSCOPE_API_KEY, and EVO_MEM_EMBED_MODEL to the corresponding endpoint/model. The DASHSCOPE variable names are retained for compatibility with the upstream adapters; they do not require DashScope specifically.

## 4. Start ALFWorld server

In a terminal using the server environment:

    conda activate evomem-server
    cd /path/to/evo-mem-bench
    bash scripts/start_server.sh

Keep this terminal alive. The runner checks the default port 36005 before starting. The patched evaluator closes each completed environment in a finally block; do not remove this cleanup patch.

## 5. Configure and preflight

In the evaluator environment, from the repository root:

    cp .env.example .env
    # Edit WORKER_MODEL and the endpoint/model IDs to match your servers.
    python scripts/prepare_manifest.py --validate-only
    python scripts/run_experiment.py --plan smoke --dry-run
    python scripts/run_experiment.py --plan competence --dry-run
    python scripts/run_experiment.py --plan core --dry-run

For upstream systems:

    bash scripts/install_memory_methods.sh amem agent_kb awm mem0
    python scripts/run_experiment.py --plan systems --dry-run

The runner checks the local chat and embedding endpoints, the ALFWorld port, optional Python packages, the pinned upstream commit, and patch version. A dry-run does not call the model.

## 6. Run in stages

Recommended order:

    python scripts/run_experiment.py --plan smoke
    python scripts/run_experiment.py --plan competence
    python scripts/run_experiment.py --plan core
    python scripts/run_experiment.py --plan systems
    python scripts/run_experiment.py --plan original
    python scripts/run_experiment.py --plan transfer

The full plan combines core and upstream arms and is expensive. Review the dry-run episode counts and worker-call upper bound before starting it. Writer and embedding calls are additional.

**Important:** the default `smoke` plan checks only the no-memory, playbook, random, and BM25 paths. Before the full run, smoke-test every selected adapter with a paired no-memory baseline. For the complete registered matrix, after installing optional packages and starting both local endpoints, run:

    for method in playbook random recent full_history bm25 success_bm25 outcome_bm25 bandit contextual_bandit supervised_writer reflexion expel letta_consolidation bayesian_state evolutionary amem amem_evolving reasoning_bank ace agent_kb awm mem0 skillweaver lightweight_memory; do
      python scripts/run_experiment.py --plan smoke --methods "none,$method" || exit 1
    done

Each command creates a separate small run directory. Do not proceed to the full plan if any adapter smoke fails; inspect its log/configuration and fix it first. The evolutionary smoke includes its validation-only candidate search.

Do not resume by reusing an old output directory. The runner intentionally refuses this because upstream's cached episode JSON does not replay the corresponding memory update. Start a fresh run with a fresh store.

## 7. Codex handoff prompt

Use this in Codex from the repository root after the allocation and endpoints are ready:

> Read README.md, docs/SCIENTIFIC_PROTOCOL.md, docs/METHODS.md, and docs/CLUSTER_RUN.md. Inspect the current git status and branch before changing anything. Verify the pinned upstream commit and patch version. Run pytest, compileall, manifest validation, and dry-runs for smoke and competence. Check that the worker, embedding endpoint (if selected), and ALFWorld server are reachable. Do not launch the full plan until I approve the exact dry-run counts. If tests fail, fix the implementation and add a regression test. Do not change task mappings, worker prompt, success definition, memory cap, or method list without explaining the scientific impact. When approved, run one stage at a time, preserve all raw episode files/logs, monitor the process, and report completed/failed episode counts and actual GPU/runtime use. Never treat a setup failure or missing episode as a zero-success result.

## 8. Results

    python scripts/analyze_results.py runs/RUN_ID

Copy the run directory to persistent storage after it completes. The manifest includes the repository/upstream commits, method settings, task IDs, seeds, and evolutionary validation selection. API keys are not stored.
