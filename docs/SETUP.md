# Setup

## 1. Clone and pin the official benchmark

    git clone https://github.com/Vedant2100/evo-mem-bench.git
    cd evo-mem-bench
    bash scripts/bootstrap_upstream.sh

The bootstrap script checks out the exact upstream commit recorded in configs/experiment.json, applies the strict idempotent patcher, copies the controlled adapter into the official memory factory, and compiles the evaluator. Do not manually edit the vendored upstream checkout; change scripts/patch_upstream.py and rerun bootstrap instead.

## 2. Evaluator environment

Use Python 3.10 or newer for the experiment runner/evaluator:

    python -m venv .venv-eval
    source .venv-eval/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -e '.[eval,test]'
    python -m pip install -e vendor/EvoMemBench/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB/agentenv
    python -m pytest -q

On a managed cluster, use the cluster's supported Conda or Apptainer workflow instead of creating a venv if required by policy.

## 3. ALFWorld server environment

The upstream game server uses its own Python environment. Follow the official ALFWorld setup from the pinned checkout:

    conda create -n evomem-server python=3.9 -y
    conda activate evomem-server
    cd vendor/EvoMemBench/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB/agentenv-alfworld
    bash setup.sh

The official setup downloads ALFWorld assets and can use substantial disk space. Run it once on persistent storage. Then return to the repository root and start the server:

    cd /path/to/evo-mem-bench
    bash scripts/start_server.sh

The default server port is 36005. Keep the server alive in the same allocation/node as the evaluator. The runner uses sequential episode execution and explicitly closes each finished environment.

## 4. Local model endpoints

The runner uses the official OpenAI-compatible worker interface. A local vLLM server is supported.

Set the worker variables in .env:

    OPENAI_BASE_URL=http://127.0.0.1:8000/v1
    OPENAI_API_KEY=local
    WORKER_MODEL=YOUR_CHAT_MODEL_ID
    WORKER_TEMPERATURE=0

Memory-writer arms default to the same local model:

    EVO_MEM_LLM_BASE_URL=http://127.0.0.1:8000/v1
    EVO_MEM_LLM_API_KEY=local
    EVO_MEM_LLM_MODEL=YOUR_CHAT_MODEL_ID

Embedding-backed upstream systems use a separate OpenAI-compatible embedding endpoint:

    DASHSCOPE_BASE_URL=http://127.0.0.1:8001/v1
    DASHSCOPE_API_KEY=local
    EVO_MEM_EMBED_MODEL=YOUR_EMBEDDING_MODEL_ID

The variable names DASHSCOPE_* are retained because the upstream adapters read them; the endpoint may be local. They do not imply that a DashScope account is required.

For the custom core plan, only the chat endpoint is needed. For A-MEM, ReasoningBank, AgentKB, AWM, and Mem0, start the embedding endpoint too.

## 5. Optional upstream method packages

Install only the systems selected in your plan:

    bash scripts/install_memory_methods.sh amem
    bash scripts/install_memory_methods.sh agent_kb awm
    bash scripts/install_memory_methods.sh mem0

Or install all optional packages:

    bash scripts/install_memory_methods.sh all

ACE, ReasoningBank, SkillWeaver, and Lightweight Memory use the pinned upstream source and do not need a separate provider API key when local endpoints are configured. A-MEM, AgentKB, AWM, and Mem0 also use the local endpoints configured above.

## 6. Verify before running

    python scripts/prepare_manifest.py --validate-only
    python scripts/run_experiment.py --plan smoke --dry-run
    python scripts/run_experiment.py --plan competence --dry-run

The runner checks the pinned upstream commit, patch version, task maps, method packages, model endpoints, and ALFWorld server port. Dry-run never invokes inference.

Run a smoke test before any larger plan:

    python scripts/run_experiment.py --plan smoke

Then run the competence gate. If the worker is too weak or already at ceiling, stop and reconsider before running every memory method.

## 7. Analyze

    python scripts/analyze_results.py runs/RUN_ID

Each run directory is unique and immutable. The runner refuses to reuse it. Upstream episode caching is not used to resume online memory streams because skipping an episode does not replay its memory update.
