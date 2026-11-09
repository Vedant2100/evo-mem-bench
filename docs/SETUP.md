# Setup and execution

## Bootstrap
Run `bash scripts/bootstrap_upstream.sh`. The script clones the official EvoMemBench and pins commit `3efc221fd8d5f4de7c2dc2022357e600d26325c7`; it fails closed if the evaluator signature changes.

## ALFWorld server
Upstream recommends a separate Python 3.9 Conda environment:
```bash
conda create -n evomem-server python=3.9 -y
conda activate evomem-server
cd vendor/EvoMemBench/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB/agentenv-alfworld
bash setup.sh
cd /path/to/evo-mem-bench
bash scripts/start_server.sh
```
The server's default port is 36005. The upstream `setup.sh` downloads ALFWorld data; expect significant disk usage.

## Evaluator
A separate Python 3.10 environment:
```bash
conda create -n evomem-eval python=3.10 -y
conda activate evomem-eval
pip install -e '.[eval,test]'
pip install -e vendor/EvoMemBench/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB/agentenv
```
Some optional upstream memory adapters need their own packages. Only install expensive dependencies you are going to use.

## Model endpoint

For OpenAI-compatible model serving, fill `.env` with `WORKER_API_MODE=standard`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `WORKER_MODEL`; the runner passes `--model WORKER_MODEL` to the unchanged upstream worker. Local vLLM is possible. **Do not assume a Qwen checkpoint is capable before running the competence pilot.**

For the original Volcengine ARK batch path, set `WORKER_API_MODE=batch`, `DEEPSEEK_API_KEY`, `BATCH_MODEL`, and run the same worker model across conditions. Official memory writers can have additional ARK and DashScope keys.

## Commands
```bash
python -m pytest -q
python scripts/prepare_manifest.py --validate-only
python scripts/run_experiment.py --plan smoke --dry-run
python scripts/run_experiment.py --plan smoke
python scripts/run_experiment.py --plan competence --dry-run
python scripts/run_experiment.py --plan heldout --dry-run
python scripts/run_experiment.py --plan original --dry-run
python scripts/run_experiment.py --plan learned --dry-run
python scripts/run_experiment.py --plan transfer --dry-run
```
Dry-run never calls inference. To limit expense, use `--train-per-category`, `--test-per-category`, `--methods`, and `--seeds`. Do not run costly plans without reviewing the estimated upper-bound worker calls.

## Cross-run statistics
```bash
python scripts/analyze_results.py runs/RUN_ID
```

The runner refuses to reuse existing run directories. Upstream per-episode file caching **is not** safe as a memory-learning resume mechanism because skipped outputs do not replay memory updates.
