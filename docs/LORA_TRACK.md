# Separate continual LoRA/SFT worker track

This track changes the worker's parameters. It is not an external memory backend and must not be combined with the memory-method table.

## Protocol

1. Run a no-memory data-collection stream on official train items only.
2. Export successful episodes only; the exporter checks that every source item is in the official train split and that no test item appears.
3. Fine-tune a LoRA adapter with loss on assistant tokens only. User/environment observations are context, not targets.
4. Serve the same base checkpoint with the trained LoRA adapter.
5. Evaluate base and LoRA workers with no external memory on the identical official test IDs and seed/order.
6. Compare paired per-task results with the supplied script.

The LoRA track uses reward-filtered behavioral cloning from successful official-train trajectories. It does not use test outcomes for training or hyperparameter selection.

## Collect training trajectories

Use the evaluator environment and local worker endpoint. The default plan collects 50 tasks per category:

    python scripts/run_experiment.py --plan collect --dry-run
    python scripts/run_experiment.py --plan collect

The collection run has protocol=collect and methods=['none']. It stores full per-episode conversations in the run directory. Do not use original/online outputs as SFT data.

## Export successful trajectories

    python scripts/export_lora_data.py runs/COLLECT_RUN \
      --output runs/lora/train.jsonl

The exporter requires a completed collection run, checks split disjointness and item IDs, and exports only successful no-memory trajectories. If fewer than ten successes are available, collect more official train tasks.

## Train LoRA

Install the optional training dependencies in an environment with a CUDA-compatible PyTorch build:

    python -m pip install -e '.[lora]'

BitsAndBytes is required for the default 4-bit QLoRA path:

    python -m pip install bitsandbytes

Run with the same base checkpoint used by the worker:

    python scripts/train_lora.py \
      --dataset runs/lora/train.jsonl \
      --base-model YOUR_BASE_CHAT_MODEL \
      --output-dir runs/lora/adapter \
      --load-in-4bit \
      --epochs 1 \
      --max-length 8192

The script requires a tokenizer chat template that exposes assistant-token masks (for example, a Qwen instruct template with generation markers). It fails closed if those masks are unavailable; it does not silently train on user/environment observations. Episodes longer than max-length are skipped rather than truncated mid-trajectory. Review training_manifest.json for exact source/train/validation item IDs.

## Evaluate base and LoRA workers

Run the base model and the LoRA-served model as separate no-memory online plans. The vLLM LoRA alias must match the --model argument supplied to the second run.

Example vLLM adapter serving:

    vllm serve YOUR_BASE_CHAT_MODEL \
      --enable-lora \
      --lora-modules evomem-lora=/absolute/path/to/runs/lora/adapter/adapter \
      --host 127.0.0.1 --port 8000

Base run:

    python scripts/run_experiment.py --plan original --methods none --seeds 0 \
      --model YOUR_BASE_CHAT_MODEL --output-dir runs/lora/base_eval

LoRA run:

    python scripts/run_experiment.py --plan original --methods none --seeds 0 \
      --model evomem-lora --output-dir runs/lora/lora_eval

Both runs must use the same base checkpoint, task manifest, action rounds, temperature, and decoding configuration. Only the LoRA adapter differs.

## Paired comparison

    python scripts/compare_lora_runs.py \
      runs/lora/base_eval runs/lora/lora_eval \
      --output runs/lora/paired_comparison.json

The comparison script refuses different test item IDs and reports paired success/reward differences overall and by category. It also stores per-task paired rows and a simple paired bootstrap interval. With one order seed, interpret the interval descriptively; repeat with additional seeds for stronger evidence.
