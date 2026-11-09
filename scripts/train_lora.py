#!/usr/bin/env python3
"""Train a separate reward-filtered ALFWorld LoRA worker from official train data."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


class EpisodeDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def encode_episodes(tokenizer, records, max_length):
    encoded_rows = []
    skipped_long = 0
    for row in records:
        encoded = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        input_ids = encoded["input_ids"]
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        if len(input_ids) > max_length:
            skipped_long += 1
            continue
        mask = encoded.get("assistant_masks", encoded.get("assistant_tokens_mask"))
        if mask is None:
            raise RuntimeError(
                "Tokenizer chat template did not return assistant token masks. "
                "Use a Qwen instruct template with generation markers; refusing to train "
                "on user/environment observations as if they were assistant targets."
            )
        if hasattr(mask, "tolist"):
            mask = mask.tolist()
        if mask and isinstance(mask[0], list):
            mask = mask[0]
        if len(mask) != len(input_ids):
            raise RuntimeError("Assistant token mask length does not match tokenized episode")
        labels = [token if bool(is_assistant) else -100 for token, is_assistant in zip(input_ids, mask)]
        if not any(label != -100 for label in labels):
            continue
        attention = encoded.get("attention_mask", [1] * len(input_ids))
        if hasattr(attention, "tolist"):
            attention = attention.tolist()
        if attention and isinstance(attention[0], list):
            attention = attention[0]
        encoded_rows.append({
            "input_ids": input_ids,
            "attention_mask": attention,
            "labels": labels,
            "item_id": row["item_id"],
        })
    return encoded_rows, skipped_long


class AssistantOnlyCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_length = max(len(item["input_ids"]) for item in features)
        input_ids, attention, labels = [], [], []
        for item in features:
            padding = max_length - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * padding)
            attention.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("LoRA training requires a CUDA GPU allocation")
    if args.max_length < 512 or not 0 < args.validation_fraction < 0.5:
        raise ValueError("Invalid max length or validation fraction")
    records = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    if len(records) < 10:
        raise RuntimeError("Need at least 10 successful official-train episodes")
    item_ids = [int(row["item_id"]) for row in records]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Duplicate item IDs in SFT dataset; collect one seed or deduplicate first")
    if any(row.get("source_phase") != "official_train_no_memory" for row in records):
        raise ValueError("SFT data must come from official train no-memory trajectories")
    if any(float(row.get("reward", 0)) < 1.0 for row in records):
        raise ValueError("SFT dataset must contain successful trajectories only")

    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise ImportError("Install: pip install -e '.[lora]' and a CUDA-compatible torch build") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    encoded, skipped_long = encode_episodes(tokenizer, records, args.max_length)
    if len(encoded) < 10:
        raise RuntimeError(
            f"Only {len(encoded)} trajectories fit the assistant-only tokenization "
            f"(skipped too-long={skipped_long}); increase max_length or collect more data"
        )
    rng = random.Random(args.seed)
    order = list(range(len(encoded)))
    rng.shuffle(order)
    n_val = max(1, int(round(len(order) * args.validation_fraction)))
    val_indices, train_indices = order[:n_val], order[n_val:]
    if len(train_indices) < 5:
        raise RuntimeError("Too few training episodes after validation split")
    train_rows = [encoded[i] for i in train_indices]
    val_rows = [encoded[i] for i in val_indices]

    bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    quantization = None
    if args.load_in_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=compute_dtype,
        device_map="auto",
        quantization_config=quantization,
        trust_remote_code=True,
    )
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        max_grad_norm=0.3,
        report_to="none",
        remove_unused_columns=False,
        save_total_limit=2,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=EpisodeDataset(train_rows),
        eval_dataset=EpisodeDataset(val_rows),
        data_collator=AssistantOnlyCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "adapter"))

    metadata = {
        "base_model": args.base_model,
        "dataset": str(args.dataset.resolve()),
        "source_item_ids": sorted(item_ids),
        "train_item_ids": sorted(encoded[i]["item_id"] for i in train_indices),
        "validation_item_ids": sorted(encoded[i]["item_id"] for i in val_indices),
        "encoded_episodes": len(encoded),
        "skipped_too_long": skipped_long,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "assistant_only_loss": True,
        "successful_official_train_trajectories_only": True,
        "official_test_used_for_training": False,
    }
    (args.output_dir / "training_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
