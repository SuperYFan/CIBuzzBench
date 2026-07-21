#!/usr/bin/env python
"""LoRA/QLoRA supervised fine-tuning for Qwen/GLM chat models."""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--validation_file", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", default="auto", help='Use "auto" or comma-separated module suffixes.')
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--report_to", default="none")
    return parser.parse_args()


class ChatSFTDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer: AutoTokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.rows = []
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def _format_prompt(self, messages: list[dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        messages = row["messages"]
        prompt_messages = messages[:-1]
        answer = messages[-1]["content"]
        prompt_text = self._format_prompt(prompt_messages)
        eos = self.tokenizer.eos_token or ""
        full_text = prompt_text + answer + eos

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        full = self.tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=self.max_seq_length)
        input_ids = full.input_ids
        labels = list(input_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        if all(x == -100 for x in labels) and labels:
            labels[-1] = input_ids[-1]
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


@dataclass
class DataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        input_ids = pad_sequence([f["input_ids"] for f in features], batch_first=True, padding_value=pad_id)
        attention_mask = pad_sequence([f["attention_mask"] for f in features], batch_first=True, padding_value=0)
        labels = pad_sequence([f["labels"] for f in features], batch_first=True, padding_value=-100)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def find_lora_targets(model: torch.nn.Module) -> list[str]:
    targets = set()
    excluded = {"lm_head", "output_layer", "embed_tokens"}
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__.lower()
        if "linear" not in cls_name:
            continue
        leaf = name.split(".")[-1]
        if leaf not in excluded:
            targets.add(leaf)
    if not targets:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return sorted(targets)


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        dtype=torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else "auto",
        quantization_config=quantization_config,
        device_map="auto" if args.load_in_4bit else None,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = find_lora_targets(model) if args.target_modules == "auto" else [x.strip() for x in args.target_modules.split(",") if x.strip()]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = ChatSFTDataset(args.train_file, tokenizer, args.max_seq_length)
    eval_dataset = ChatSFTDataset(args.validation_file, tokenizer, args.max_seq_length) if args.validation_file else None

    training_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "report_to": [] if args.report_to == "none" else args.report_to.split(","),
        "remove_unused_columns": False,
        "optim": "paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        "lr_scheduler_type": "cosine",
    }
    signature = inspect.signature(TrainingArguments.__init__)
    eval_strategy_name = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    if eval_dataset is not None:
        training_kwargs["eval_steps"] = args.eval_steps
        training_kwargs[eval_strategy_name] = "steps"
    else:
        training_kwargs[eval_strategy_name] = "no"
    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    (Path(args.output_dir) / "sft_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
