#!/usr/bin/env python
"""Build supervised JSONL files for CIBuzzBench fine-tuning experiments.

The script converts the released benchmark CSV into chat-style examples for the
three paper tasks. It uses the official entry-level train/test split already
stored in the public CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluate_api import (
    build_options,
    equivalent_messages,
    harmfulness_messages,
    load_data,
    meaning_messages,
    parse_task_list,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/cibuzzbench.csv")
    parser.add_argument("--output_dir", default="data/sft_jsonl")
    parser.add_argument("--splits", default="train,test", help="Comma-separated subset of train,test.")
    parser.add_argument(
        "--tasks",
        default="meaning,equivalent,harmfulness",
        help="Comma-separated subset of meaning,equivalent,harmfulness.",
    )
    parser.add_argument("--prompt_languages", default="en,zh", help="Comma-separated subset of en,zh.")
    parser.add_argument("--option_seed", type=int, default=1111)
    return parser.parse_args()


def write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")


def make_example(row: pd.Series, task: str, lang: str, option_seed: int) -> dict:
    if task == "meaning":
        messages = meaning_messages(str(row["term_zh"]), lang)
        answer = str(row["meaning_en"])
    elif task == "equivalent":
        options_text, gold_letter, _ = build_options(row, option_seed)
        messages = equivalent_messages(str(row["term_zh"]), options_text, lang)
        answer = gold_letter
    elif task == "harmfulness":
        messages = harmfulness_messages(str(row["term_zh"]), lang)
        answer = str(row["harmful_label"])
    else:
        raise ValueError(f"Unknown task: {task}")

    return {
        "row_id": int(row["row_id"]),
        "split": str(row["split"]),
        "task": task,
        "prompt_language": lang,
        "messages": messages + [{"role": "assistant", "content": answer}],
    }


def main() -> None:
    args = parse_args()
    tasks = parse_task_list(args.tasks)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    prompt_languages = [item.strip() for item in args.prompt_languages.split(",") if item.strip()]
    unknown_splits = set(splits) - {"train", "test"}
    unknown_langs = set(prompt_languages) - {"en", "zh"}
    if unknown_splits:
        raise ValueError(f"Unknown splits: {sorted(unknown_splits)}")
    if unknown_langs:
        raise ValueError(f"Unknown prompt languages: {sorted(unknown_langs)}")

    full_df = load_data(args.data, split="all", max_samples=0)
    out_dir = Path(args.output_dir)
    summary = []

    for split in splits:
        split_df = full_df[full_df["split"] == split].reset_index(drop=True)
        for lang in prompt_languages:
            for task in tasks:
                examples = [make_example(row, task, lang, args.option_seed) for _, row in split_df.iterrows()]
                filename = f"{split}_{lang}_{task}.jsonl"
                write_jsonl(out_dir / filename, examples)
                summary.append(
                    {
                        "split": split,
                        "prompt_language": lang,
                        "task": task,
                        "examples": len(examples),
                        "file": filename,
                    }
                )

    pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
