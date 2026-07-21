#!/usr/bin/env python
"""Validate the released CIBuzzBench CSV and split files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "row_id",
    "split",
    "term_zh",
    "explanation_zh",
    "meaning_en",
    "equivalent_en",
    "distractor_literal_en",
    "distractor_cultural_mismatch_en",
    "distractor_pragmatic_neighbor_en",
    "harmful_label",
    "category_label",
]
EXPECTED_HARMFUL_LABELS = {"harmful", "non-harmful"}
EXPECTED_CATEGORIES = {
    "Experience",
    "Quotation",
    "Stylistic device",
    "Homophonic pun",
    "Slang",
    "Abbreviation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/cibuzzbench.csv")
    parser.add_argument("--expected_total", type=int, default=3001)
    parser.add_argument("--expected_train", type=int, default=2401)
    parser.add_argument("--expected_test", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.data)
    df = pd.read_csv(path, dtype=str).fillna("")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    if len(df) != args.expected_total:
        raise SystemExit(f"Expected {args.expected_total} rows, found {len(df)}")

    df["row_id"] = df["row_id"].astype(int)
    expected_ids = list(range(len(df)))
    if df["row_id"].tolist() != expected_ids:
        raise SystemExit("row_id must be contiguous and 0-indexed")

    split_counts = df["split"].value_counts().to_dict()
    if split_counts.get("train", 0) != args.expected_train or split_counts.get("test", 0) != args.expected_test:
        raise SystemExit(f"Unexpected split counts: {split_counts}")

    harmful_labels = set(df["harmful_label"])
    if harmful_labels != EXPECTED_HARMFUL_LABELS:
        raise SystemExit(f"Unexpected harmful labels: {sorted(harmful_labels)}")

    categories = set(df["category_label"])
    if categories != EXPECTED_CATEGORIES:
        raise SystemExit(f"Unexpected category labels: {sorted(categories)}")

    text_columns = [column for column in REQUIRED_COLUMNS if column not in {"row_id", "split"}]
    empty_counts = {column: int((df[column].astype(str).str.strip() == "").sum()) for column in text_columns}
    bad_empty = {column: count for column, count in empty_counts.items() if count}
    if bad_empty:
        raise SystemExit(f"Unexpected empty cells: {bad_empty}")

    print("CIBuzzBench dataset validation passed.")
    print(f"Rows: {len(df)}")
    print(f"Split counts: {split_counts}")
    print(f"Harmful labels: {sorted(harmful_labels)}")
    print(f"Categories: {sorted(categories)}")


if __name__ == "__main__":
    main()
