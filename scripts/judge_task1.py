#!/usr/bin/env python
"""Score Meaning Explanation predictions with an OpenAI-compatible LLM judge."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from evaluate_api import ApiConfig, ChatClient, load_data, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/cibuzzbench.csv")
    parser.add_argument("--predictions", required=True, help="CSV with row_id and prediction columns.")
    parser.add_argument("--output", default="", help="Output CSV. Defaults to <predictions>.judged.csv.")
    parser.add_argument("--model", default=os.getenv("LLM_JUDGE_MODEL", ""))
    parser.add_argument("--base_url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=8)
    parser.add_argument("--disable_reasoning", action="store_true")
    return parser.parse_args()


def parse_score(text: str) -> str:
    raw = str(text).strip()
    if re.fullmatch(r"[0-5]", raw):
        return raw
    match = re.search(r"(?<!\d)([0-5])(?!\d)", raw)
    return match.group(1) if match else ""


def judge_messages(row: pd.Series) -> list[dict[str, str]]:
    user = f"""你正在评估 Meaning Explanation 任务中模型预测的质量。请严格按照以下人工评估评分规则，对 prediction 打 0--5 分。

评估时请同时参考中文梗词、中文解释和 gold 英文解释，并以中文解释所锚定的词义为准。评估目标是 prediction 与 gold 在语义和语用含义上是否等价，而不是英文措辞是否完全一致。

评分规则：
5 分：完全正确。prediction 准确表达了 gold 中的核心非字面含义、语用功能、指向对象和使用条件。表达方式不同也可以给 5 分。
4 分：基本正确。主要含义正确，但遗漏了较小的语气、使用边界或语用细节。
3 分：部分正确。prediction 与目标词义相关，但不完整、过于笼统，或含义有一定偏移。
2 分：大部分错误。prediction 只抓住了表层字面线索或较弱的相关含义，没有解释出该梗词的真实网络含义。
1 分：错误但非空。prediction 给出了另一个词义、错误的文化来源、错误的指向对象，或错误的语用功能。
0 分：无效或完全错误。prediction 为空、拒答、只重复原词而没有解释、使用了无法评估的错误语言，或内容完全无关。

如果某条 prediction 介于两个分数之间，当遗漏的信息会改变读者对该梗词实际含义的理解时，请给较低分；如果差异主要只是英文措辞不同，请给较高分。

请只输出一个整数分数：0、1、2、3、4 或 5。不要输出解释、标点、JSON 或其他文字。

待评估样本：
中文梗词：{row["term_zh"]}
中文解释：{row["explanation_zh"]}
gold 英文解释：{row["meaning_en"]}
prediction：{row["prediction"]}"""
    return [
        {
            "role": "system",
            "content": "你是严格、一致的中文互联网梗词评测员。你的输出必须只有一个 0 到 5 的整数分数。",
        },
        {"role": "user", "content": user},
    ]


def main() -> None:
    args = parse_args()
    if not args.model:
        raise SystemExit("Set --model or the LLM_JUDGE_MODEL environment variable.")
    if not args.api_key:
        raise SystemExit("Set --api_key or the OPENAI_API_KEY environment variable.")

    data = load_data(args.data, split="all", max_samples=0)
    preds = pd.read_csv(args.predictions, dtype=str).fillna("")
    if "row_id" not in preds.columns or "prediction" not in preds.columns:
        raise ValueError("--predictions must contain row_id and prediction columns")
    preds["row_id"] = preds["row_id"].astype(int)
    merged = preds.merge(
        data[["row_id", "term_zh", "explanation_zh", "meaning_en"]],
        on="row_id",
        how="left",
        validate="many_to_one",
    )
    if merged["meaning_en"].isna().any():
        raise ValueError("Some prediction row_id values are not present in the benchmark data")

    client = ChatClient(
        ApiConfig(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
            top_p=None,
            disable_reasoning=args.disable_reasoning,
        )
    )

    rows = [row for _, row in merged.iterrows()]

    def score(row: pd.Series) -> dict[str, Any]:
        base = row.to_dict()
        try:
            result = client.complete(judge_messages(row), max_tokens=args.max_tokens)
            raw = result["content"].strip()
            base.update(
                {
                    "llm_score_0_5": parse_score(raw),
                    "llm_judge_raw_response": raw,
                    "llm_judge_model": args.model,
                    "llm_judge_success": True,
                    "llm_judge_error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            base.update(
                {
                    "llm_score_0_5": "",
                    "llm_judge_raw_response": "",
                    "llm_judge_model": args.model,
                    "llm_judge_success": False,
                    "llm_judge_error": str(exc),
                }
            )
        return base

    if args.workers <= 1:
        scored = [score(row) for row in rows]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(score, row) for row in rows]
            scored = [future.result() for future in futures]

    output = Path(args.output) if args.output else Path(args.predictions).with_suffix(".judged.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    scored_df = pd.DataFrame(scored)
    scored_df.to_csv(output, index=False, encoding="utf-8")
    numeric = pd.to_numeric(scored_df["llm_score_0_5"], errors="coerce")
    save_json(
        output.with_suffix(".metrics.json"),
        {
            "judge_model": args.model,
            "num_examples": int(len(scored_df)),
            "num_scored": int(numeric.notna().sum()),
            "mean_score_0_5": float(numeric.mean()) if numeric.notna().any() else None,
        },
    )


if __name__ == "__main__":
    main()
