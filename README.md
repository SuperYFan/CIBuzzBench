# CIBuzzBench

CIBuzzBench is a benchmark for Chinese-to-English cross-lingual understanding of
Chinese internet buzzwords. It contains 3,001 annotated entries and supports the
three tasks described in the paper:

1. **Meaning Explanation**: generate a concise English explanation of the
   non-literal meaning of a Chinese internet buzzword.
2. **Equivalent Selection**: choose the best English equivalent from one gold
   option and three controlled distractors.
3. **Harmfulness Detection**: classify the annotated buzzword sense as
   `harmful` or `non-harmful`.

## Repository Layout

```text
CIBuzzBench/
  data/
    cibuzzbench.csv
    splits/
      entry_splits.csv
      split_counts.csv
      train.csv
      test.csv
  scripts/
    evaluate_api.py
    judge_task1.py
    build_sft_jsonl.py
    train_lora_sft.py
    validate_dataset.py
  requirements.txt
  requirements-sft.txt
```

## Data Fields

`data/cibuzzbench.csv` uses UTF-8 encoding and contains the following columns:

- `row_id`: 0-indexed entry id.
- `split`: `train` or `test` for the 4:1 split used in few-shot and LoRA
  fine-tuning experiments.
- `term_zh`: Chinese internet buzzword.
- `explanation_zh`: Chinese explanation used as the source-language sense
  anchor.
- `meaning_en`: gold English meaning explanation for Task 1.
- `equivalent_en`: gold English equivalent for Task 2.
- `distractor_literal_en`: literal or surface-form distractor.
- `distractor_cultural_mismatch_en`: keyword or cultural-mismatch distractor.
- `distractor_pragmatic_neighbor_en`: pragmatic-neighbor distractor.
- `harmful_label`: gold label for Task 3, either `harmful` or `non-harmful`.
- `category_label`: one of `Experience`, `Quotation`, `Stylistic device`,
  `Homophonic pun`, `Slang`, or `Abbreviation`.

The official train/test split contains 2,401 training entries and 600 test
entries.

## Installation

```bash
python -m pip install -r requirements.txt
```

`bert-score` downloads its default English model on first use. Add
`--skip_bertscore` for fast smoke tests.

## Validate the Dataset

```bash
python scripts/validate_dataset.py
```

Expected output includes 3,001 total entries, 2,401 train entries, and 600 test
entries.

## Run API Evaluation

The evaluation script uses an OpenAI-compatible chat-completions API. Set the
model and credentials through environment variables or command-line arguments.

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export MODEL="gpt-5.5"
```

On Windows PowerShell, use `$env:OPENAI_API_KEY`, `$env:OPENAI_BASE_URL`,
and `$env:MODEL` instead.

Smoke test:

```bash
python scripts/evaluate_api.py \
  --data data/cibuzzbench.csv \
  --tasks meaning,equivalent,harmfulness \
  --prompt_language en \
  --max_samples 10 \
  --skip_bertscore
```

Full zero-shot evaluation with five option-shuffle seeds for Task 2:

```bash
python scripts/evaluate_api.py \
  --data data/cibuzzbench.csv \
  --tasks meaning,equivalent,harmfulness \
  --prompt_language en \
  --option_seeds 1111,2222,3333,4444,5555
```

Chinese-prompt evaluation:

```bash
python scripts/evaluate_api.py \
  --data data/cibuzzbench.csv \
  --tasks meaning,equivalent,harmfulness \
  --prompt_language zh \
  --option_seeds 1111,2222,3333,4444,5555
```

Outputs are written under `runs/api_eval/<model>/<prompt_language>/`.

## Task 1 LLM-Judge Scoring

To score Meaning Explanation predictions on a 0-5 semantic-equivalence scale,
prepare a CSV containing at least `row_id` and `prediction`, then run the judge
with the same OpenAI-compatible API settings.

```bash
export LLM_JUDGE_MODEL="gemini-3.1-pro-preview"

python scripts/judge_task1.py \
  --data data/cibuzzbench.csv \
  --predictions runs/api_eval/gpt-5.5/en/task1_meaning_explanation/predictions.csv \
  --output runs/api_eval/gpt-5.5/en/task1_meaning_explanation/predictions.judged.csv
```

## Build SFT JSONL Files

The following command builds chat-style supervised examples for the three tasks
under both English and Chinese prompt settings:

```bash
python scripts/build_sft_jsonl.py \
  --data data/cibuzzbench.csv \
  --output_dir data/sft_jsonl \
  --option_seed 1111
```

This writes task- and prompt-language-specific JSONL files for the official
train/test split. The script only constructs supervised data; it does not train
or merge LoRA adapters.

## LoRA Fine-Tuning

Install the optional SFT dependencies:

```bash
python -m pip install -r requirements-sft.txt
```

Build supervised JSONL files first, then train a task-specific adapter. For
example, to train Qwen3-8B on English-prompt Meaning Explanation:

```bash
python scripts/train_lora_sft.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --train_file data/sft_jsonl/train_en_meaning.jsonl \
  --output_dir outputs/qwen3-8b-task1-en-lora \
  --fp16 \
  --num_train_epochs 3 \
  --learning_rate 2e-4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05
```

After training, serve the base model plus adapter with your preferred local
inference stack and run `scripts/evaluate_api.py` against that
OpenAI-compatible endpoint.

## Local Model-Size Experiments

For local model-size experiments such as Qwen3-4B, Qwen3-8B, and Qwen3-14B,
serve each model through an OpenAI-compatible endpoint and reuse the same
evaluation script:

```bash
python scripts/evaluate_api.py \
  --base_url http://127.0.0.1:8000/v1 \
  --api_key local-key \
  --model Qwen/Qwen3-8B \
  --data data/cibuzzbench.csv \
  --tasks meaning,equivalent,harmfulness \
  --prompt_language zh \
  --option_seeds 1111,2222,3333,4444,5555 \
  --top_p 1.0 \
  --skip_bertscore
```

## Evaluation Notes

- Decoding is deterministic by default with `temperature=0.0`.
- API `top_p` is omitted unless explicitly passed through `--top_p`.
- Task 2 reports macro F1 and supports multiple option-shuffle seeds. Options
  are shuffled deterministically from the seed, `row_id`, and Chinese term.
- Task 3 reports accuracy, macro F1, and harmful-class F1.
- The benchmark contains offensive or harmful expressions for diagnostic
  research purposes. Use it responsibly and avoid deploying it directly as a
  moderation system without context-specific review.
