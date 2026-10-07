#!/usr/bin/env python
"""Precompute initial-policy and source-model sequence statistics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgpo.io import read_jsonl, write_jsonl
from cgpo.scoring import (
    completion_scores_from_logits,
    make_mismatched_anchor_completions,
    tokenize_prompt_completion_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial-model", required=True)
    parser.add_argument("--source-model", nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--anchor-seed", type=int, default=42)
    return parser.parse_args()


def validate_rows(rows: list[dict]) -> None:
    required = {"prompt_id", "prompt", "completion", "label"}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"row {index} is missing {sorted(missing)}")
        if row["label"] not in {"D", "U"}:
            raise ValueError(f"row {index} label must be D or U")


def score_texts(
    model_name: str,
    rows: list[dict],
    completions: list[str],
    args: argparse.Namespace,
) -> tuple[list[float], list[float]]:
    dtype = torch.float32 if str(args.device).startswith("cpu") else getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=False,
    ).to(args.device)
    model.eval()

    all_scores: list[float] = []
    all_entropies: list[float] = []
    for start in range(0, len(rows), args.batch_size):
        stop = start + args.batch_size
        batch = tokenize_prompt_completion_batch(
            tokenizer,
            [str(row["prompt"]) for row in rows[start:stop]],
            completions[start:stop],
            max_length=args.max_length,
            device=args.device,
        )
        with torch.inference_mode():
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits
            scores, entropies = completion_scores_from_logits(logits, batch["labels"])
        all_scores.extend(scores.cpu().tolist())
        all_entropies.extend(entropies.cpu().tolist())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_scores, all_entropies


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    validate_rows(rows)
    completions = [str(row["completion"]) for row in rows]
    anchors = make_mismatched_anchor_completions(
        [row["prompt_id"] for row in rows],
        completions,
        seed=args.anchor_seed,
    )

    initial_scores, _ = score_texts(args.initial_model, rows, completions, args)
    anchor_scores, _ = score_texts(args.initial_model, rows, anchors, args)

    source_scores: list[list[float]] = []
    source_entropies: list[list[float]] = []
    for model_name in args.source_model:
        scores, entropies = score_texts(model_name, rows, completions, args)
        source_scores.append(scores)
        source_entropies.append(entropies)

    for index, row in enumerate(rows):
        row["initial_logp"] = initial_scores[index]
        row["anchor_completion"] = anchors[index]
        row["anchor_initial_logp"] = anchor_scores[index]
        row["source_logps"] = [values[index] for values in source_scores]
        row["source_entropies"] = [values[index] for values in source_entropies]
        row["source_models"] = list(args.source_model)
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
