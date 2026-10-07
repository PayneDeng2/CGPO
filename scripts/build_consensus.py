#!/usr/bin/env python
"""Build calibrated and baseline-aligned source references offline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgpo.consensus import (
    align_prompt_baselines,
    calibrated_consensus,
    fit_entropy_calibration,
    mean_consensus,
)
from cgpo.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration-file")
    parser.add_argument("--aggregation", choices=["calibrated_entropy", "mean"], default="calibrated_entropy")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--weight-floor", type=float, default=0.02)
    parser.add_argument("--z-clip", type=float, default=5.0)
    parser.add_argument("--baseline-alignment", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def matrix(rows: list[dict], key: str) -> torch.Tensor:
    try:
        value = torch.tensor([row[key] for row in rows], dtype=torch.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"rows must contain aligned numeric lists in {key}") from error
    if value.ndim != 2:
        raise ValueError(f"{key} must form [num_examples, num_sources]")
    return value


def source_order(rows: list[dict]) -> tuple[str, ...]:
    try:
        expected = tuple(str(name) for name in rows[0]["source_models"])
    except (KeyError, TypeError) as error:
        raise ValueError("scored rows must record source_models") from error
    if not expected:
        raise ValueError("source_models must not be empty")
    for index, row in enumerate(rows[1:], start=1):
        if tuple(str(name) for name in row.get("source_models", [])) != expected:
            raise ValueError(f"row {index} uses a different source order")
    return expected


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    expected_sources = source_order(rows)
    source_scores = matrix(rows, "source_logps")
    source_entropies = matrix(rows, "source_entropies")

    if args.aggregation == "calibrated_entropy":
        if not args.calibration_file:
            raise ValueError(
                "--calibration-file is required for calibrated_entropy aggregation."
            )
        calibration_rows = read_jsonl(args.calibration_file)
        if source_order(calibration_rows) != expected_sources:
            raise ValueError(
                "calibration and training files must use the same source models in the same order"
            )
        calibration = fit_entropy_calibration(matrix(calibration_rows, "source_entropies"))
        consensus, weights, calibrated = calibrated_consensus(
            source_scores,
            source_entropies,
            calibration,
            temperature=args.temperature,
            weight_floor=args.weight_floor,
            z_clip=args.z_clip,
        )
    else:
        consensus = mean_consensus(source_scores)
        weights = torch.full_like(source_scores, 1.0 / source_scores.shape[1])
        calibrated = torch.zeros_like(source_scores)

    initial = torch.tensor([row["initial_logp"] for row in rows], dtype=torch.float64)
    if args.baseline_alignment:
        aligned, shift = align_prompt_baselines(
            [row["prompt_id"] for row in rows], initial, consensus
        )
    else:
        aligned = consensus.detach()
        shift = torch.zeros_like(consensus)

    for index, row in enumerate(rows):
        row["source_consensus"] = consensus[index].item()
        row["aligned_consensus"] = aligned[index].item()
        row["baseline_shift"] = shift[index].item()
        row["source_weights"] = weights[index].tolist()
        row["calibrated_entropies"] = calibrated[index].tolist()
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
