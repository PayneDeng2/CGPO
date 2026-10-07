#!/usr/bin/env python
"""Compute the category and seven-benchmark averages used in the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MATH = ("GSM8K", "MATH", "TheoremQA")
REASONING = ("BBH", "MMLU")
ALL_BENCHMARKS = MATH + REASONING + ("IFEval", "MBPP")


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def summarize(scores: dict[str, float]) -> dict[str, float]:
    missing = [name for name in ALL_BENCHMARKS if name not in scores]
    if missing:
        raise ValueError(f"missing benchmark scores: {missing}")
    return {
        "Math Avg.": mean([float(scores[name]) for name in MATH]),
        "Reasoning Avg.": mean([float(scores[name]) for name in REASONING]),
        "Overall Avg.": mean([float(scores[name]) for name in ALL_BENCHMARKS]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="JSON object containing seven percentage scores")
    args = parser.parse_args()
    scores = json.loads(Path(args.results).read_text(encoding="utf-8"))
    print(json.dumps(summarize(scores), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

