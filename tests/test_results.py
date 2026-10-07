import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_results import summarize


def test_overall_is_mean_of_all_seven_benchmarks():
    scores = {
        "GSM8K": 10.0,
        "MATH": 20.0,
        "TheoremQA": 30.0,
        "BBH": 40.0,
        "MMLU": 50.0,
        "IFEval": 60.0,
        "MBPP": 70.0,
    }
    result = summarize(scores)
    assert result["Math Avg."] == pytest.approx(20.0)
    assert result["Reasoning Avg."] == pytest.approx(45.0)
    assert result["Overall Avg."] == pytest.approx(40.0)

