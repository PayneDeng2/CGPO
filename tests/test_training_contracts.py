from pathlib import Path

import pytest

from cgpo.io import read_yaml
from scripts.train import class_distribution
from scripts.train import deep_update
from scripts.train import validate_rows


ROOT = Path(__file__).resolve().parents[1]


def base_row() -> dict:
    return {
        "prompt": "p",
        "completion": "c",
        "label": "D",
        "initial_logp": -1.0,
        "anchor_completion": "a",
        "anchor_initial_logp": -2.0,
    }


def test_no_source_guidance_needs_neither_consensus_nor_source_weight():
    validate_rows(
        [base_row()],
        requires_consensus=False,
        allocation_weight_field=None,
    )


def test_policy_anchored_retains_source_allocation_weight():
    row = base_row()
    row["source_allocation_weight"] = 1.25
    validate_rows(
        [row],
        requires_consensus=False,
        allocation_weight_field="source_allocation_weight",
    )


def test_policy_anchored_fails_if_source_allocation_is_missing():
    with pytest.raises(ValueError, match="source_allocation_weight"):
        validate_rows(
            [base_row()],
            requires_consensus=False,
            allocation_weight_field="source_allocation_weight",
        )


def test_source_allocation_defines_normalized_label_distribution():
    rows = []
    for label, completion, weight in [
        ("D", "d1", 1.0),
        ("D", "d2", 3.0),
        ("U", "u1", 8.0),
    ]:
        row = base_row()
        row.update(
            label=label,
            completion=completion,
            source_allocation_weight=weight,
        )
        rows.append(row)

    selected, probabilities = class_distribution(
        rows,
        "D",
        "source_allocation_weight",
    )
    assert [row["completion"] for row in selected] == ["d1", "d2"]
    assert probabilities.tolist() == pytest.approx([0.25, 0.75])


def test_uniform_distribution_for_source_agnostic_rows():
    first = base_row()
    second = base_row()
    second["completion"] = "c2"
    _, probabilities = class_distribution([first, second], "D", None)
    assert probabilities.tolist() == pytest.approx([0.5, 0.5])


def test_consensus_reference_requires_aligned_consensus():
    row = base_row()
    row["source_allocation_weight"] = 1.0
    with pytest.raises(ValueError, match="aligned_consensus"):
        validate_rows(
            [row],
            requires_consensus=True,
            allocation_weight_field="source_allocation_weight",
        )


def test_policy_and_no_source_overlays_have_distinct_allocation_contracts():
    policy = read_yaml(ROOT / "configs" / "main.yaml")
    deep_update(
        policy,
        read_yaml(ROOT / "configs" / "ablations" / "policy_anchored.yaml"),
    )
    no_source = read_yaml(ROOT / "configs" / "main.yaml")
    deep_update(
        no_source,
        read_yaml(ROOT / "configs" / "ablations" / "no_source_guidance.yaml"),
    )

    assert policy["method"]["reference_assignment"] == "policy_anchored"
    assert no_source["method"]["reference_assignment"] == "policy_anchored"
    assert policy["method"]["source_allocation_weight_field"] == "source_allocation_weight"
    assert no_source["method"]["source_allocation_weight_field"] is None


def test_reference_variants_share_source_allocation_contract():
    for filename in [
        "policy_anchored.yaml",
        "consensus_anchored.yaml",
        "label_conditional.yaml",
    ]:
        config = read_yaml(ROOT / "configs" / "main.yaml")
        deep_update(
            config,
            read_yaml(ROOT / "configs" / "ablations" / filename),
        )
        assert config["method"]["source_allocation_weight_field"] == (
            "source_allocation_weight"
        )
