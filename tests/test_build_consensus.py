import pytest

from scripts.build_consensus import source_order


def test_source_order_requires_consistent_metadata():
    rows = [
        {"source_models": ["a", "b"]},
        {"source_models": ["a", "b"]},
    ]
    assert source_order(rows) == ("a", "b")


def test_source_order_rejects_reordered_sources():
    with pytest.raises(ValueError, match="different source order"):
        source_order(
            [
                {"source_models": ["a", "b"]},
                {"source_models": ["b", "a"]},
            ]
        )


def test_source_order_requires_metadata():
    with pytest.raises(ValueError, match="must record source_models"):
        source_order([{}])
