import math

import pytest
import torch

from cgpo.scoring import (
    completion_scores_from_logits,
    make_mismatched_anchor_completions,
    tokenize_prompt_completion_batch,
)


def test_completion_scoring_ignores_prompt_and_padding():
    logits = torch.zeros(1, 5, 4)
    logits[0, 1, 2] = 8.0
    logits[0, 2, 3] = 8.0
    labels = torch.tensor([[-100, -100, 2, 3, -100]])
    scores, entropies = completion_scores_from_logits(logits, labels)

    expected = torch.log_softmax(logits[:, :-1], dim=-1)
    expected_score = (expected[0, 1, 2] + expected[0, 2, 3]) / 2
    assert scores.item() == pytest.approx(expected_score.item())
    assert 0.0 <= entropies.item() <= 1.0


def test_uniform_logits_have_unit_normalized_entropy():
    logits = torch.zeros(2, 4, 11)
    labels = torch.tensor([[-100, 1, 2, 3], [-100, 3, 2, 1]])
    _, entropies = completion_scores_from_logits(logits, labels)
    assert entropies.tolist() == pytest.approx([1.0, 1.0], abs=1e-6)


def test_each_example_requires_completion_tokens():
    with pytest.raises(ValueError):
        completion_scores_from_logits(
            torch.zeros(1, 3, 4),
            torch.full((1, 3), -100),
        )


class BoundaryMergeTokenizer:
    pad_token_id = 0
    eos_token_id = 3
    bos_token_id = 4
    eos_token = "<eos>"
    chat_template = None

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == "a":
            return [1]
        if text == "ab<eos>":
            return [9, 3]
        if text == "b<eos>":
            return [2, 3]
        return [ord(character) % 17 + 5 for character in text]


def test_boundary_merge_is_assigned_to_completion():
    encoded = tokenize_prompt_completion_batch(
        BoundaryMergeTokenizer(),
        ["a"],
        ["b"],
        max_length=8,
        device="cpu",
    )
    assert encoded["input_ids"].tolist() == [[4, 9, 3]]
    assert encoded["labels"].tolist() == [[-100, 9, 3]]


def test_anchor_completions_always_come_from_another_prompt():
    prompt_ids = ["a", "a", "b", "c"]
    completions = ["a0", "a1", "b0", "c0"]
    anchors = make_mismatched_anchor_completions(prompt_ids, completions, seed=7)
    owner = {completion: prompt for completion, prompt in zip(completions, prompt_ids)}
    assert all(owner[anchor] != prompt for anchor, prompt in zip(anchors, prompt_ids))


def test_anchor_construction_rejects_single_prompt():
    with pytest.raises(ValueError, match="distinct prompts"):
        make_mismatched_anchor_completions(["a", "a"], ["x", "y"], seed=0)
