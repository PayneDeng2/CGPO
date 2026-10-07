import pytest
import torch

from cgpo.consensus import (
    EntropyCalibration,
    align_prompt_baselines,
    calibrated_consensus,
    calibrated_entropy_weights,
    fit_entropy_calibration,
)


def test_calibrated_weights_form_required_simplex():
    entropies = torch.tensor([[0.1, 0.5, 0.9], [0.8, 0.5, 0.2]])
    calibration = EntropyCalibration(
        mean=torch.tensor([0.5, 0.5, 0.5]),
        std=torch.tensor([0.2, 0.2, 0.2]),
    )
    _, weights = calibrated_entropy_weights(
        entropies,
        calibration,
        temperature=0.8,
        weight_floor=0.02,
    )

    assert torch.allclose(weights.sum(dim=-1), torch.ones(2))
    assert torch.all(weights >= 0.02)
    assert weights[0, 0] > weights[0, 2]
    assert weights[1, 2] > weights[1, 0]


def test_consensus_remains_in_convex_hull():
    scores = torch.tensor([[-1.0, -3.0], [-5.0, -2.0]])
    entropies = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    calibration = fit_entropy_calibration(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    consensus, _, _ = calibrated_consensus(
        scores,
        entropies,
        calibration,
        temperature=1.0,
        weight_floor=0.0,
    )

    assert torch.all(consensus >= scores.min(dim=-1).values)
    assert torch.all(consensus <= scores.max(dim=-1).values)


def test_prompt_alignment_matches_means_and_preserves_differences():
    initial = torch.tensor([-1.0, -3.0, -2.0, -4.0])
    source = torch.tensor([-5.0, -8.0, -10.0, -11.0])
    aligned, shift = align_prompt_baselines(["a", "a", "b", "b"], initial, source)

    assert shift.tolist() == pytest.approx([4.5, 4.5, 7.5, 7.5])
    assert aligned[:2].mean().item() == pytest.approx(initial[:2].mean().item())
    assert aligned[2:].mean().item() == pytest.approx(initial[2:].mean().item())
    assert (aligned[0] - aligned[1]).item() == pytest.approx((source[0] - source[1]).item())


def test_constant_entropy_source_uses_epsilon_scale():
    calibration = fit_entropy_calibration(
        torch.tensor([[0.2, 0.7], [0.2, 0.8], [0.2, 0.9]]),
        eps=1e-4,
    )
    assert calibration.std[0].item() == pytest.approx(1e-4)

