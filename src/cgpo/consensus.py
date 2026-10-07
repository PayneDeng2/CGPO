"""Offline source-consensus construction from Section 4.2 of the paper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import torch


@dataclass(frozen=True)
class EntropyCalibration:
    """Per-source entropy statistics fitted on a calibration split."""

    mean: torch.Tensor
    std: torch.Tensor

    def to(self, device: torch.device | str) -> "EntropyCalibration":
        return EntropyCalibration(self.mean.to(device), self.std.to(device))


def _check_matrix(name: str, value: torch.Tensor) -> None:
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{name} must have shape [num_examples, num_sources].")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values.")


def fit_entropy_calibration(
    normalized_entropies: torch.Tensor,
    eps: float = 1e-6,
) -> EntropyCalibration:
    """Fit the per-source mean and standard deviation used in Eq. (10)."""

    _check_matrix("normalized_entropies", normalized_entropies)
    if eps <= 0:
        raise ValueError("eps must be positive.")
    mean = normalized_entropies.mean(dim=0)
    std = normalized_entropies.std(dim=0, unbiased=False).clamp_min(eps)
    return EntropyCalibration(mean=mean, std=std)


def calibrated_entropy_weights(
    normalized_entropies: torch.Tensor,
    calibration: EntropyCalibration,
    *,
    temperature: float,
    weight_floor: float,
    z_clip: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return calibrated entropies and convex source weights.

    Input tensors use shape ``[num_examples, num_sources]``. The returned
    weights satisfy ``sum_i w_i = 1`` and ``w_i >= weight_floor``.
    """

    _check_matrix("normalized_entropies", normalized_entropies)
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if z_clip < 0:
        raise ValueError("z_clip must be non-negative.")

    num_sources = normalized_entropies.shape[1]
    if not 0 <= weight_floor < 1.0 / num_sources:
        raise ValueError("weight_floor must satisfy 0 <= floor < 1 / num_sources.")
    if calibration.mean.shape != (num_sources,) or calibration.std.shape != (num_sources,):
        raise ValueError("calibration must contain one mean and std per source.")
    if not torch.isfinite(calibration.mean).all() or not torch.isfinite(calibration.std).all():
        raise ValueError("calibration contains non-finite values.")
    if torch.any(calibration.std <= 0):
        raise ValueError("calibration standard deviations must be positive.")

    mean = calibration.mean.to(normalized_entropies)
    std = calibration.std.to(normalized_entropies)
    calibrated = (normalized_entropies - mean) / std
    if z_clip > 0:
        calibrated = calibrated.clamp(-z_clip, z_clip)
    soft_weights = torch.softmax(-calibrated / temperature, dim=-1)
    weights = weight_floor + (1.0 - num_sources * weight_floor) * soft_weights
    return calibrated, weights


def calibrated_consensus(
    source_scores: torch.Tensor,
    normalized_entropies: torch.Tensor,
    calibration: EntropyCalibration,
    *,
    temperature: float,
    weight_floor: float,
    z_clip: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ``C_S(x,y)`` and return consensus, weights, and calibrated entropy."""

    _check_matrix("source_scores", source_scores)
    if normalized_entropies.shape != source_scores.shape:
        raise ValueError("source_scores and normalized_entropies must have the same shape.")
    calibrated, weights = calibrated_entropy_weights(
        normalized_entropies,
        calibration,
        temperature=temperature,
        weight_floor=weight_floor,
        z_clip=z_clip,
    )
    consensus = (weights * source_scores).sum(dim=-1)
    return consensus, weights, calibrated


def mean_consensus(source_scores: torch.Tensor) -> torch.Tensor:
    """Uniform-source ablation used by the mean-consensus experiment."""

    _check_matrix("source_scores", source_scores)
    return source_scores.mean(dim=-1)


def align_prompt_baselines(
    prompt_ids: Sequence[Hashable],
    initial_policy_scores: torch.Tensor,
    source_consensus: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the prompt-level shift from Eq. (13).

    For every prompt ``x``, this computes
    ``delta_BS = mean_y ell_pi0(x,y) - mean_y C_S(x,y)`` and returns
    ``C_tilde = C_S + delta_BS``. The shift is detached because all inputs are
    offline statistics.
    """

    if len(prompt_ids) != initial_policy_scores.numel():
        raise ValueError("prompt_ids and score tensors must have the same length.")
    if initial_policy_scores.shape != source_consensus.shape or initial_policy_scores.ndim != 1:
        raise ValueError("initial_policy_scores and source_consensus must be one-dimensional and aligned.")
    if not torch.isfinite(initial_policy_scores).all() or not torch.isfinite(source_consensus).all():
        raise ValueError("baseline inputs contain non-finite values.")

    groups: dict[Hashable, list[int]] = {}
    for index, prompt_id in enumerate(prompt_ids):
        groups.setdefault(prompt_id, []).append(index)

    shift = torch.empty_like(source_consensus)
    for indices in groups.values():
        index = torch.tensor(indices, device=source_consensus.device)
        group_shift = (
            initial_policy_scores.index_select(0, index).mean()
            - source_consensus.index_select(0, index).mean()
        )
        shift.index_fill_(0, index, group_shift)
    return (source_consensus + shift).detach(), shift.detach()

