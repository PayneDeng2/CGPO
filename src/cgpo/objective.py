"""Label-conditional CGPO references and pointwise objective."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class ReferenceAssignment(str, Enum):
    """Reference assignments studied in Appendix B.5 and C.1."""

    POLICY_ANCHORED = "policy_anchored"
    CONSENSUS_ANCHORED = "consensus_anchored"
    LABEL_CONDITIONAL = "label_conditional"

    @classmethod
    def from_name(cls, value: str) -> "ReferenceAssignment":
        return cls(value)


@dataclass(frozen=True)
class CGPOConfig:
    beta: float = 0.08
    desirable_weight: float = 1.2
    undesirable_weight: float = 1.0
    undesirable_warmup_start: int = 0
    undesirable_warmup_steps: int = 100
    reference_assignment: ReferenceAssignment = ReferenceAssignment.LABEL_CONDITIONAL

    def __post_init__(self) -> None:
        if self.beta <= 0:
            raise ValueError("beta must be positive.")
        if self.desirable_weight < 0 or self.undesirable_weight < 0:
            raise ValueError("label weights must be non-negative.")
        if self.undesirable_warmup_start < 0 or self.undesirable_warmup_steps < 0:
            raise ValueError("warmup steps must be non-negative.")

    def undesirable_coefficient(self, step: int) -> float:
        if self.undesirable_warmup_steps == 0:
            return 1.0
        progress = (step - self.undesirable_warmup_start) / self.undesirable_warmup_steps
        return max(0.0, min(1.0, progress))


@dataclass
class CGPOOutput:
    loss: torch.Tensor
    desirable_loss: torch.Tensor
    undesirable_loss: torch.Tensor
    signed_margins: torch.Tensor
    rewards: torch.Tensor
    references: torch.Tensor
    undesirable_coefficient: float


def reference_scores(
    labels: torch.Tensor,
    initial_policy_scores: torch.Tensor,
    aligned_consensus_scores: torch.Tensor,
    assignment: ReferenceAssignment,
) -> torch.Tensor:
    """Select ``b_z(x,y)`` for each label without constructing response pairs."""

    if labels.dtype != torch.bool:
        raise TypeError("labels must be bool, where True denotes desirable.")
    if labels.shape != initial_policy_scores.shape or labels.shape != aligned_consensus_scores.shape:
        raise ValueError("labels and reference scores must have the same shape.")
    if assignment is ReferenceAssignment.POLICY_ANCHORED:
        return initial_policy_scores
    if assignment is ReferenceAssignment.CONSENSUS_ANCHORED:
        return aligned_consensus_scores
    if assignment is ReferenceAssignment.LABEL_CONDITIONAL:
        return torch.where(labels, initial_policy_scores, aligned_consensus_scores)
    raise ValueError(f"Unknown reference assignment: {assignment}")


def _class_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if torch.any(mask):
        return values[mask].mean()
    return values.sum() * 0.0


def cgpo_loss(
    policy_scores: torch.Tensor,
    initial_policy_scores: torch.Tensor,
    aligned_consensus_scores: torch.Tensor,
    labels: torch.Tensor,
    z0: torch.Tensor | float,
    *,
    step: int,
    config: CGPOConfig,
) -> CGPOOutput:
    """Compute Eq. (15) with class-conditional aggregation.

    All scores are completion-only, length-normalized log probabilities.
    ``z0`` and all reference scores are detached from model computation.
    """

    if policy_scores.ndim != 1:
        raise ValueError("score tensors must be one-dimensional.")
    if not (policy_scores.shape == initial_policy_scores.shape == aligned_consensus_scores.shape == labels.shape):
        raise ValueError("all per-example tensors must have the same shape.")
    if labels.dtype != torch.bool:
        raise TypeError("labels must be bool, where True denotes desirable.")

    references = reference_scores(
        labels,
        initial_policy_scores.detach(),
        aligned_consensus_scores.detach(),
        config.reference_assignment,
    )
    rewards = policy_scores - references
    signs = torch.where(labels, torch.ones_like(rewards), -torch.ones_like(rewards))
    z0_tensor = torch.as_tensor(z0, device=rewards.device, dtype=rewards.dtype).detach()
    margins = signs * (rewards - z0_tensor)
    pointwise = 1.0 - torch.sigmoid(config.beta * margins)

    desirable = (
        _class_mean(pointwise, labels) * config.desirable_weight
    )
    undesirable_coefficient = config.undesirable_coefficient(step)
    undesirable = (
        _class_mean(pointwise, ~labels)
        * config.undesirable_weight
        * undesirable_coefficient
    )
    return CGPOOutput(
        loss=desirable + undesirable,
        desirable_loss=desirable,
        undesirable_loss=undesirable,
        signed_margins=margins,
        rewards=rewards,
        references=references,
        undesirable_coefficient=undesirable_coefficient,
    )
