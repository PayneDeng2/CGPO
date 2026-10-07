"""Core components for Consensus-Guided Preference Optimization."""

from .consensus import (
    EntropyCalibration,
    align_prompt_baselines,
    calibrated_consensus,
    calibrated_entropy_weights,
    fit_entropy_calibration,
)
from .objective import (
    CGPOConfig,
    CGPOOutput,
    ReferenceAssignment,
    cgpo_loss,
    reference_scores,
)
from .scoring import (
    completion_scores_from_logits,
    make_mismatched_anchor_completions,
    render_prompt,
    tokenize_prompt_completion_batch,
)

__all__ = [
    "CGPOConfig",
    "CGPOOutput",
    "EntropyCalibration",
    "ReferenceAssignment",
    "align_prompt_baselines",
    "calibrated_consensus",
    "calibrated_entropy_weights",
    "cgpo_loss",
    "completion_scores_from_logits",
    "fit_entropy_calibration",
    "make_mismatched_anchor_completions",
    "reference_scores",
    "render_prompt",
    "tokenize_prompt_completion_batch",
]
