import pytest
import torch

from cgpo.objective import (
    CGPOConfig,
    ReferenceAssignment,
    cgpo_loss,
    reference_scores,
)


def test_label_conditional_reference_assignment():
    labels = torch.tensor([True, False, True, False])
    initial = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    consensus = torch.tensor([-5.0, -6.0, -7.0, -8.0])
    references = reference_scores(
        labels,
        initial,
        consensus,
        ReferenceAssignment.LABEL_CONDITIONAL,
    )
    assert references.tolist() == [-1.0, -6.0, -3.0, -8.0]


def test_reference_assignment_does_not_alias_system_ablation():
    with pytest.raises(ValueError):
        ReferenceAssignment.from_name("no_source_guidance")


def test_gradient_descent_promotes_d_and_suppresses_u():
    policy = torch.tensor([0.0, 0.0], requires_grad=True)
    output = cgpo_loss(
        policy,
        torch.zeros(2),
        torch.zeros(2),
        torch.tensor([True, False]),
        z0=0.0,
        step=10,
        config=CGPOConfig(
            beta=1.0,
            undesirable_warmup_steps=0,
            reference_assignment=ReferenceAssignment.LABEL_CONDITIONAL,
        ),
    )
    output.loss.backward()

    assert policy.grad[0] < 0  # gradient descent increases desirable logp
    assert policy.grad[1] > 0  # gradient descent decreases undesirable logp


def test_lower_source_support_increases_undesirable_loss():
    labels = torch.tensor([False])
    policy = torch.tensor([-1.0])
    initial = torch.tensor([-1.0])
    config = CGPOConfig(beta=1.0, undesirable_warmup_steps=0)
    high_support = cgpo_loss(
        policy, initial, torch.tensor([-0.5]), labels, 0.0, step=0, config=config
    )
    low_support = cgpo_loss(
        policy, initial, torch.tensor([-2.0]), labels, 0.0, step=0, config=config
    )
    assert low_support.undesirable_loss > high_support.undesirable_loss


def test_curriculum_is_d_first_then_linear():
    config = CGPOConfig(undesirable_warmup_start=5, undesirable_warmup_steps=10)
    assert config.undesirable_coefficient(4) == 0.0
    assert config.undesirable_coefficient(5) == 0.0
    assert config.undesirable_coefficient(10) == pytest.approx(0.5)
    assert config.undesirable_coefficient(15) == 1.0


def test_baseline_shift_and_label_conditional_assignment_can_coexist():
    labels = torch.tensor([True, False])
    initial = torch.tensor([-1.0, -2.0])
    aligned_consensus = torch.tensor([-3.0, -1.5])
    output = cgpo_loss(
        torch.tensor([-0.9, -1.8]),
        initial,
        aligned_consensus,
        labels,
        z0=0.0,
        step=100,
        config=CGPOConfig(undesirable_warmup_steps=10),
    )
    assert output.references.tolist() == pytest.approx([-1.0, -1.5])
    assert torch.isfinite(output.loss)


def test_objective_uses_separate_class_conditional_means():
    output = cgpo_loss(
        torch.zeros(3),
        torch.zeros(3),
        torch.zeros(3),
        torch.tensor([True, True, False]),
        z0=0.0,
        step=0,
        config=CGPOConfig(
            beta=1.0,
            desirable_weight=1.0,
            undesirable_weight=1.0,
            undesirable_warmup_steps=0,
        ),
    )
    assert output.desirable_loss.item() == pytest.approx(0.5)
    assert output.undesirable_loss.item() == pytest.approx(0.5)
