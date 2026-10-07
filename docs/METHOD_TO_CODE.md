# Method-to-code map

This document maps the paper's mathematical objects to the released reference
implementation. All model scores are completion-only, length-normalized log
probabilities.

| Paper object | Implementation |
|---|---|
| `ell_p(x,y)` and normalized token entropy | `cgpo.scoring.completion_scores_from_logits` |
| entropy calibration | `cgpo.consensus.fit_entropy_calibration` |
| confidence weights | `cgpo.consensus.calibrated_entropy_weights` |
| source consensus | `cgpo.consensus.calibrated_consensus` |
| prompt baseline shift and aligned consensus | `cgpo.consensus.align_prompt_baselines` |
| source-informed signal allocation | class-conditional sampler in `scripts/train.py` |
| label-conditional reference | `cgpo.objective.reference_scores` |
| D-first/U-warmup | `CGPOConfig.undesirable_coefficient` |
| unified pointwise objective | `cgpo.objective.cgpo_loss` |
| detached policy anchor | anchor block in `scripts/train.py` |

## Reference assignments

`ReferenceAssignment` implements the three definitions from Appendix B.5:

| Name | Desirable reference | Undesirable reference |
|---|---|---|
| `policy_anchored` | initial policy | initial policy |
| `consensus_anchored` | aligned consensus | aligned consensus |
| `label_conditional` | initial policy | aligned consensus |

Reference assignment is independent of source-informed data construction. All
three appendix variants use the same source-informed rows and class-conditional
sampling distributions; `policy_anchored` changes only the two branch
references. The broader `no_source_guidance` control instead uses
source-agnostic rows, uniform class-conditional sampling, and the initial policy
for both branches.

For a configured allocation weight `omega_j`, each label branch samples row
`j` with probability `omega_j / sum_k omega_k`. The objective then averages the
sampled pointwise losses separately within the desirable and undesirable
branches.

## Baseline alignment in label-conditional mode

Baseline alignment is applied while constructing offline source references.
It changes only `aligned_consensus`; it does not alter the desirable initial
policy reference. The training objective therefore implements:

```text
D: policy_logp - initial_policy_logp
U: policy_logp - (source_consensus + prompt_baseline_shift)
```

No runtime guard disables prompt-level alignment in `label_conditional` mode.
The shift is already part of the detached undesirable reference.

## Gradient direction

The objective forms the signed margin

```text
m_z = s_z * (policy_logp - reference_logp - z0)
```

with `s_D = +1` and `s_U = -1`. Gradient descent increases the policy score of
desirable completions and decreases that of undesirable ones. Changing the
reference affects the sigmoid scale while preserving the direction defined by
the verifier label.
