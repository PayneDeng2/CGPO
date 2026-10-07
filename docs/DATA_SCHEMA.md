# Data schema

CGPO uses independently labeled rows. Responses with the same prompt do not
need to be arranged into chosen/rejected pairs.

## Raw rows

```json
{
  "prompt_id": "example-001",
  "prompt": "...",
  "completion": "...",
  "label": "D",
  "source_allocation_weight": 1.0
}
```

`label` is `D` for desirable and `U` for undesirable. It may come from a
verifier, executable checker, reward model, or another response-level judge.

`source_allocation_weight` is a finite, non-negative value produced by the
source-informed data-construction pipeline. Within each label, its normalized
value defines the empirical sampling probability. It is separate from the
source consensus used as a reward reference. The appendix reference-assignment
variants retain this field unchanged. A source-agnostic dataset used by
`No Source Guidance` may omit it because that control samples uniformly.

## Scored rows

`scripts/score_models.py` adds:

- `initial_logp`: the initial-policy completion score;
- `source_logps`: one completion score per source;
- `source_entropies`: one normalized predictive entropy per source;
- `source_models`: the ordered source identifiers shared by those two arrays;
- `anchor_completion` and `anchor_initial_logp`: a deterministic completion
  drawn from a different prompt and used to estimate the detached policy
  anchor.

Each model applies its own tokenizer to the same raw prompt and completion.
Only sequence-level scores are exchanged, so vocabulary alignment is not
required. The calibration and training files must record the same source order;
the consensus builder validates this before fitting or applying statistics.

## Training rows

`scripts/build_consensus.py` additionally writes:

- `source_consensus`: calibrated source consensus;
- `baseline_shift`: the initial-policy mean minus the source-consensus mean;
- `aligned_consensus`: source consensus plus the prompt shift;
- `source_weights` and `calibrated_entropies` for inspection.

All these fields are offline constants. Training uses separate desirable and
undesirable samplers and detaches the initial-policy, source-consensus, and
anchor quantities from the target-policy graph.

## Prompt candidate sets

Baseline alignment groups rows by `prompt_id`. The rows for one prompt define
the fixed candidate set used to estimate both means. Every prompt should retain
the same candidate population when comparing the full method and the
no-alignment ablation.
