# Experiment configurations

The repository exposes the method changes needed to inspect the main model and
the supplementary experiments. Hardware launch details and generated artifacts
are kept separate from the mathematical implementation.

## Main configuration

`configs/main.yaml` records the target backbone, source model families, and the
method hyperparameters stated in the paper. The frozen initial policy is the
corresponding SFT checkpoint and is supplied explicitly through `--model`; it
is not confused with the unmodified target evaluated in the main table. A run
uses calibrated entropy consensus,
prompt-level baseline alignment, label-conditional references, and the
D-first/U-warmup curriculum.

## Component ablations

| Paper setting | Configuration change |
|---|---|
| Mean consensus | `configs/ablations/mean_consensus.yaml` |
| No baseline shift | `configs/ablations/no_baseline_alignment.yaml` |
| No D/U curriculum | `configs/ablations/no_curriculum.yaml` |
| No Source Guidance | `configs/ablations/no_source_guidance.yaml` |

For example, the no-curriculum objective can be inspected with:

```bash
python scripts/train.py \
  --config configs/main.yaml \
  --override-config configs/ablations/no_curriculum.yaml \
  --data DATA.jsonl --output-dir OUTPUT
```

Consensus ablations are applied during `scripts/build_consensus.py`: use
`--aggregation mean` or `--no-baseline-alignment` as indicated by the YAML.

## Reference-assignment variants

Appendix B.5/C.1 is represented by `policy_anchored.yaml`,
`consensus_anchored.yaml`, and `label_conditional.yaml`. They change only the
branch reference assignment. Source-informed rows, class-conditional sampling
distributions, label weights, curriculum, and the anchor estimator remain fixed
in this controlled comparison.

Policy-Anchored retains source-informed example allocation, while both reward
branches are measured against the frozen initial policy. By contrast,
No Source Guidance is an end-to-end control: it uses a separately constructed
source-agnostic dataset, samples uniformly within each label, and uses the
initial policy for both references. The two experiments therefore answer
different questions.

An initial-policy-only scored file for the source-free control can be produced
without `--source-model`, after constructing the source-agnostic raw rows:

```bash
python scripts/score_models.py \
  --input data/source_agnostic.jsonl \
  --output data/source_agnostic_scored.jsonl \
  --initial-model INITIAL_MODEL

python scripts/train.py \
  --config configs/main.yaml \
  --override-config configs/ablations/no_source_guidance.yaml \
  --data data/source_agnostic_scored.jsonl \
  --model INITIAL_MODEL --output-dir outputs/no_source_guidance
```

## Target-model scale variants

Appendix C.2 changes the target backbone through
`configs/scales/phi3_small.yaml` and `configs/scales/phi3_medium.yaml`. The
consensus and objective code are unchanged. Apply either file as a training
overlay, construct its SFT initial policy, and pass that same checkpoint to the
offline initial-policy scorer and `scripts/train.py`.

## Source-pool size

Figure 3(b) changes only the source pool. Run `scripts/score_models.py` with an
ordered subset of the four `models.sources` entries in `configs/main.yaml`,
then rebuild the consensus and train with the unchanged main objective. This
keeps the target, labels, calibration rule, and training loss fixed while
varying which source scores are available. Exact experiment-specific subsets
should be recorded with the generated run metadata rather than embedded in the
objective implementation.

## Benchmark aggregation

The paper evaluates GSM8K, MATH, TheoremQA, BBH, MMLU, IFEval, and MBPP.
Their prompting, metric, and inference settings are recorded in
`configs/evaluation.yaml`.
Category averages are unweighted within Math and General Reasoning. Overall Avg.
is the unweighted mean of all seven benchmark scores. It is not the mean of the
two category averages.
