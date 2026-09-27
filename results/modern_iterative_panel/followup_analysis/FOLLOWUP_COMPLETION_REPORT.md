# E17C and final modern recursive-panel completion report

## Outcome

E17C is classified as `matched_horizon_restoration`. Matching the recursive
attention model's training horizon to 32 steps reduced the mean terminal
T=8-to-32 loss from 2.42 +/- 0.29 pp in E17 to 0.04 +/- 0.17 pp in E17C. The
paired interaction is 2.38 +/- 0.43 pp, is positive for all three fresh seeds,
and no significance test is used at n=3.

E16 and E17 remain locked and unchanged. All follow-up robustness analyses use
existing checkpoints. No gain sweep, architecture change, threshold change,
probe tuning, extra seed, or new dataset was introduced.

## E17C evaluation

### Headline seed-level control

| Seed | Terminal T=8 | Terminal T=32 | Drop | Early T=32 | Best step | Peak gap | Probe drop |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 79.18% | 79.33% | -0.15 pp | 78.46% | 23 | 0.27 pp | +0.28 pp |
| 102 | 79.34% | 79.24% | +0.10 pp | 78.91% | 14 | 0.36 pp | +0.47 pp |
| 103 | 79.09% | 78.91% | +0.18 pp | 78.36% | 12 | 0.34 pp | +0.30 pp |

Mean terminal T=8 is 79.20 +/- 0.13%; mean terminal T=32 is
79.16 +/- 0.22%. Every endpoint is within 0.36 pp of its best fixed timestep.
The mean fixed-probe drop is 0.35 +/- 0.10 pp. All states and logits are finite
and materially above chance.

### Frozen readouts

The following are the exact three-seed aggregates saved by the E17C readout
evaluator:

| T_test | Terminal | Early 1--8 | Uniform | Adaptive | GRACE |
|---:|---:|---:|---:|---:|---:|
| 8 | 79.19 +/- 0.11% | 78.58 +/- 0.29% | 78.58 +/- 0.29% | 76.81 +/- 0.33% | 57.47 +/- 3.94% |
| 16 | 79.36 +/- 0.19% | 78.58 +/- 0.29% | 79.40 +/- 0.16% | 76.81 +/- 0.33% | 57.47 +/- 3.94% |
| 24 | 79.32 +/- 0.37% | 78.58 +/- 0.29% | 79.41 +/- 0.26% | 76.81 +/- 0.33% | 57.47 +/- 3.94% |
| 32 | 79.15 +/- 0.22% | 78.58 +/- 0.29% | 79.41 +/- 0.24% | 76.81 +/- 0.33% | 57.47 +/- 3.94% |

GRACE and adaptive stopping were applied with their frozen submitted
parameters and were not tuned for this architecture. Their lower accuracies are
retained as negative readout results; they do not determine the matched-horizon
terminal classification.

The mean endpoint relative residual decreases from 0.180568 at step 8 to
0.034775 at step 32. Per-timestep accuracy, cross-entropy, class-token norm,
relative residual, prediction entropy, class-change fraction, non-finite counts,
and all per-seed readouts are retained under
`results/modern_iterative_panel/attention_t32_control/seed_*/`.

## E17/E17C causal aggregation

| Seed | E17 T8 | E17 T32 | E17 drop | E17C T8 | E17C T32 | E17C drop | Interaction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 79.53% | 76.83% | +2.70 pp | 79.18% | 79.33% | -0.15 pp | +2.85 pp |
| 102 | 79.61% | 77.49% | +2.12 pp | 79.34% | 79.24% | +0.10 pp | +2.02 pp |
| 103 | 79.72% | 77.27% | +2.45 pp | 79.09% | 78.91% | +0.18 pp | +2.27 pp |

All frozen causal criteria pass: every interaction is positive; the control
drop is smaller in every seed; every control endpoint is within 0.5 pp of its
best fixed timestep; fixed-probe loss is smaller in every seed; and all
finite-state checks pass.

## Final four-condition panel

| Model | T_train | Terminal T=8 | Terminal T=32 | Drop | Early T=32 | Best step | Probe drop | Classification |
|---|---:|---:|---:|---:|---:|---|---:|---|
| Recursive SwiGLU | 8 | 80.87 +/- 0.09% | 75.76 +/- 0.96% | +5.11 +/- 0.94 pp | 81.22 +/- 0.05% | 4/4/5 | +2.23 +/- 0.17 pp | `endpoint_invalidity` |
| Recursive SwiGLU | 32 | 80.43 +/- 0.05% | 80.84 +/- 0.15% | -0.42 +/- 0.18 pp | 79.72 +/- 0.14% | 17/24/22 | -0.24 +/- 0.08 pp | `matched_horizon_restoration` |
| Recursive attention | 8 | 79.62 +/- 0.10% | 77.20 +/- 0.34% | +2.42 +/- 0.29 pp | 79.88 +/- 0.11% | 4/5/4 | +1.28 +/- 0.08 pp | `endpoint_invalidity` |
| Recursive attention | 32 | 79.20 +/- 0.13% | 79.16 +/- 0.22% | +0.04 +/- 0.17 pp | 78.58 +/- 0.29% | 23/14/12 | +0.35 +/- 0.10 pp | `matched_horizon_restoration` |

The paired SwiGLU interaction is 5.53 +/- 0.92 pp. The paired attention
interaction is 2.38 +/- 0.43 pp. The same causal pattern therefore appears in
both residual SwiGLU and shared-attention recursion.

## Classwise robustness

All 200 Tiny ImageNet classes are retained.

| Model | Median class drop | IQR | Macro mean | Lower at T=32 | Drop >=2 pp |
|---|---:|---:|---:|---:|---:|
| E16 T_train=8 | 2.67 pp | [0.00, 8.67] pp | 5.11 pp | 144/200 (72.0%) | 124/200 (62.0%) |
| E17 T_train=8 | 2.00 pp | [0.67, 4.00] pp | 2.42 pp | 153/200 (76.5%) | 105/200 (52.5%) |

The loss is distributed broadly across classes rather than being driven by a
small selected subset.

## Checkpoint sensitivity

Every audited saved checkpoint has a positive T=8-minus-T=32 terminal drop.

- E16 selected checkpoints: 4.27--6.13 pp. Final-epoch checkpoints:
  0.59--0.68 pp. The selected early checkpoints are more horizon-sensitive,
  so the E16 effect magnitude is checkpoint-age sensitive even though its sign
  is consistent across the saved selected and final states. Intermediate
  top-three E16 states were not retained.
- E17 selected checkpoints: 2.12--2.70 pp. Across the top three saved improving
  checkpoints: 2.12--3.49 pp. The selected E17 checkpoint is not unusually
  sensitive; it has the smallest saved drop in each seed. Final-epoch E17 model
  states were not retained.

The preregistered selected checkpoints remain the headline checkpoints.

## Fixed-probe mechanism audit

| Model/condition | Step-8 probe | Early-8 probe | Step-32 probe | Step-8 minus step-32 |
|---|---:|---:|---:|---:|
| E16 T_train=8 | 81.31% mean | 81.48% mean | 79.08% mean | +2.23 +/- 0.17 pp |
| E16 T_train=32 | 81.12% mean | 80.75% mean | 81.36% mean | -0.24 +/- 0.08 pp |
| E17 T_train=8 | 79.59% mean | 79.79% mean | 78.31% mean | +1.28 +/- 0.08 pp |
| E17C T_train=32 | 79.33% mean | 79.39% mean | 78.98% mean | +0.35 +/- 0.10 pp |

For both eight-step-trained architectures, the shared head and fixed probe
degrade, supporting representation deterioration plus readout degradation.
Matched-horizon training removes the strong probe loss.

## Model and data scale

| Item | E16 SwiGLU | E17 Attention |
|---|---:|---:|
| Dataset | Tiny ImageNet-200 | Tiny ImageNet-200 |
| Train/validation/classes | 100,000 / 10,000 / 200 | 100,000 / 10,000 / 200 |
| Frozen backbone | ConvNeXt-Tiny | ConvNeXt-Tiny |
| Input representation | pooled 768-d | 49 x 768 spatial tokens |
| Recursive width / intermediate width | 512 / 2048 | 256 / 1024 |
| Attention heads | N/A | 8 |
| Trainable head parameters | 3,648,712 | 1,301,960 |
| Executed frozen-backbone parameters | 27,820,128 | 27,820,128 |
| Executed total system parameters | 31,468,840 | 29,122,088 |
| Trained-horizon terminal accuracy, T_train=8 / 32 | 80.87 +/- 0.09% / 80.84 +/- 0.15% | 79.62 +/- 0.10% / 79.16 +/- 0.22% |
| Dominant recurrent FLOPs per step | 6,291,456 | 107,517,600 |

One multiply-add is counted as two FLOPs. SwiGLU uses `6*d*m`. Attention uses
`8*n*d^2 + 4*n^2*d + 5*h*n^2 + 6*n*d*m` with 50 recurrent tokens. These
recurrent-step formulas exclude the frozen backbone, one-time input projection,
final classifier, LayerNorm, and lower-order elementwise costs.

## Measured readout cost

The existing E14 benchmark directly answers the compute/memory concern:

| Readout | Full inference | Readout only | Peak full memory | Can halt early |
|---|---:|---:|---:|---|
| Terminal | 6.727 ms | 0.113 ms | 8.00 MiB | No |
| Early K=8 | 1.803 ms | 0.135 ms | 11.50 MiB | Yes |
| Adaptive | 13.840 ms | 0.208 ms | 9.01 MiB | Yes |
| Uniform | 6.889 ms | 0.117 ms | 8.50 MiB | No |
| GRACE naive | 6.948 ms | 0.317 ms | 48.56 MiB | No |
| GRACE streaming | 9.681 ms | 3.653 ms | 8.51 MiB | No |

Streaming GRACE nearly removes the naive trajectory-storage memory cost, at the
price of slower eager execution in this implementation.

## Provenance and deviations

The provenance audit passes. Frozen scientific seeds are 101, 102, and 103.
All selected-checkpoint payloads match their frozen configuration hashes and
seed identifiers. No post-result architecture, threshold, horizon, gain,
readout, probe, or checkpoint-selection rule changed.

Recorded deviations:

1. Windows newline normalisation changed a copied JSON's byte hash; validation
   was repaired before data extraction or scientific execution, with no config
   value changed.
2. The E17C seed-102 runner pipe closed after epoch 8. The incomplete attempt
   was preserved and a clean deterministic restart used the identical frozen
   seed/config because an exact optimizer/RNG resume was unavailable.
3. The first E17C seed-103 attempt hit a command timeout during epoch 30. It was
   preserved and cleanly restarted for the same reason.
4. One system-Python invocation stopped immediately because matplotlib was not
   installed; it produced no scientific artifact.
5. Initial post-hoc classwise launcher attempts ended before producing an
   artifact.
6. The first classwise reproduction guard detected two-to-four BF16 boundary
   predictions caused by separate 2-D classifier GEMMs. The analysis driver was
   aligned to the frozen evaluator's full-trajectory GEMM; exact headline
   reproduction then passed before predictions or aggregates were saved.

## Rebuttal outputs and DEQ decision

- Compressed response: 590 characters.
- Medium response: 1,077 characters.
- Full technical response: 1,992 characters.
- Sentence-to-concern mapping: included with the response variants.
- Optional DEQ: **skipped**. Protected EP compute and the required at-least
  30-hour remaining response budget could not be verified, so the frozen start
  conditions did not all pass.

## Artifact index

- E17C report: `results/modern_iterative_panel/attention_t32_control/ATTENTION_RUN_REPORT.md`
- E17C aggregate: `results/modern_iterative_panel/attention_t32_control/attention_summary.json`
- E17C per-seed artifacts: `results/modern_iterative_panel/attention_t32_control/seed_*/`
- Paired causal result: `results/modern_iterative_panel/followup_analysis/e17_e17c_causal_aggregation.{json,md,csv}`
- Final panel: `results/modern_iterative_panel/final_architecture_panel.{json,md,csv}`
- Figures: `results/modern_iterative_panel/final_panel_figure_{1,2,3}.{png,svg}`
- Classwise summary: `results/modern_iterative_panel/followup_analysis/audits/classwise_robustness.{json,md}`
- Complete class tables: `results/modern_iterative_panel/followup_analysis/audits/classwise_e16_t8.csv` and `classwise_e17_t8.csv`
- Checkpoint audit: `results/modern_iterative_panel/followup_analysis/audits/checkpoint_sensitivity.{json,md,csv}`
- E16 probe audit: `results/modern_iterative_panel/followup_analysis/audits/e16_probe_audit.{json,md,csv}`
- Scale audit: `results/modern_iterative_panel/followup_analysis/audits/model_scale.{json,md,csv}`
- Provenance: `results/modern_iterative_panel/followup_analysis/audits/provenance_audit.{json,md}`
- Rebuttal variants: `results/modern_iterative_panel/followup_analysis/rebuttal_responses.{json,md}`
- Measured cost response: `Rebuttal/Terminal_Failure_Measured_Cost_Response.md`
- Final hash manifest: `results/modern_iterative_panel/followup_analysis/final_analysis_manifest.json`
