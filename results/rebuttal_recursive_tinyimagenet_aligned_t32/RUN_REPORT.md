# E15 Aligned-Horizon Seed-0 Run Report

## Decision

**The predeclared E15 continuation gate failed. Seeds 1 and 2 were not run,
and the gain grid, architecture, window, thresholds, and decoder settings were
not changed.**

Frozen configuration SHA-256:
`48e0405e2e2ff59cfa348e3f9958b7699f1ec0f2440ac618e8f0350943f25f8f`.

E15 changed only the training horizon from 8 to 32 and therefore used the
aligned update scale \(\lambda/32\) during both training and evaluation. It
reused the exact validated E14 Tiny ImageNet / frozen ConvNeXt feature cache.

## Engineering and training

- The engineering smoke gate passed: loss decreased from 5.3669 to 3.4819 in
  three epochs and validation accuracy reached 49.90%.
- The seed-0 checkpoint was selected only by terminal validation accuracy at
  `(T=32, lambda=1)`.
- The selected checkpoint is epoch 3 with 80.75% terminal accuracy.
- All evaluated states and logits were finite.
- Direct fixed-alpha probe on the frozen ConvNeXt features: 80.52%.

## Seed-0 shared-head results

| Gain | Terminal | Early (K=8) | Early - terminal | Uniform | GRACE | Fixed profile |
|---:|---:|---:|---:|---:|---:|---:|
| 0.75 | 80.77% | 79.25% | -1.52 pp | 80.63% | 80.50% | 80.06% |
| 1.00 | 80.75% | 79.72% | -1.03 pp | 80.85% | 80.64% | 80.33% |
| 1.10 | 80.62% | 79.77% | -0.85 pp | 80.78% | 80.52% | 80.33% |
| 1.20 | 80.54% | 79.84% | -0.70 pp | 80.77% | 80.54% | 80.39% |
| 1.30 | 80.33% | 79.95% | -0.38 pp | 80.71% | 80.59% | 80.40% |
| 1.40 | 80.15% | 80.07% | -0.08 pp | 80.80% | 80.63% | 80.51% |
| 1.50 | 80.15% | 80.15% | +0.00 pp | 80.77% | 80.67% | 80.70% |
| 1.60 | 79.93% | 80.20% | +0.27 pp | 80.79% | 80.66% | 80.68% |

The fixed-alpha representation probes show the same directional crossover:
early-minus-terminal moves from -0.25 pp at gain 1.20 to +0.34 pp at gain
1.60.

## Gate audit

Passed:

- Stable endpoint: at gain 1.0, terminal is 80.75% and the best
  trajectory-aware readout is uniform at 80.85%, a deficit of only 0.10 pp
  versus the allowed 0.5 pp.
- Stress trend: from gain 1.2 to 1.6, early-minus-terminal increases from
  -0.70 pp to +0.27 pp.
- Best timestep shifts earlier from step 16 at gain 1.2 to step 12 at gain
  1.6.
- All values are finite and accuracy remains materially above chance.

Failed:

- Early-window never exceeds terminal by at least 1 pp at three consecutive
  frozen gains at or above 1.2. Its largest advantage is only +0.27 pp at gain
  1.6.

Therefore E15 cannot support the predeclared multi-seed aligned-horizon stress
claim. It does show that correcting the horizon mismatch restores a
competitive nominal endpoint and that the endpoint-prefix ordering moves in
the expected direction with gain, but the frozen stress range does not
produce the required effect size.

## GRACE interpretation

E15 does not support an accuracy-novelty claim for GRACE. Uniform averaging,
GRACE, and the fixed global profile are all close; at gain 1.6 they score
80.79%, 80.66%, and 80.68%, respectively. The submitted modReLU noise and
variance experiments remain the appropriate evidence for GRACE's distinctive
value.

## Artifacts

- `config_frozen.json`: immutable E15 configuration.
- `gate_decision.json`: machine-readable gate audit.
- `seed_0_training.json` and `seed_0_metrics.json`: complete seed-0 results.
- `trajectory_accuracy.png`, `readout_baselines.png`, and `stress_grid.png`:
  visually checked figures.
- `seed_0_checkpoint.pt`: frozen best checkpoint.
