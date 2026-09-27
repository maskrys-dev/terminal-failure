# e17c_recursive_attention_tinyimagenet_ttrain32 run report

**Classification:** matched_horizon_control.

All three frozen seeds completed. Evaluation used gain 1.0 and only test horizons 8, 16, 24, and 32.

## Seed-level headline results

| Seed | Terminal T=8 | Terminal T=32 | Drop | Early T=32 | Best step | Probe drop |
|---:|---:|---:|---:|---:|---:|---:|
| 101 | 79.18% | 79.33% | -0.15 pp | 78.46% | 23 | +0.28 pp |
| 102 | 79.34% | 79.24% | +0.10 pp | 78.91% | 14 | +0.47 pp |
| 103 | 79.09% | 78.91% | +0.18 pp | 78.36% | 12 | +0.30 pp |

Mean terminal drop: +0.04 +/- 0.17 pp.

Mean fixed-probe terminal drop: +0.35 +/- 0.10 pp.

Mean endpoint residual changed from 0.180568 at step 8 to 0.034775 at step 32.

## Frozen category audit

- Endpoint criterion `mean_drop_at_least_2pp`: False.
- Endpoint criterion `drop_positive_every_seed`: False.
- Endpoint criterion `mean_early_within_1pp_of_trained_terminal`: True.
- Endpoint criterion `finite_and_material`: True.
- Endpoint criterion `probe_mean_directional_agreement`: True.
- Stable criterion `mean_drop_below_0_5pp`: True.
- Stable criterion `mean_peak_gap_below_0_5pp`: True.
- Stable criterion `finite_norms_and_residuals`: True.
- Stable criterion `endpoint_residual_no_greater_than_step8`: True.
- Stable criterion `finite_and_material`: True.

All state/logit finite checks, per-timestep metrics, readout metrics, probes, runtimes, and individual checkpoints are retained.

No gain sweep or architecture-specific readout tuning was run.
