# e17_recursive_attention_tinyimagenet_ttrain8 run report

**Classification:** endpoint_invalidity.

All three frozen seeds completed. Evaluation used gain 1.0 and only test horizons 8, 16, 24, and 32.

## Seed-level headline results

| Seed | Terminal T=8 | Terminal T=32 | Drop | Early T=32 | Best step | Probe drop |
|---:|---:|---:|---:|---:|---:|---:|
| 101 | 79.53% | 76.83% | +2.70 pp | 79.77% | 4 | +1.25 pp |
| 102 | 79.61% | 77.49% | +2.12 pp | 79.88% | 5 | +1.37 pp |
| 103 | 79.72% | 77.27% | +2.45 pp | 79.98% | 4 | +1.22 pp |

Mean terminal drop: +2.42 +/- 0.29 pp.

Mean fixed-probe terminal drop: +1.28 +/- 0.08 pp.

Mean endpoint residual changed from 0.154636 at step 8 to 0.035364 at step 32.

## Frozen category audit

- Endpoint criterion `mean_drop_at_least_2pp`: True.
- Endpoint criterion `drop_positive_every_seed`: True.
- Endpoint criterion `mean_early_within_1pp_of_trained_terminal`: True.
- Endpoint criterion `finite_and_material`: True.
- Endpoint criterion `probe_mean_directional_agreement`: True.
- Stable criterion `mean_drop_below_0_5pp`: False.
- Stable criterion `mean_peak_gap_below_0_5pp`: False.
- Stable criterion `finite_norms_and_residuals`: True.
- Stable criterion `endpoint_residual_no_greater_than_step8`: True.
- Stable criterion `finite_and_material`: True.

**Matched-control trigger:** PASS.
- Trigger `mean_terminal_drop_at_least_2pp`: True.
- Trigger `terminal_drop_positive_every_seed`: True.
- Trigger `mean_early_within_1pp_of_trained_terminal`: True.

All state/logit finite checks, per-timestep metrics, readout metrics, probes, runtimes, and individual checkpoints are retained.

No gain sweep or architecture-specific readout tuning was run.
