# E17/E17C paired causal aggregation

**Classification:** `matched_horizon_restoration`.

| Seed | E17 T8 | E17 T32 | E17 drop | E17C T8 | E17C T32 | E17C drop | Interaction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 79.53% | 76.83% | +2.70 pp | 79.18% | 79.33% | -0.15 pp | +2.85 pp |
| 102 | 79.61% | 77.49% | +2.12 pp | 79.34% | 79.24% | +0.10 pp | +2.02 pp |
| 103 | 79.72% | 77.27% | +2.45 pp | 79.09% | 78.91% | +0.18 pp | +2.27 pp |

Mean paired interaction: **2.38 +/- 0.43 pp**. No significance test is used at n=3.

## Frozen criterion audit

- `e17c_terminal_within_0_5pp_best_fixed_every_seed`: True.
- `mean_interaction_at_least_1pp`: True.
- `e17c_drop_smaller_every_seed`: True.
- `interaction_positive_every_seed`: True.
- `all_states_and_logits_finite`: True.
- `e17c_probe_drop_smaller_every_seed`: True.
