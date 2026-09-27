# Final four-condition architecture panel

Means and sample standard deviations use paired seeds 101--103. No significance test is used.

| Model | T_train | Terminal T=8 | Terminal T=32 | Drop | Early T=32 | Best step | Probe drop | Classification |
|---|---:|---:|---:|---:|---:|---|---:|---|
| Recursive SwiGLU | 8 | 80.87 +/- 0.09% | 75.76 +/- 0.96% | +5.11 +/- 0.94 pp | 81.22 +/- 0.05% | 4/4/5 | +2.23 +/- 0.17 pp | `endpoint_invalidity` |
| Recursive SwiGLU | 32 | 80.43 +/- 0.05% | 80.84 +/- 0.15% | -0.42 +/- 0.18 pp | 79.72 +/- 0.14% | 17/24/22 | -0.24 +/- 0.08 pp | `matched_horizon_restoration` |
| Recursive attention | 8 | 79.62 +/- 0.10% | 77.20 +/- 0.34% | +2.42 +/- 0.29 pp | 79.88 +/- 0.11% | 4/5/4 | +1.28 +/- 0.08 pp | `endpoint_invalidity` |
| Recursive attention | 32 | 79.20 +/- 0.13% | 79.16 +/- 0.22% | +0.04 +/- 0.17 pp | 78.58 +/- 0.29% | 23/14/12 | +0.35 +/- 0.10 pp | `matched_horizon_restoration` |
