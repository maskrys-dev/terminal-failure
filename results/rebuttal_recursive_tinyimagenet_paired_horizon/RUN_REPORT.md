# E16 Paired Inference-Horizon Experiment

**Confirmatory decision:** PASS.

The protocol, paired seeds, evaluation horizons, gain, readout, and criteria were frozen before execution. All three seed pairs were run.

## Seed-level results

| Seed | T8 train: A(8) | T8 train: A(32) | T8 drop | T32 train: A(8) | T32 train: A(32) | T32 drop | Interaction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 80.98% | 76.05% | +4.93 pp | 80.39% | 81.01% | -0.62 pp | +5.55 pp |
| 102 | 80.82% | 74.69% | +6.13 pp | 80.48% | 80.78% | -0.30 pp | +6.43 pp |
| 103 | 80.82% | 76.55% | +4.27 pp | 80.41% | 80.74% | -0.33 pp | +4.60 pp |

## Aggregate endpoint result

- T_train=8: 80.87% +/- 0.09% at T_test=8 versus 75.76% +/- 0.96% at T_test=32; drop 5.11 +/- 0.94 pp.
- T_train=32: 80.43% +/- 0.05% at T_test=8 versus 80.84% +/- 0.15% at T_test=32; drop -0.42 +/- 0.18 pp.
- Paired interaction: 5.53 +/- 0.92 pp (seed values 5.55, 6.43, 4.60 pp).

Values are means +/- sample standard deviations across the three fresh paired seeds; no significance test is used.

## Mean trajectory diagnostics

### T_train=8

| T_test | Terminal accuracy | Early-eight accuracy | Terminal - early | Best fixed timesteps (seeds 101/102/103) | Endpoint relative residual |
|---:|---:|---:|---:|:---:|---:|
| 8 | 80.87% +/- 0.09% | 81.22% +/- 0.05% | -0.35 +/- 0.09 pp | 4/4/5 | 0.08650 +/- 0.00729 |
| 16 | 79.30% +/- 0.26% | 81.22% +/- 0.05% | -1.92 +/- 0.30 pp | 4/4/5 | 0.04860 +/- 0.00326 |
| 24 | 77.46% +/- 0.50% | 81.22% +/- 0.05% | -3.77 +/- 0.53 pp | 4/4/5 | 0.03386 +/- 0.00211 |
| 32 | 75.76% +/- 0.96% | 81.22% +/- 0.05% | -5.46 +/- 1.00 pp | 4/4/5 | 0.02624 +/- 0.00155 |

### T_train=32

| T_test | Terminal accuracy | Early-eight accuracy | Terminal - early | Best fixed timesteps (seeds 101/102/103) | Endpoint relative residual |
|---:|---:|---:|---:|:---:|---:|
| 8 | 80.43% +/- 0.05% | 79.72% +/- 0.14% | +0.71 +/- 0.14 pp | 8/8/8 | 0.03336 +/- 0.00035 |
| 16 | 81.07% +/- 0.16% | 79.72% +/- 0.14% | +1.35 +/- 0.22 pp | 16/12/16 | 0.02834 +/- 0.00023 |
| 24 | 81.10% +/- 0.07% | 79.72% +/- 0.14% | +1.38 +/- 0.17 pp | 17/24/22 | 0.02350 +/- 0.00016 |
| 32 | 80.84% +/- 0.15% | 79.72% +/- 0.14% | +1.13 +/- 0.12 pp | 17/24/22 | 0.01970 +/- 0.00013 |

## Criterion audit

- **PASS - mean T_train=8 terminal drop:** 5.11 pp >= 2.00 pp.
- **PASS - early-eight stability:** mean and maximum seed drop were both 0.00 pp <= 1.00 pp.
- **PASS - sign replication:** T_train=8 terminal drops were 4.93, 6.13, 4.27 pp.
- **PASS - aligned control endpoint:** mean terminal deficit to the best fixed timestep was 0.33 pp; the largest seed deficit was 0.49 pp <= 0.50 pp.
- **PASS - paired interaction:** mean 5.53 pp >= 1.00 pp and all seed values were positive.
- **PASS - numerical checks:** all state and logit values were finite and every reported accuracy was materially above chance.

Full timestep accuracies, terminal-minus-early gaps, best fixed timesteps, relative residual trajectories, state norms, and finite-state counts are retained in each seed's horizon metrics.

E16 evaluates only gain 1.0. No gain sweep, GRACE/probe comparison, or post-result threshold change was performed.

## Frozen artifact hashes

- Aggregate config: `e637a6854ac8b990d5a4822b4e7beece1b88226839b88360a86b84c89b60b605`
- T_train=8 config: `6bdb5bdcc994f8e7f145042f6f2e5f5b85b0cb73c99dbeb740e13ead80084741`
- T_train=32 config: `33a36523d381d7996f5e4f11a5d3e129f9d90542085227a3f7b8932e4c3f7bac`
- Protocol: `9cc8f8204b2fb92cb9e42c0f73ba172a3ba994264d4fe80596d35f00c30c9c28`
