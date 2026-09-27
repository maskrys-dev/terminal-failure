# E14 Seed-0 Run Report

## Decision

**The predeclared scientific continuation gate failed. Seeds 1 and 2 were not
run, and the model, grid, windows, thresholds, and probe regularisation were
not tuned after the result.**

Frozen configuration SHA-256:
`093e28a08fbc0d90f666bea8d0107b8881da2f2a448a1f2cd200647e4a9a316e`.

The recursive implementation passed its engineering gate, so the DEQ
engineering fallback is not activated. This is a valid scientific pilot that
fails the stricter transition/ordinary-regime criterion.

## Setup and engineering gate

- Tiny ImageNet validation: 200 classes, 100,000 train images and 10,000
  labelled validation images, with exact balanced class counts.
- Frozen ConvNeXt-Tiny cache: 768 dimensions, 323 MiB, all finite.
- Runtime: Python 3.12.10, PyTorch 2.13.0+cu130, BF16, RTX 5080 Laptop GPU.
- Smoke test: training loss decreased from 5.3667 to 3.4807 in three epochs;
  validation accuracy reached 50.15%, versus the predeclared 2.5% minimum.
- Scientific seed-0 checkpoint: epoch 3, selected only by terminal validation
  accuracy at the training condition `(T=8, lambda=1.0)`, 80.79%.
- Direct fixed-alpha probe on frozen ConvNeXt features: 80.52%.

## Seed-0 shared-head results

All reported trajectory readouts use `T=32`; the early readout averages states
1--8.

| Gain | Terminal | Early | Early - terminal | GRACE | Fixed GRACE profile |
|---:|---:|---:|---:|---:|---:|
| 0.75 | 77.72% | 80.76% | +3.04 pp | 80.70% | 80.36% |
| 1.00 | 75.92% | 80.85% | +4.93 pp | 80.70% | 80.39% |
| 1.10 | 75.09% | 80.89% | +5.80 pp | 80.69% | 80.56% |
| 1.20 | 74.47% | 80.78% | +6.31 pp | 80.72% | 80.59% |
| 1.30 | 73.77% | 80.82% | +7.05 pp | 80.68% | 80.68% |
| 1.40 | 73.08% | 80.87% | +7.79 pp | 80.71% | 80.78% |
| 1.50 | 72.44% | 80.83% | +8.39 pp | 80.69% | 80.80% |
| 1.60 | 71.86% | 80.82% | +8.96 pp | 80.76% | 80.81% |

The fixed-alpha representation probes show the same direction, with
early-minus-terminal increasing from +1.40 pp at gain 0.75 to +3.07 pp at
gain 1.60.

All states and logits were finite. Shared-head timestep accuracy is
non-monotonic at every gain, and its best timestep moves from step 6 at gain
0.75 to step 3 at gain 1.60.

## Why the gate failed

The prefix gap passes the size and consecutive-gain requirement, the
trajectory is finite and non-monotonic, and performance remains far above
chance. The stable-regime condition fails:

- shared head at gain 1.00: terminal is 4.93 pp below early;
- fixed probe at gain 1.00: terminal is 2.07 pp below early;
- the permitted stable deficit is only 0.5 pp.

Thus the endpoint-prefix mismatch is already present at the nominal
32-step reference rather than emerging only under the intended gain stress.
This pilot cannot support the predeclared claim that ordinary-regime terminal
performance remains competitive. It should not be promoted as the planned
three-seed rebuttal result.

The absolute probe grace-period thresholds 0.40, 0.50, and 0.60 all remain at
the full 32-step horizon. They are too low to resolve the observed high-
accuracy deterioration, but they were frozen before the pilot and were not
changed afterward.

## Preserved horizon-overcomputation study

E14 is retained as a separate horizon-extension result at fixed gain
\(\lambda=1\), using the already saved per-timestep curves rather than a new
or tuned run:

| Test horizon | Shared-head terminal | Change from T=8 | Fixed probe | Probe change |
|---:|---:|---:|---:|---:|
| 8 | 80.82% | +0.00 pp | 81.22% | +0.00 pp |
| 12 | 80.13% | -0.69 pp | 80.71% | -0.51 pp |
| 16 | 79.30% | -1.52 pp | 80.65% | -0.57 pp |
| 24 | 77.83% | -2.99 pp | 79.90% | -1.32 pp |
| 32 | 75.92% | -4.90 pp | 79.33% | -1.89 pp |

This supports the narrower secondary conclusion that, for the E14 model,
terminal accuracy degrades when test-time recurrence extends beyond its
trained eight-step horizon. It does not establish the intended aligned-horizon
gain-stress transition.

## GRACE novelty control

Per-sample GRACE and the fixed global GRACE-profile control are effectively
tied. At gain 1.30 both score 80.68%; at gain 1.60 the fixed profile scores
80.81% and per-sample GRACE scores 80.76%. This setting provides no evidence
that per-sample growth adaptation improves on the matched global temporal
profile.

## Measured cost

One BF16 validation batch with `B=512`, `T=32`, and `d=512`; medians over 25
timed repetitions after 10 warm-ups:

| Readout | Full inference | Readout only | Peak full additional memory |
|---|---:|---:|---:|
| Terminal | 6.727 ms | 0.113 ms | 8.00 MiB |
| Early window | 1.803 ms | 0.135 ms | 11.50 MiB |
| Adaptive stopping | 13.840 ms | 0.208 ms | 9.01 MiB |
| Uniform streaming | 6.889 ms | 0.117 ms | 8.50 MiB |
| GRACE naive | 6.948 ms | 0.317 ms | 48.56 MiB |
| GRACE streaming | 9.681 ms | 3.653 ms | 8.51 MiB |

Streaming GRACE removes the stored-trajectory memory cost but is slower in
this eager implementation. Early-window inference is fastest because it
executes only eight recurrent steps. The adaptive implementation uses
per-sample active-set compaction and is slower at this batch size despite its
ability to halt samples early.

## Primary artifacts

- `gate_decision.json`: machine-readable predeclared decision.
- `seed_0_metrics.json`: shared-head curves, readout scores, diagnostics, and
  all probe results.
- `runtime_memory.json` and `runtime_memory_table.txt`: measured cost.
- `trajectory_accuracy.png`, `readout_baselines.png`, and `stress_grid.png`:
  checked renderings.
- `horizon_extension_audit.json`, `horizon_extension_table.txt`, and
  `horizon_extension_audit.png`: derived horizon-overcomputation audit.
- `seed_0_checkpoint.pt`: frozen best checkpoint.
