# Anonymous Supplementary Code Package

This archive contains the anonymized code, saved result files, and exact
commands needed to reproduce the main figures and appendix analyses for the
submission on pre-collapse learning regimes in iterative dynamical learners.
The manuscript PDF, written appendices, checklist, LaTeX source, style files,
and compiled manuscript figure assets are intentionally excluded from this
supplementary ZIP; they are submitted as part of the main paper PDF.

All commands below assume the working directory is the root of this extracted
archive.

## Environment

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The experiments use PyTorch and automatically select CUDA when available.
The reported runs were executed on a single `NVIDIA RTX PRO 1000 Blackwell
Generation Laptop GPU` with approximately `8 GB` VRAM.

## Datasets

The archive does not include raw benchmark data.

- `MNIST` and `FashionMNIST` are downloaded automatically by the torchvision-based scripts.
- `CIFAR-10` is also downloaded automatically by `transient_geometry/experiments/e7_cifar10.py`.
- The frozen-feature CIFAR-10 check uses torchvision's ImageNet-pretrained `ResNet-18` weights, downloaded automatically if not already cached.
- By default, datasets are cached under `data/` relative to the project root.
- See `DATASET_NOTES.md` for a compact summary of dataset usage.

## Fastest Path: Regenerate the Paper Figures from Saved Results

The saved JSON files in `results/` are sufficient to regenerate the paper
figures without re-running the expensive sweeps:

```bash
python scripts/generate_figures.py
python results/e7_cifar10/plot_e7.py
```

This recreates the main manuscript figures in `figures/`.

Appendix-style numeric summaries can be regenerated with:

```bash
python gen_appendix_tables.py
```

## Main Experiment Commands

The following commands rerun the principal experiments used in the paper.

```bash
python -m transient_geometry.experiments.overnight_10seed
python transient_geometry/grace_period.py
python -m transient_geometry.experiments.r_revision_readout_audits
python -m transient_geometry.experiments.r_clipping_audit
python -m transient_geometry.experiments.c4_noise_robustness
python -m transient_geometry.experiments.a3_fashion
python -m transient_geometry.experiments.a3_fashion_grace
python -m transient_geometry.experiments.a5_cross_system
python -m transient_geometry.experiments.e6_trained_at_operating_point
python transient_geometry/experiments/e7_cifar10.py
python -W ignore transient_geometry/experiments/e12_cifar10_resnet18_features.py
python -m transient_geometry.experiments.e13_sequential_cifar10_lstm
python -m transient_geometry.experiments.e2_deq_bifurcation
python -m transient_geometry.experiments.e10_operator_diagnostics
python -m transient_geometry.experiments.e11_cocycle_envelope
python transient_geometry/pilot_sweep.py
```

These scripts write fresh outputs into the corresponding subdirectories of
`results/`.

For the optional single-timestep LSTM audit, add
`--per-timestep-audit --audit-lambdas 1.0,1.6,2.0` to the E13 command. This
fits a separate linear probe at each post-input settling timestep for the
selected test-time gains and writes `e13_seq_cifar10_lstm_timestep_audit.*`.

## Script Inventory

The script names preserve legacy internal experiment labels so that saved
result directories and manuscript commands remain stable. In particular,
`d1` is the internal name for the trajectory-weighted readout now called
`GRACE` in the paper. The tables below are exhaustive for the Python scripts
included in this anonymous archive: every `.py` file is listed once, with its
plain-English purpose and whether it is a primary reproduction entry point,
an appendix/diagnostic script, or an imported helper.

### Primary Reproduction Scripts

| Script | What it does | Output |
| --- | --- | --- |
| `scripts/generate_figures.py` | Regenerates the main paper figures from saved JSON results, without rerunning experiments. | `figures/` |
| `results/e7_cifar10/plot_e7.py` | Regenerates the raw-pixel CIFAR-10 figure from the saved E7 JSON result. | `figures/fig_cifar10_regime.*` |
| `gen_appendix_tables.py` | Prints appendix-facing numeric summaries and sensitivity tables from saved results. | Console output |
| `transient_geometry/experiments/overnight_10seed.py` | Publication-quality 10-seed MNIST complex-modReLU sweep comparing final, early-window, adaptive, and GRACE readouts. | `results/overnight_10seed/` |
| `transient_geometry/grace_period.py` | Computes the formal accuracy-defined grace period from per-timestep probe accuracies. | `results/grace_period/` |
| `transient_geometry/experiments/r_revision_readout_audits.py` | Runs appendix readout audits: early-window size sensitivity and reference-radius hyperparameter selection. | `results/revision_readout_audits/` |
| `transient_geometry/experiments/r_clipping_audit.py` | Audits whether the complex-modReLU magnitude clip is active in the early prefix or only late in unstable rollouts. | `results/clipping_audit/` |
| `transient_geometry/experiments/c4_noise_robustness.py` | Tests the readouts under injected dynamical noise in the complex-modReLU reservoir. | `results/c4_noise/` |
| `transient_geometry/experiments/a3_fashion.py` | Repeats the fixed-reservoir readout sweep on FashionMNIST. | `results/a3_fashion/` |
| `transient_geometry/experiments/a3_fashion_grace.py` | Targeted FashionMNIST audit computing the GRACE column for the appendix table without rerunning the slower best-timestep diagnostics. | `results/a3_fashion/grace_results.json` |
| `transient_geometry/experiments/a5_cross_system.py` | Compares the regime across complex modReLU, real tanh ESN, and linear systems. | `results/a5_cross_system/` |
| `transient_geometry/experiments/e6_trained_at_operating_point.py` | Trains LeakyReLU iterative classifiers at short-horizon and at-operating-point conditions, then sweeps test-time gain. | `results/e6_trained_at_op/` |
| `transient_geometry/experiments/e7_cifar10.py` | Runs the raw-pixel CIFAR-10 complex-modReLU reservoir sweep. | `results/e7_cifar10/` |
| `transient_geometry/experiments/e12_cifar10_resnet18_features.py` | Runs the CIFAR-10 frozen ImageNet-pretrained ResNet-18 feature experiment. | `results/e12_cifar10_resnet18/` |
| `transient_geometry/experiments/e13_sequential_cifar10_lstm.py` | Trains a row-sequential CIFAR-10 LSTM and sweeps post-input recurrent gain for the appendix trained-recurrent check; optional `--per-timestep-audit` fits separate probes at every post-input timestep. | `results/e13_seq_cifar10_lstm/` |
| `transient_geometry/experiments/e10_operator_diagnostics.py` | Reconstructs fixed operators and computes spectral/operator diagnostics used in the mechanism appendix. | `results/e10_operator_diagnostics/` |
| `transient_geometry/experiments/e11_cocycle_envelope.py` | Computes finite-horizon cocycle-envelope diagnostics for the real tanh ESN branch. | `results/e11_cocycle_envelope/` |

### Secondary, Legacy, and Diagnostic Experiment Scripts

| Script | What it does | Status |
| --- | --- | --- |
| `transient_geometry/pilot_sweep.py` | Early coarse MNIST phase-diagram sweep used for go/no-go checks before the final 10-seed run. | Diagnostic/pilot |
| `transient_geometry/refined_sweep.py` | Denser transition-region sweep measuring best-timestep accuracy and collapse timing. | Diagnostic/pilot |
| `transient_geometry/experiments/c1_c2_readout.py` | Early version of the final-state, transient-mean, early-window, and adaptive-stopping comparison. | Superseded by `overnight_10seed.py` |
| `transient_geometry/experiments/d1_regime_readout.py` | Early GRACE prototype on the complex-modReLU system. | Superseded by `overnight_10seed.py` and audits |
| `transient_geometry/experiments/d1_cross_system.py` | GRACE hyperparameter check on complex modReLU and real ESN systems. | Appendix/diagnostic |
| `transient_geometry/experiments/d1_esn_tune.py` | Hyperparameter sweep showing how GRACE behaves under bounded tanh ESN dynamics. | Appendix/diagnostic |
| `transient_geometry/experiments/e1_trained_recurrent.py` | Early trained tanh recurrent classifier under test-time gain shift. | Exploratory |
| `transient_geometry/experiments/e2_deq_bifurcation.py` | DEQ-style tanh fixed-point solver experiment near instability. | Appendix/diagnostic |
| `transient_geometry/experiments/e3_trained_complex_modrelu.py` | Trained complex modReLU recurrent classifier using real-valued implementation of complex arithmetic. | Exploratory |
| `transient_geometry/experiments/e4_trained_leakyrelu.py` | Earlier trained LeakyReLU gain-sweep experiment. | Superseded by `e6_trained_at_operating_point.py` |
| `transient_geometry/experiments/e5_deq_leakyrelu.py` | LeakyReLU DEQ-style fixed-point solver experiment. | Exploratory |
| `transient_geometry/experiments/e6_grace_ablation.py` | GRACE hyperparameter ablation using saved E6 per-timestep statistics; does not rerun training. | Appendix/diagnostic |
| `transient_geometry/experiments/e8_sequential_esn.py` | Permuted sequential MNIST ESN experiment with one pixel per timestep. | Exploratory |
| `transient_geometry/experiments/final_stats.py` | Computes statistical tests, effect sizes, confidence intervals, and LaTeX tables from saved results. | Analysis helper |
| `transient_geometry/experiments/pilot_phase_coherence.py` | Pilot alternative stopping signal based on phase-coherence of multiple timestep heads. | Exploratory |

### Library and Small Inspection Helpers

| Script | What it does | Status |
| --- | --- | --- |
| `transient_geometry/system.py` | Defines the fixed dynamical systems: complex modReLU reservoir, real tanh ESN, and linear null. | Imported by experiments |
| `transient_geometry/probes.py` | Defines feature extraction and linear probe helpers for final, transient, timestep, and related readouts. | Imported by experiments |
| `transient_geometry/__init__.py` | Marks `transient_geometry` as a Python package. | Package file |
| `transient_geometry/experiments/__init__.py` | Marks `transient_geometry.experiments` as a Python package. | Package file |
| `calc_bound.py` | Small scratch calculation for the grace-period upper-bound discussion using legacy result paths. | Inspection helper |
| `inspect_d1.py` | Prints keys and sample values from saved GRACE cross-system results. | Inspection helper |
| `print_e6.py` | Prints a compact table from the trained LeakyReLU E6 saved JSON files. | Inspection helper |

## Included Saved Results

The archive includes the saved artifacts needed to regenerate the manuscript
figures and appendix evidence:

- `results/overnight_10seed/results.json`
- `results/grace_period/accuracy_grace_period.json`
- `results/grace_period/grace_period.json`
- `results/revision_readout_audits/readout_audits.json`
- `results/clipping_audit/results.json`
- `results/c4_noise/results.json`
- `results/a3_fashion/results.json`
- `results/a3_fashion/grace_results.json`
- `results/a5_cross_system/results.json`
- `results/d1_cross_system/results.json`
- `results/d1_cross_system/weight_profiles.json`
- `results/d1_esn_tune/results.json`
- `results/e6_trained_at_op/*.json`
- `results/e7_cifar10/e7_cifar10.json`
- `results/e12_cifar10_resnet18/e12_cifar10_resnet18.json`
- `results/e13_seq_cifar10_lstm/e13_seq_cifar10_lstm.json`
- `results/e13_seq_cifar10_lstm/e13_seq_cifar10_lstm_timestep_audit.json`
- `results/e2_deq_bifurcation/e2_deq_bifurcation.json`
- `results/e10_operator_diagnostics/*.json`
- `results/e11_cocycle_envelope/*.json`
- `results/pilot/separability.json`
- `results/pilot/results.json`

Some file and directory names contain the legacy internal label `d1`; these
refer to the trajectory-weighted readout called `GRACE` in the paper.

## Compute Notes

Representative recorded runtimes from the saved result files:

- Core 10-seed MNIST reservoir sweep: `10,737 s` (`~3.0 h`)
- Dynamical-noise robustness study: `2,720 s` (`~45 min`)
- Accuracy-defined grace-period appendix analysis: `360 s` (`~6 min`)
- Readout-audit appendix sweep: `13,952 s` (`~3.9 h`)
- Sequential CIFAR-10 LSTM appendix check and timestep audit: hour-scale; the saved audit run recorded `3,229 s` for 3 seeds.

The full set of reported experiments stayed comfortably below one day of
single-GPU compute. Additional pilot and exploratory runs required extra
compute of similar order.

## Notes

- This package is anonymized for review.
- No raw data, author identities, or repository metadata are included.
- Some helper scripts in the archive were used for bound inspection or table
  generation during manuscript preparation; the primary reproduction path is
  through the commands listed above.
