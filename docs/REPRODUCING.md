# Reproducing the paper

Run commands from the repository root unless a step says otherwise. Saved
results and frozen protocol files are evidence: keep the release unchanged
and use a disposable copy or `scripts/prepare_run.py` for new execution.

## 1. Saved-result reproduction

```bash
python -m pip install -r requirements.txt
python scripts/verify_release.py
python scripts/reproduce_figures.py
python gen_appendix_tables.py
```

`reproduce_figures.py` uses saved JSON and stages the saved operator, cocycle,
frozen-feature CIFAR, and LSTM images. Recomputing those diagnostics requires
their experiment scripts. The modern figure/table generator reads the final
four-condition panel and its supporting per-seed files. None of these commands
trains a model or downloads images. Plot bytes can differ with plotting-library
versions; compare the underlying numeric results. Run hash verification before
regenerating outputs.

`scripts/generate_figures.py` is the untouched historical supplement generator;
use `scripts/reproduce_figures.py` for the current preprint, including its
weight-only GRACE figure.

## 2. Original experiments

The commands and full original script inventory are preserved in
[ORIGINAL_SUPPLEMENT.md](ORIGINAL_SUPPLEMENT.md). The principal entry points are:

```bash
python -m transient_geometry.experiments.overnight_10seed
python transient_geometry/grace_period.py
python -m transient_geometry.experiments.e6_trained_at_operating_point
python -m transient_geometry.experiments.e12_cifar10_resnet18_features
python -m transient_geometry.experiments.e13_sequential_cifar10_lstm
python -m transient_geometry.experiments.e10_operator_diagnostics
python -m transient_geometry.experiments.e11_cocycle_envelope
```

These commands train/evaluate models and write to `results/`; run in a copy.
MNIST, FashionMNIST, and CIFAR-10 are downloaded by their loaders. Use
`--per-timestep-audit --audit-lambdas 1.0,1.6,2.0` with E13 for the saved
single-timestep LSTM diagnostic.

## 3. Tiny ImageNet data and environment

Obtain the [Tiny ImageNet distribution](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
and extract it under `data/` in each run directory. Expected files include:

```text
data/tiny-imagenet-200/wnids.txt
data/tiny-imagenet-200/train/<class>/images/*.JPEG
data/tiny-imagenet-200/val/images/*.JPEG
data/tiny-imagenet-200/val/val_annotations.txt
```

There are 100,000 training and 10,000 labelled validation images. The latter
split is also used for checkpoint selection, not an untouched test set.
ConvNeXt-Tiny weights are downloaded through torchvision. E14 validates the
dataset; E17 validates it before cache extraction. The recorded archive SHA-256
is `6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821`.

Python 3.12 is the tested packaging environment. The recorded modern training
environment is in each run's `environment.json` (Python 3.12.10, PyTorch
2.13.0+cu130, torchvision 0.28.0+cu130, NumPy 2.4.4, scikit-learn 1.9.0).
These are historical records, not a portable lockfile. Use a compatible
PyTorch/torchvision installation for your hardware. CUDA BF16 was used for the
reported modern results; CPU execution is slower and not bitwise equivalent.
Spatial feature arrays occupy about 7.7 GiB; the frozen protocol also checks
its configured free-disk threshold before extraction.

## 4. SwiGLU pilots and paired horizon experiment

Create a clean run directory (destination must not already exist):

```bash
python scripts/prepare_run.py ../terminal-failure-swiglu --family swiglu
```

Change into that directory, install its requirements if necessary, and place
Tiny ImageNet at the path above. Then:

```bash
python -m transient_geometry.experiments.e14_recursive_tinyimagenet_rebuttal validate
python -m transient_geometry.experiments.e14_recursive_tinyimagenet_rebuttal all
python -m transient_geometry.experiments.e14_recursive_tinyimagenet_rebuttal benchmark
python -m transient_geometry.experiments.e14_horizon_extension_postprocess
python -m transient_geometry.experiments.e15_recursive_tinyimagenet_aligned_t32 all
python -m transient_geometry.experiments.e16_paired_inference_horizon model --variant e16_t8
python -m transient_geometry.experiments.e16_paired_inference_horizon model --variant e16_t32
python -m transient_geometry.experiments.e16_paired_inference_horizon aggregate
```

The E14/E15 `all` workflows stop after seed 0 when their continuation gates
fail. This is the recorded outcome, not an execution error. The explicit E14
`benchmark` command is needed because `all` returns after a failed gate.
E16 independently trains both configurations on seeds 101–103.

## 5. Shared-attention experiment and matched control

Create a separate clean directory from the unchanged release:

```bash
python scripts/prepare_run.py ../terminal-failure-attention --family attention
```

This copies the exact published E16 control artifacts required by E17's hash
checks. Do not overwrite those artifacts with a fresh E16 run in this directory.
Change into it, provide the dataset, and run:

```bash
python -m transient_geometry.experiments.e17_recursive_attention_tinyimagenet verify --variant t8
python -m transient_geometry.experiments.e17_recursive_attention_tinyimagenet features --variant t8
python -m transient_geometry.experiments.e17_recursive_attention_tinyimagenet smoke --variant t8
python -m transient_geometry.experiments.e17_recursive_attention_tinyimagenet train --variant t8 --seed 101
python -m transient_geometry.experiments.e17_recursive_attention_tinyimagenet evaluate --variant t8 --seed 101
python -m transient_geometry.experiments.e17_recursive_attention_tinyimagenet probe --variant t8 --seed 101
```

Repeat the last three commands for seeds **102 and 103**, then:

```bash
python -m transient_geometry.experiments.e17_recursive_attention_tinyimagenet aggregate --variant t8
```

The aggregate writes the predeclared matched-control decision. If it triggers,
run `train`, `evaluate`, and `probe` for **variant t32**, each at seeds 101, 102,
and 103, then `aggregate --variant t32`. Preserve the decision if your rerun
does not trigger the control; do not alter thresholds to force the result.

E17's own `panel` stage expects the full original E16 run artifacts; use a
working copy containing those published results for this stage. The preprint's
final panel is the subsequent follow-up `final_architecture_panel.json`.

## 6. Saved checkpoints and follow-up audits

Extract the companion checkpoint archive at the repository root and run
`python scripts/verify_release.py --checkpoints`. The archive restores 67
selected, alternative, last-epoch, smoke, and interrupted-run checkpoints.
Some `last` files contain optimizer state; interrupted attempts are not
additional scientific replicates.

The follow-up driver is:

```bash
python -m transient_geometry.experiments.e17_causal_panel_audit --help
```

Stages include `causal`, `classwise`, `checkpoint`, `probe`, `scale`,
`provenance`, `panel`, `responses`, and `manifest`. It never trains a model.
Stages reuse existing outputs; a successful run with published outputs present
is not evidence of fresh re-evaluation. In a disposable copy, move the specific
audit outputs aside before recalculating them. Checkpoint/probe stages need
the companion checkpoints, dataset, and feature caches.

For cache rebuilding in such a copy, move the **entire** published
`results/modern_iterative_panel/spatial_cache/` directory aside before running
E17 `features`; its included metadata/labels alone do not constitute a complete
cache, and the driver rejects partial caches. E14 `features` regenerates the
pooled cache. Keep the untouched release for provenance comparison.

## Evidence interpretation

Use trajectory metrics for the modern horizon headline, and readout metrics
for decoder comparisons. BF16 classification of differently shaped tensors
changes a few attention predictions, so these two saved series differ slightly.
The raw series are preserved. The selected-checkpoint and final-epoch SwiGLU
effect sizes differ substantially; see the checkpoint audit. Modern statistics
are descriptive, and training horizon changes together with residual step scale.
