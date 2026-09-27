# Terminal Failure Is Not Computational Failure

**A Pre-Collapse Readout Regime in Nonlinear Dynamical Learners**  
Rohan Maskrey · rohanmaskrey@dunelm.org.uk

Code, frozen experimental protocols, saved results, and the arXiv preprint for
the paper accepted to NeurIPS 2026. This repository combines the original
supplementary package with the Tiny ImageNet experiments and analyses added
during rebuttal.

The central finding is the **pre-collapse regime**: terminal representations
can deteriorate while earlier trajectory prefixes remain predictive. Fixed
early-window readout, norm-ratio stopping, and GRACE are potential decoders
whose usefulness depends on the dynamics and training horizon.

[Read the paper](paper/main.pdf) · [Reproduction guide](docs/REPRODUCING.md) ·
[Experiment index](docs/EXPERIMENTS.md) · [Dataset notes](DATASET_NOTES.md)

## Quick start: saved results, no training

Use Python 3.12. Create and activate a virtual environment, then run from the
repository root:

```bash
python -m pip install -r requirements.txt
python scripts/verify_release.py
python scripts/reproduce_figures.py
```

The last command regenerates the original plots and the modern Tiny ImageNet
figures/tables, staging them in `paper/figures/` and `paper/sections/`. It uses
saved metrics; no datasets, checkpoints, GPU, or training are required. The
three-seed statistics use sample standard deviations.

The source paper can be compiled from `paper/` with `latexmk -pdf main.tex`.
The included PDF was built using Tectonic 0.17.0.

## What is included

| Location | Contents |
| --- | --- |
| `transient_geometry/` | Original experimental code plus E14–E17 training, evaluation, benchmarking, and audit scripts |
| `results/` | Original supplementary results and modern per-seed metrics, training histories, frozen configurations, plots, and audits |
| `Rebuttal/` | Three frozen SwiGLU experimental protocols, preserved byte-for-byte for provenance |
| `results/modern_iterative_panel/protocol/` | Frozen attention protocol, configurations, seeds, and hashes |
| `results/modern_iterative_panel/followup_analysis/` | Saved follow-up analyses and their protocol |
| `paper/` | Current preprint PDF, LaTeX, figure assets, and figure/table generators |
| `scripts/` | Release verification, figure reproduction, and clean-run preparation |
| `docs/` | Run guide, provenance, checkpoint manifest, and historical supplementary inventory |

The original supplementary scripts and result files are preserved. Its old
README is retained as [historical documentation](docs/ORIGINAL_SUPPLEMENT.md);
its anonymisation, appendix labels, and hardware/compute statements describe
the original submission only. The current instructions are here and in the
reproduction guide. Legacy names such as `d1` refer to GRACE.

## Modern experiments

The final panel has 12 completed runs: SwiGLU and shared-attention heads,
training horizons 8 and 32, and seeds 101–103. Both operate on frozen
ImageNet-pretrained ConvNeXt-Tiny features. At the selected checkpoints,
extending eight-step-trained models to 32 steps reduces terminal accuracy by
5.11 ± 0.94 pp (SwiGLU) and 2.42 ± 0.29 pp (attention). Matched training
largely removes the loss. These are descriptive three-seed results.

Training horizon and residual step scale change together in the matched
controls. The validation split also selects checkpoints. SwiGLU's final-epoch
losses are only 0.59–0.68 pp; the selected-checkpoint losses are 4.27–6.13 pp.
The earlier E14/E15 pilots stopped after seed 0, and interrupted attention
attempts are retained separately, not counted as extra replicates. Negative
GRACE results are retained. No new canonical-DEQ rebuttal experiment was run.

## Checkpoints and caches

The companion **`terminal-failure-checkpoints.zip`** contains the trained
rebuttal checkpoints, including saved alternatives and interrupted attempts.
It is intended as a separate release asset. Extract it at the repository root
to restore the original `results/...` paths, then run:

```bash
python scripts/verify_release.py --checkpoints
```

The code/results repository is sufficient for saved-result figure generation.
Checkpoint-based re-evaluation additionally needs the dataset and regenerated
feature caches. Raw images, downloaded ImageNet weights, virtual environments,
and roughly 8 GB of regenerable feature caches are not distributed. Exact
checkpoint hashes and omitted cache sizes are recorded in `docs/`.

## Re-running experiments

Use the [reproduction guide](docs/REPRODUCING.md). Frozen drivers deliberately
reuse completed runs or refuse to overwrite them. `scripts/prepare_run.py`
creates a separate run directory with the required code and frozen inputs,
avoiding collisions with the published evidence. SwiGLU and attention runs
use separate directories because the attention protocol locks the original
SwiGLU control artifacts by hash.

The original studies used an RTX PRO 1000 Blackwell Laptop GPU. The modern
Tiny ImageNet runs used an RTX 5080 Laptop GPU and BF16. Recorded environment
files accompany the runs; hardware and numerical precision can affect exact
results. `requirements.txt` lists portable dependencies rather than claiming
to reproduce a particular CUDA installation.

## Citation and licence

Please cite the paper and credit Rohan Maskrey; machine-readable metadata is
provided in [CITATION.cff](CITATION.cff). The repository is
[maskrys-dev/terminal-failure](https://github.com/maskrys-dev/terminal-failure).
An arXiv identifier can be added after publication.

Copyright © 2026 Rohan Maskrey. The author's code, documentation, results,
and paper are licensed under **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**.
See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). Third-party datasets,
pretrained weights, dependencies, and the NeurIPS style file retain their
respective terms.
