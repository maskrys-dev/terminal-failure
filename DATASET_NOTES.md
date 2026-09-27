# Dataset Notes

This supplementary package does not redistribute benchmark datasets. MNIST, FashionMNIST, and CIFAR-10 are downloaded automatically by the
corresponding torchvision loaders; Tiny ImageNet must be supplied separately.

## Datasets Used

- `MNIST`
  - Used in the main reservoir sweeps, cross-system comparison, true grace
    period analysis, readout audits, and trained-system experiments.
  - Accessed through `torchvision.datasets.MNIST(..., download=True)`.

- `FashionMNIST`
  - Used for the appendix cross-dataset robustness check.
  - Accessed through `torchvision.datasets.FashionMNIST(..., download=True)`.

- `CIFAR-10`
  - Used for the main-text bridge experiment testing endpoint-vs-trajectory
    mismatch on a harder image dataset.
  - Accessed through `torchvision.datasets.CIFAR10(..., download=True)`.

## Pretrained Weights Used

- `torchvision` ImageNet-pretrained `ResNet-18` weights
  - Used only as a frozen feature extractor for the CIFAR-10 feature
    robustness check.
  - Downloaded automatically by the corresponding torchvision model-loading
    call when not already cached locally.

## Archive Policy

- No raw dataset files are included in this archive.
- No pretrained weight files are redistributed in this archive.
- The scripts expect a local `data/` directory and will populate it
  automatically when internet access is available.
- The experimental subsampling used in the paper is controlled inside the
  scripts and result files (for example, `5,000` train / `5,000` test splits
  for the main reservoir studies).

## Tiny ImageNet-200 (rebuttal and arXiv extension)

- Frozen ImageNet-pretrained ConvNeXt-Tiny features support the recursive
  SwiGLU and shared-attention experiments.
- Download the standard Tiny ImageNet distribution and extract it to
  `data/tiny-imagenet-200/`; see `docs/REPRODUCING.md` for the expected layout.
- Training uses 100,000 images; the labelled 10,000-image validation split is
  used for both checkpoint selection and evaluation.
- Pooled and spatial feature caches are generated locally. The large feature
  arrays and pretrained backbone weights are not included in this repository.
- Included prediction arrays, probe-subset indices and cached labels are
  analysis artifacts, not copies of the source images.
