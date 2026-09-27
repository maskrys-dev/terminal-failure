# Final E14 Recursive Tiny ImageNet Rebuttal Protocol

**Status:** frozen before the first scientific pilot result.

**Authority:** the user's attached brief titled “Tiny ImageNet-200 + frozen
ConvNeXt-Tiny + a trained weight-tied recursive SwiGLU head,” SHA-256
`26425A07C907FD861B8B2F4A8F8C7EC091197C2AAC55C4F9DEB4A01F3F35BB6D`.

This protocol supersedes the earlier post-meta and DEQ-first experiment
documents. A valid null result does not activate the DEQ fallback. The DEQ is
used only if this recursive implementation fails its engineering smoke test.

## Scientific question and claim ceiling

Test whether terminal readout becomes less effective than trajectory-prefix
readout under controlled recurrent stress in a trained, modern, weight-tied
recursive classifier operating on strong non-CIFAR representations.

The strongest permitted claim is:

> In a trained weight-tied recursive classifier operating on frozen
> ConvNeXt-Tiny representations of Tiny ImageNet-200, terminal readout becomes
> less effective than trajectory-prefix readout under controlled recurrent
> stress, while ordinary-regime terminal performance remains competitive.

Do not claim that ConvNeXt is iterative or suffers terminal failure, that this
is a Universal Transformer, that the result establishes production-scale
prevalence, that GRACE universally wins, or that Tiny ImageNet is a deployment
benchmark.

## Dataset and frozen feature extractor

- Tiny ImageNet-200, standard 100,000-image train and 10,000-image labelled
  validation splits.
- ImageNet-1K-pretrained torchvision ConvNeXt-Tiny, completely frozen.
- Use the weight package's standard ImageNet inference preprocessing.
- Cache the 768-dimensional globally pooled penultimate features.
- Fit feature mean and standard deviation on the training split only.
- Use no backbone fine-tuning and no image augmentation.

## Locked model

For a standardised feature \(x\in\mathbb{R}^{768}\):

\[
h_0 = W_{\rm in}x+b_{\rm in},
\]

\[
u_t = \operatorname{LayerNorm}(h_t),
\qquad
F_\theta(u_t) =
W_o[\operatorname{SiLU}(W_g u_t)\odot W_v u_t],
\]

\[
h_{t+1}=h_t+\frac{\lambda}{T_{\rm train}}F_\theta(u_t).
\]

One pre-norm SwiGLU block is shared across every step. The classifier is also
shared:

\[
\hat y_t = W_c\operatorname{LayerNorm}(h_t)+b_c.
\]

Locked values:

| Choice | Value |
|---|---:|
| State dimension | 512 |
| SwiGLU intermediate dimension | 2,048 |
| Dropout | 0.1 |
| Training horizon | 8 |
| Training gain | 1.0 |
| Number of classes | 200 |

The block LayerNorm and classifier LayerNorm are distinct modules, each shared
across timesteps. There is no intermediate supervision.

## Locked training

| Choice | Value |
|---|---:|
| Objective | terminal cross-entropy only |
| Optimiser | AdamW |
| Learning rate | \(3\times10^{-4}\) |
| Weight decay | \(10^{-2}\) |
| Batch size | 512 |
| Epochs | 30 |
| Warm-up | 2 epochs |
| Schedule | cosine decay |
| Label smoothing | 0.1 |
| Gradient clipping | 1.0 |
| Seeds | 0, 1, 2 |
| Precision | BF16 when supported, otherwise FP16 |

Select each final checkpoint only by terminal validation accuracy at
\((T,\lambda)=(8,1.0)\). Never select it using stressed-gain performance.

## Frozen trajectory evaluation

- Roll every scientific trajectory to \(T_{\max}=32\).
- Use exactly
  \(\lambda\in\{0.75,1.00,1.10,1.20,1.30,1.40,1.50,1.60\}\).
- Do not densify or extend the grid after results are visible.

At every timestep record shared-head accuracy, cross-entropy, mean state norm,
relative update residual, prediction entropy, class-change fraction, and
non-finite counts.

## Locked readouts

All readouts aggregate states and then apply the same trained shared head.

1. Terminal: \(h_{32}\).
2. Early window: \(\frac{1}{8}\sum_{t=1}^{8} h_t\).
3. Uniform: \(\frac{1}{32}\sum_{t=1}^{32} h_t\).
4. Adaptive stopping: the first state whose safeguarded norm ratio exceeds the
   submitted fixed threshold \(\tau_{\rm stop}=1.5\), or the terminal state if
   no crossing occurs.
5. GRACE: per-sample weights proportional to
   \(\exp[-10\max(0,g_t-1.2)]\).
6. Fixed-profile GRACE control: at \(\lambda_{\rm ref}=1.30\), average the
   normalised per-sample GRACE weights over the full training split, freeze
   that global temporal profile, and apply it to every validation example and
   every gain.

The fixed-profile control replaces the earlier arbitrary exponential control;
it isolates per-sample adaptation while matching GRACE's average temporal
shape.

## Probe audit

- Use a fixed stratified 20,000-example training subset, seed 2026.
- Use the full validation split.
- Fit `RidgeClassifier(alpha=1.0)` probes identically for all representations.
- Fit aggregate-readout probes over the full gain grid.
- Run the dense single-timestep audit only at gains 1.00, 1.30, and 1.60.
- Predeclare absolute grace-period thresholds
  \(\alpha\in\{0.40,0.50,0.60\}\); report all three without selecting among
  them after viewing results.

## Scientific continuation gate

Run seed 0 first. Continue to seeds 1 and 2 only if all conditions hold:

1. Early-window accuracy exceeds terminal accuracy by at least 1.0 percentage
   point at three consecutive frozen gains in either the shared-head or
   fixed-probe comparison.
2. At \(\lambda=1.0\), terminal accuracy is within 0.5 percentage point of the
   best trajectory-aware readout under the comparison that passes condition 1.
3. A grace period shortens with gain or stressed timestep accuracy is visibly
   non-monotonic.
4. States remain finite and classification remains above 0.5% random chance;
   NaNs or pure numerical blow-up do not pass.
5. Neither the model, grid, windows, thresholds, nor probe regularisation was
   changed after viewing the pilot.

If the gate fails, stop after seed 0. Do not tune the architecture or grid in
search of a positive result.

For final reporting, require the early-minus-terminal sign to agree in at
least two of three seeds at every headline setting. Show individual values,
paired differences, mean, and standard deviation; do not foreground an
underpowered significance test.

## Cost benchmark

Use one fixed validation batch with \(B=512,T=32,d=512\). After warm-up,
synchronise CUDA around repeated measurements and report medians for:

- terminal;
- early-window;
- adaptive stopping;
- uniform averaging;
- naive GRACE storing all states;
- streaming GRACE.

Report readout-only and full-inference time where meaningful, peak allocated
GPU memory, whether all states are stored, and whether recurrence can halt
early.

## Primary outputs

1. Shared-head timestep accuracy at gains 1.00, 1.30, and 1.60.
2. Readout performance versus gain for terminal, early, uniform,
   fixed-profile, adaptive, and GRACE.
3. Grace period, terminal residual, and terminal-minus-early gap.
4. Three-seed headline table if and only if the continuation gate passes.
5. Measured runtime and memory table.
