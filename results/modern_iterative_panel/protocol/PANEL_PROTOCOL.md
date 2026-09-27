# Modern Iterative Panel: E17 Recursive Attention Protocol

**Status:** frozen before the first E17 smoke trajectory or scientific result.

**Primary new experiment:** E17, a weight-tied
Universal-Transformer-style recursive attention head trained for eight
iterations on frozen ConvNeXt-Tiny spatial features from Tiny ImageNet-200.

**Triggered control:** E17C, the identical attention system trained for 32
iterations. E17C is run only if the frozen three-seed trigger below passes.

E16 is complete, locked, and external to E17. No E14, E15, or E16 file may be
modified, overwritten, regenerated, or reinterpreted by this workflow.

## Locked E16 anchor

The E16 paired experiment used seeds 101, 102, and 103. Its frozen hashes are:

- aggregate config:
  `e637a6854ac8b990d5a4822b4e7beece1b88226839b88360a86b84c89b60b605`;
- \(T_{\rm train}=8\) config:
  `6bdb5bdcc994f8e7f145042f6f2e5f5b85b0cb73c99dbeb740e13ead80084741`;
- \(T_{\rm train}=32\) config:
  `33a36523d381d7996f5e4f11a5d3e129f9d90542085227a3f7b8932e4c3f7bac`;
- protocol:
  `9cc8f8204b2fb92cb9e42c0f73ba172a3ba994264d4fe80596d35f00c30c9c28`.

E16's mean terminal drop for the eight-step-trained model was
\(5.11\pm0.94\) percentage points, the matched-control drop was
\(-0.42\pm0.18\) points, and the paired interaction was
\(5.53\pm0.92\) points.

## Dataset and frozen spatial representation

- Dataset: standard Tiny ImageNet-200.
- Training split: 100,000 images, 500 per class.
- Validation split: 10,000 labelled images, 50 per class.
- Backbone: `torchvision.models.convnext_tiny` with
  `ConvNeXt_Tiny_Weights.DEFAULT`.
- Image preprocessing: exactly `ConvNeXt_Tiny_Weights.DEFAULT.transforms()`.
- The backbone is in evaluation mode and all parameters are frozen.
- Spatial extraction is the output of `model.features(images)`, followed by
  the frozen `model.classifier[0]` LayerNorm2d applied at every spatial
  location.
- The required output shape is exactly \(768\times7\times7\). Any other shape
  stops the experiment.
- The map is permuted and flattened to 49 tokens of dimension 768.
- Extraction uses BF16 autocast on the available CUDA device.
- Cached token values are stored as NumPy `.npy` arrays in IEEE FP16.
- No dataset-derived feature standardisation is applied; the frozen ConvNeXt
  classifier LayerNorm2d is the only feature normalisation.
- Labels are cached as signed 64-bit integers.
- Cache construction writes new partial files and atomically renames them only
  after complete finite-value, shape, count, and class-balance validation.
- Existing complete cache files are reused only when their metadata and file
  hashes validate. Existing files are never silently overwritten.

The expected feature-cache payload is approximately 7.7 GiB. Extraction stops
before allocation if the available disk margin is below 20 GiB.

## Token preparation

Each 768-dimensional spatial token is projected to \(d_{\rm model}=256\) by a
learned affine map. One learned class token is prepended.

Spatial positions use a fixed, analytic two-dimensional sine/cosine embedding
on the \(7\times7\) grid:

- 64 channels encode sine of the row coordinate;
- 64 encode cosine of the row coordinate;
- 64 encode sine of the column coordinate;
- 64 encode cosine of the column coordinate;
- frequencies use the standard \(10000^{-k/64}\) schedule.

The class token has a fixed zero positional vector. Positional embeddings are
added once before the first recurrent step, not at every step. The learned
class token is initialised with a truncated normal distribution of standard
deviation 0.02. Linear, LayerNorm, and MultiheadAttention parameters otherwise
use PyTorch's deterministic default initialisation under the declared seed.

## Shared recursive attention block

One block is weight-tied across all recurrent steps. For state \(X_t\):

\[
\widetilde X_t
=
X_t
+
\frac{1}{T_{\rm train}}
\operatorname{Dropout}
\left(
\operatorname{MHA}
\left[
\operatorname{LN}_1(X_t)
\right]
\right),
\]

\[
X_{t+1}
=
\widetilde X_t
+
\frac{1}{T_{\rm train}}
\operatorname{Dropout}
\left(
W_o
\left[
\operatorname{SiLU}(W_g\operatorname{LN}_2(\widetilde X_t))
\odot
W_v\operatorname{LN}_2(\widetilde X_t)
\right]
\right).
\]

Frozen architectural choices:

- model dimension 256;
- eight attention heads, head dimension 32;
- one PyTorch `nn.MultiheadAttention` module with `batch_first=True`;
- attention-probability dropout 0.1;
- attention residual-output dropout 0.1;
- SwiGLU gate and value width 1024 each;
- SwiGLU residual-output dropout 0.1;
- pre-norm LayerNorm with epsilon \(10^{-5}\);
- no attention mask;
- no causal mask;
- a single shared LayerNorm-plus-linear 200-class head on the class token;
- terminal-only cross-entropy supervision;
- no intermediate loss or auxiliary objective;
- no input re-injection after token preparation.

E17 uses \(T_{\rm train}=8\). E17C, if triggered, changes only
\(T_{\rm train}:8\rightarrow32\), the implied residual scale, experiment
identifier, output directory, and checkpoint-selection label.

## Optimisation

- AdamW;
- learning rate \(3\times10^{-4}\);
- weight decay \(10^{-2}\);
- 30 epochs;
- two linear warm-up epochs followed by cosine decay;
- label smoothing 0.1;
- global gradient-norm clipping at 1.0;
- BF16 autocast when supported;
- physical batch size 128;
- four-microbatch gradient accumulation;
- nominal effective batch size 512;
- every example is retained; the final accumulation group in an epoch may be
  smaller and is weighted by its actual example count;
- deterministic seed controls parameter initialisation, dropout, and the
  epoch-to-epoch shuffled sample order;
- paired scientific seeds 101, 102, and 103.

Checkpoint selection uses only terminal validation accuracy at
\(\lambda=1\) and \(T_{\rm test}=T_{\rm train}\). Ties retain the earliest
epoch. No \(T=32\) result is consulted when selecting E17 checkpoints.

## Engineering smoke gate

The smoke run is an engineering test, not a scientific seed:

- 4,000 stratified training examples;
- 2,000 stratified validation examples;
- subset-selection seed 2026;
- model/training seed 0;
- three epochs;
- all frozen architecture and optimiser settings are retained.

The gate passes only if:

1. epoch-three training loss is lower than epoch-one training loss;
2. best smoke validation terminal accuracy at \(T=8\) is strictly above 10%;
3. all token states, class-token states, and logits are finite through 32
   recurrent steps;
4. a 32-step class-token trajectory is logged with exact shape
   `[validation_examples, 32, 256]` in batchwise evaluation;
5. terminal, first-eight, uniform, adaptive, and GRACE representations and
   logits all execute and are finite;
6. peak CUDA allocated memory is at most 14.0 GiB;
7. no scientific horizon comparison is used to alter any setting.

Only implementation repairs for crashes, shapes, numerical errors, or
impossible memory use are permitted. Any repair is logged as a protocol
deviation. A failed gate stops E17.

After a passing gate, all three E17 seeds run regardless of scientific result.

## Frozen scientific evaluation

For every selected checkpoint, evaluate only

\[
\lambda=1,\qquad
T_{\rm test}\in\{8,16,24,32\}.
\]

One deterministic 32-step trajectory is computed and its prefixes define the
four test horizons. The primary representation is the class token.

At every timestep record:

- shared-head validation accuracy and cross-entropy;
- mean class-token norm;
- mean relative class-token update residual
  \(\|c_t-c_{t-1}\|_2/\max(\|c_{t-1}\|_2,10^{-12})\);
- prediction entropy;
- fraction of examples changing predicted class from the preceding step;
- non-finite token-state, class-state, and logit counts.

For each test horizon record the earliest best fixed shared-head timestep, its
accuracy, and peak-minus-terminal accuracy. This best timestep is an oracle
trajectory diagnostic, not a deployable per-sample rule.

Readouts use the shared trained head:

1. terminal: \(c_T\);
2. early: \(\frac18\sum_{t=1}^{8}c_t\);
3. uniform: \(\frac1T\sum_{t=1}^{T}c_t\);
4. adaptive: the first \(c_t\) for which
   \(\|c_t\|/\max(\|c_1\|,10^{-12})>1.5\), otherwise \(c_T\);
5. GRACE: the weighted class-token representation with
   \(w_t\propto\exp[-10\max(0,g_t-1.2)]\) and
   \(g_t=\|c_t\|/\max(\|c_1\|,10^{-12})\).

No readout threshold is tuned for attention.

## Fixed-alpha probe audit

The secondary probe audit uses the exact E14 stratified training indices:

- source:
  `results/rebuttal_recursive_tinyimagenet/probe_subset_indices.npy`;
- expected SHA-256:
  `9f7378e1c3cde52953ea5096b6b80cfb4b52fff8cafcd87be19d9b366590a19d`;
- 20,000 training examples;
- full 10,000-example validation split;
- `sklearn.linear_model.RidgeClassifier(alpha=1.0)`;
- no representation-specific standardisation;
- no solver or regularisation selection.

Separate fixed probes are fit to:

- class token at step 8;
- mean class token over steps 1--8;
- class token at step 32.

The early representation is identical at test horizons 8 and 32 and is fit
once. Probe directional agreement means that the across-seed mean accuracy of
the step-8 terminal representation is strictly greater than that of the
step-32 terminal representation.

## Frozen scientific classification

Let, for seed \(s\),

\[
d_s=A_s(T=8)-A_s(T=32).
\]

E17 is classified as **endpoint invalidity** only if all hold:

1. the across-seed mean shared-head drop is at least 2.0 percentage points;
2. \(d_s>0\) for all three seeds;
3. the absolute difference between across-seed mean early-eight accuracy at
   \(T=32\) and mean terminal accuracy at \(T=8\) is at most 1.0 percentage
   point;
4. all states and logits are finite and every terminal or early accuracy is
   above 2.5%;
5. the across-seed mean fixed-probe step-8 terminal accuracy is strictly
   greater than the mean fixed-probe step-32 terminal accuracy.

E17 is classified as **stable** only if all hold:

1. the across-seed mean terminal drop is below 0.5 percentage point;
2. the mean peak-minus-terminal gap at \(T=32\) is below 0.5 percentage point;
3. all class-token norms and relative residuals are finite;
4. the mean endpoint residual is no greater than the mean residual at step 8;
5. all states and logits are finite and accuracy remains above 2.5%.

Every other completed outcome is **ambiguous**.

The categories and thresholds are not modified after results.

## Frozen matched-horizon trigger

E17C runs only if all hold after all three E17 seeds finish:

1. the across-seed mean E17 terminal drop from 8 to 32 is at least 2.0
   percentage points;
2. the drop is positive in all three seeds;
3. the absolute difference between across-seed mean E17 early-eight accuracy
   at \(T=32\) and terminal accuracy at \(T=8\) is at most 1.0 percentage
   point.

The trigger does not depend on probes, one seed, or visual inspection.

If triggered, all E17C seeds 101--103 run. E17C is evaluated at the same
gain and horizons. If the trigger fails, E17C is recorded as skipped.

## Scope and stop rules

E17 is an architecture-breadth study. Spatial attention and pooled SwiGLU
consume different frozen ConvNeXt representations, so the final report calls
this a comparison across tested iterative systems, not a causal isolation of
attention.

Stop and report rather than alter the protocol if the spatial output is not
\(7\times7\times768\), cache validation fails, the engineering gate fails,
or a required assumption cannot be met.

Do not add a gain sweep, horizon beyond 32, SSM, diffusion model, dataset,
architecture, GRACE variant, or post-result threshold. Do not discard null or
mixed outcomes. Do not describe E17 as a complete reproduction of the
original Universal Transformer.

The optional contractive DEQ-style control is outside the required E17
workflow and is not started until E17, any triggered E17C control, aggregation,
and writing-time/resource checks are complete.
