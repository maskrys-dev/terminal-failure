# E16 Paired Inference-Horizon Experiment

**Status:** frozen before the first E16 engineering or scientific result.

**Relationship to earlier experiments:** E14 seed 0 exposed a possible
train/test horizon mismatch. E15 corrected that mismatch at a single training
horizon but did not pass its predeclared gain-stress continuation gate. E15
therefore remains stopped after seed 0. E16 is a new confirmatory replication
of the horizon-mismatch result with fresh seeds; it is not a continuation or
retuning of E15.

## Scientific question

Does terminal accuracy degrade when test-time recurrence extends beyond the
trained horizon, and is that degradation reduced when the model is trained at
the longer horizon?

## Paired design

Use fresh paired seeds

\[
\{101,102,103\}.
\]

For every seed, train two otherwise identical recursive classifiers:

\[
T_{\rm train}\in\{8,32\}.
\]

The same seed gives each pair the same parameter initialization and minibatch
order. The only scientific difference within a pair is the training horizon
and its implied residual scale:

\[
h_{t+1}=h_t+\frac{1}{T_{\rm train}}F_\theta(h_t).
\]

Both models use:

- Tiny ImageNet-200's standard 100,000-image training and labelled
  10,000-image validation splits;
- the already validated, frozen ImageNet-pretrained ConvNeXt-Tiny features;
- the weight-tied pre-norm SwiGLU recursive head with state dimension 512 and
  intermediate dimension 2048;
- terminal-only cross-entropy;
- AdamW, learning rate \(3\times10^{-4}\), weight decay 0.01;
- batch size 512, 30 epochs, two warm-up epochs, cosine decay;
- label smoothing 0.1, gradient clipping at 1.0, and dropout 0.1;
- checkpoint selection only by terminal validation accuracy at
  \(T_{\rm test}=T_{\rm train},\lambda=1\);
- the existing no-augmentation feature protocol and mixed-precision policy.

No intermediate supervision is allowed.

## Frozen evaluation

Evaluate only at

\[
\lambda=1.0
\]

and

\[
T_{\rm test}\in\{8,16,24,32\}.
\]

There is no gain sweep, gain-grid extension, GRACE comparison, adaptive
stopping sweep, probe fitting, or threshold selection in E16.

For each model, seed, and test horizon report:

1. terminal shared-head accuracy;
2. the shared-head accuracy obtained from the mean of states 1--8
   ("early-eight-state readout");
3. the best fixed shared-head timestep among states 1 through
   \(T_{\rm test}\), its accuracy, and its timestep;
4. terminal-minus-early accuracy in percentage points;
5. the mean relative update residual
   \(\|h_t-h_{t-1}\|_2/\max(\|h_{t-1}\|_2,10^{-12})\), including the full
   trajectory and the value at every reported endpoint;
6. non-finite state and logit counts.

Because every requested horizon is at least eight steps, the early-eight-state
representation is mathematically identical across the four test horizons for
a fixed model. It is nevertheless recorded at every horizon so that the
terminal-versus-prefix comparison is explicit.

The "best fixed trajectory readout" in the control criterion means the maximum
validation accuracy obtained by applying the model's single trained shared
classifier at one fixed timestep \(t\le T_{\rm test}\). It is a trajectory
diagnostic, not a separately trained or per-sample oracle.

## Frozen confirmatory criteria

All three fresh seed pairs are run regardless of individual outcomes. No
significance test is a primary result.

For \(T_{\rm train}=8\):

1. The across-seed mean terminal drop
   \(A_{8\text{-train}}(8)-A_{8\text{-train}}(32)\) is at least 2.0 percentage
   points.
2. The early-eight-state accuracy at \(T_{\rm test}=32\) falls by no more than
   1.0 percentage point relative to its value at \(T_{\rm test}=8\), both for
   the across-seed mean and for every individual seed.
3. The terminal degradation has the same sign in all three seeds:
   \(A_{8\text{-train}}(8)>A_{8\text{-train}}(32)\).

For the \(T_{\rm train}=32\) control:

4. At \(T_{\rm test}=32\), terminal accuracy is within 0.5 percentage point of
   the best fixed shared-head timestep, both for the across-seed mean and for
   every individual seed.
5. Define the paired interaction for seed \(s\) as

\[
\Delta_{\rm interaction}^{(s)}
=
\left[A_{8\text{-train}}^{(s)}(8)
-A_{8\text{-train}}^{(s)}(32)\right]
-
\left[A_{32\text{-train}}^{(s)}(8)
-A_{32\text{-train}}^{(s)}(32)\right].
\]

   "Substantially smaller" is frozen as an across-seed mean interaction of at
   least 1.0 percentage point, with a positive interaction in every seed.

The overall confirmatory result passes only if all five criteria pass and all
states and logits are finite. Random-chance accuracy is 0.5%; any run at or
below 2.5% is flagged as not materially above chance.

## No-contingent-change rule

After the first E16 run begins, do not alter the seeds, architecture, dataset,
feature cache, optimiser, schedule, epochs, checkpoint rule, horizon grid,
gain, early window, criterion definitions, or thresholds in response to
results. Engineering fixes may repair a crash only if they do not change the
frozen scientific protocol and are documented.

## Permitted interpretation

If the criteria pass, E16 supports the specific conclusion that this trained
weight-tied recursive classifier suffers a reproducible terminal-readout
penalty when inference is extended well beyond its training horizon, and that
aligning the training horizon materially reduces that penalty.

E16 does not establish the phenomenon in ConvNeXt itself, Transformers, SSMs,
diffusion models, DEQs, or recursive architectures in general. It is not
evidence that GRACE outperforms simpler readouts.
