# E17C Causal Panel and Robustness Audit

**Status:** frozen before classwise, checkpoint-sensitivity, E16 probe, scale,
or final-panel outputs were generated.

**Governing user instruction SHA-256:**
`4e240d26858b90431ed7991ff464d53838e4456bf9874027eaeb2b5c4db9e176`.

E16 and E17 are immutable. This audit uses only their existing selected or
explicitly saved checkpoints, frozen feature caches, and frozen validation
sets. No model is retrained and no successful scientific output is
overwritten.

## E17C causal classification

For each paired seed:

`interaction = E17_drop - E17C_drop`.

`matched_horizon_restoration` requires all of:

1. E17C terminal accuracy at step 32 is within 0.5 percentage point of its
   best fixed shared-head timestep for every seed.
2. The across-seed mean paired interaction is at least 1.0 percentage point
   and E17C has a smaller terminal drop than E17 for every seed.
3. The paired interaction is positive for every seed.
4. All E17C states and logits are finite.
5. The E17C fixed-probe step-8-minus-step-32 drop is smaller than E17's for
   every paired seed and in the across-seed mean.

`partial_restoration` applies when conditions 2--4 pass but condition 1 or 5
fails. Every other completed outcome is `no_restoration`.

No significance test is used at three seeds.

## Classwise robustness

For E16 T_train=8 and E17 T_train=8, existing selected checkpoints are
evaluated at steps 8 and 32 on the complete labelled 10,000-example validation
split. For every one of the 200 classes, the primary classwise drop is the
mean of the three seed-level class accuracies at step 8 minus the corresponding
mean at step 32. The audit reports the median, 25th and 75th percentiles,
interquartile range, macro mean, count with positive drop, and count with drop
at least 2 percentage points. No class is selected or excluded.

Per-example labels and predictions are saved once so the class table can be
reproduced without another model evaluation.

## Checkpoint sensitivity

The preregistered selected checkpoint remains the headline checkpoint.
Existing saved states only are audited:

- E16 T_train=8: selected checkpoint and saved final-epoch state. Intermediate
  top-three checkpoints were not retained.
- E17 T_train=8: selected checkpoint and up to the top three distinct saved
  improving checkpoints ranked only by their recorded T_train=8 validation
  accuracy. Final-epoch states were not retained.

No missing checkpoint is reconstructed or retrained.

## Fixed-alpha probes

E16 selected checkpoints at T_train=8 and T_train=32 use the exact E17 probe
protocol: the frozen 20,000-example stratified subset with SHA-256
`9f7378e1c3cde52953ea5096b6b80cfb4b52fff8cafcd87be19d9b366590a19d`,
the full validation set, `RidgeClassifier(alpha=1.0)`, and no feature
standardisation. Separate probes are fit to step 8, the mean of steps 1--8,
and step 32.

## Parameter and recurrent-FLOP accounting

Trainable parameter counts are exact `numel()` totals from the instantiated
heads. The executed frozen ConvNeXt-Tiny backbone count excludes the unused
1,000-class ImageNet linear classifier; the full torchvision checkpoint count
is also recorded.

Dominant recurrent FLOPs count one multiply-add as two FLOPs and exclude the
frozen backbone, one-time input projection, and final classifier:

- SwiGLU: `6*d*m`, for gate, value, and output projections.
- Attention:
  `8*n*d^2 + 4*n^2*d + 5*h*n^2 + 6*n*d*m`,
  covering Q/K/V/output projections, QK and AV products, an approximate
  five-operation softmax per attention probability, and SwiGLU projections.

LayerNorm and small elementwise residual/activation costs are reported as
excluded lower-order terms.

## Figures and claim ceiling

The final panel contains exactly the four frozen model conditions and at most
three main figures. It does not claim that ConvNeXt is iterative, that the
attention head is a complete Universal Transformer, or that endpoint
invalidity is universal.

