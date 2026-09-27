# E15 Aligned-Horizon Recursive Stress Test

**Status:** frozen before the first E15 engineering or scientific result.

**Relationship to E14:** E15 is a new, versioned confirmatory experiment that
corrects the diagnosed train/evaluation horizon mismatch. E14 remains
unchanged and is preserved as a horizon-overcomputation pilot.

## Single permitted change

Keep the E14 data, frozen ConvNeXt-Tiny feature cache, architecture, hidden and
intermediate dimensions, dropout, optimiser, learning rate, weight decay,
schedule, epochs, terminal-only supervision, gain grid, early window,
adaptive/GRACE parameters, fixed-profile control, probe protocol, checkpoint
selection rule, and seed policy unchanged.

Change only:

\[
T_{\rm train}: 8 \longrightarrow 32.
\]

The recurrence is therefore

\[
h_{t+1}
=
h_t + \frac{\lambda}{32}
W_o\left[
\operatorname{SiLU}(W_g\operatorname{LayerNorm}(h_t))
\odot
W_v\operatorname{LayerNorm}(h_t)
\right].
\]

Train with terminal cross-entropy at

\[
T_{\rm train}=32,\qquad \lambda_{\rm train}=1.
\]

Select the checkpoint only by terminal validation accuracy at this same
condition. No intermediate supervision is allowed.

## Frozen evaluation

Evaluate at \(T_{\rm test}=32\) on exactly

\[
\lambda\in
\{0.75,1.00,1.10,1.20,1.30,1.40,1.50,1.60\}.
\]

All E14 readouts and diagnostics remain unchanged:

- terminal state;
- first-eight-state early window;
- uniform full-trajectory average;
- adaptive stopping with norm-ratio threshold 1.5;
- GRACE with \((\tau,\alpha)=(1.2,10)\);
- the training-split fixed GRACE profile computed at \(\lambda=1.3\);
- shared-head timestep metrics;
- fixed-alpha readout probes at every gain;
- dense timestep probes at gains 1.0, 1.3, and 1.6.

## Predeclared continuation gate

The gate uses the practical shared trained head only. Probe results are
secondary diagnostics and cannot rescue a failed shared-head gate.

Run seed 0 first. Continue to seeds 1 and 2 only if all conditions hold:

1. At \(\lambda=1\), terminal accuracy is within 0.5 percentage point of the
   best trajectory-aware shared-head readout.
2. Early-window accuracy exceeds terminal accuracy by at least 1 percentage
   point at three consecutive frozen gains, considering only gains
   \(\lambda\ge1.2\).
3. Between \(\lambda=1.2\) and \(\lambda=1.6\), either:
   - the shared-head early-minus-terminal gap strictly increases; or
   - the timestep with maximum shared-head accuracy shifts strictly earlier.
4. All states and logits are finite, and terminal or early accuracy at
   \(\lambda=1.6\) exceeds 2.5%, five times random chance.
5. No architecture, grid, window, threshold, decoder, or probe setting is
   altered after seed 0.

If the gate passes, run seeds 1 and 2 and apply the existing 2-of-3 sign rule
at the predeclared headline gains 1.0, 1.3, and 1.6. If it fails, stop after
seed 0 without tuning.

## Permitted claim

If the gate and multi-seed rule pass:

> With training and evaluation horizons aligned at 32 recursive steps, the
> terminal state remains competitive at the nominal gain, while controlled
> gain stress produces an increasing endpoint-versus-prefix mismatch in a
> trained weight-tied recursive classifier on frozen ConvNeXt-Tiny Tiny
> ImageNet representations.

Do not claim the effect occurs in ConvNeXt itself, in all recursive models, or
at production scale. Do not use E15 to claim that GRACE improves accuracy over
the matched fixed-profile control.
