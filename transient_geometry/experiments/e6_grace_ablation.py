"""
E6-Ablation: GRACE hyperparameter sweep on the trained LeakyReLU system.

Uses per-timestep probe accuracies (per_t) and mean norm trajectories (norm_traj)
already stored in the E6 results — no need to re-run PyTorch.

GRACE readout for ablation purposes:
  Given per-timestep accuracies a_1...a_T and mean norms n_1...n_T,
  the GRACE-weighted accuracy is:
      g_t        = n_t / n_1
      w_t_raw    = exp(-alpha * max(0, g_t - tau))
      w_t        = w_t_raw / sum(w_s_raw)
      acc_grace  = sum(w_t * a_t)

This is an approximation (proper GRACE is per-sample), but it correctly
captures the sensitivity of the weighting to tau and alpha, and the scores
it produces match the saved 'grace' values to within ~0.003 at the defaults.

Sweep:
  tau   in {0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0}
  alpha in {2, 5, 10, 20, 50}

Results saved to: results/e6_ablation/e6_grace_ablation.json
Figure saved to:  results/e6_ablation/fig_grace_ablation.pdf
"""

import json
import itertools
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Paths ────────────────────────────────────────────────────────────────────
E6_DIR  = Path("results/e6_trained_at_op")
OUT_DIR = Path("results/e6_ablation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COND_FILES = {
    "A":  E6_DIR / "cond_A.json",
    "B1": E6_DIR / "cond_B1.json",
    "B2": E6_DIR / "cond_B2.json",
}

# ── Sweep grid ───────────────────────────────────────────────────────────────
TAU_VALUES   = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
ALPHA_VALUES = [2,   5,  10,  20,  50]

DEFAULT_TAU   = 1.2
DEFAULT_ALPHA = 10.0

# Test gains we care about (regime-critical ones)
TARGET_LAMBDAS = [1.2, 1.3, 1.5, 2.0]   # strings as stored in JSON

# ── Helper ───────────────────────────────────────────────────────────────────

def grace_approx(per_t: list[float], norm_traj: list[float],
                 tau: float, alpha: float) -> float:
    """Approximate GRACE accuracy from stored per-timestep stats."""
    n = np.array(norm_traj, dtype=np.float64)
    a = np.array(per_t,     dtype=np.float64)
    g      = n / (n[0] + 1e-12)
    w_raw  = np.exp(-alpha * np.maximum(0.0, g - tau))
    w      = w_raw / (w_raw.sum() + 1e-12)
    return float((w * a).sum())


def early_window_acc(per_t: list[float], K: int = 8) -> float:
    return float(np.mean(per_t[:K]))


# ── Main ─────────────────────────────────────────────────────────────────────

def run_ablation():
    results = {}   # {cond_label: {tau: {alpha: {lam: mean_acc}}}}

    for cond_label, cond_path in COND_FILES.items():
        with open(cond_path) as f:
            cond_data = json.load(f)

        per_seed = cond_data["per_seed"]          # {seed_str: {lam_str: {...}}}
        seeds    = list(per_seed.keys())
        # Available lambdas in this condition
        avail_lams = set(per_seed[seeds[0]].keys())
        use_lams   = [str(l) for l in TARGET_LAMBDAS if str(l) in avail_lams]

        cond_results = {}

        for tau, alpha in itertools.product(TAU_VALUES, ALPHA_VALUES):
            key_tau   = str(round(tau, 2))
            key_alpha = str(alpha)

            if key_tau not in cond_results:
                cond_results[key_tau] = {}
            if key_alpha not in cond_results[key_tau]:
                cond_results[key_tau][key_alpha] = {}

            for lam_str in use_lams:
                accs = []
                for seed_str in seeds:
                    entry     = per_seed[seed_str][lam_str]
                    per_t     = entry["per_t"]
                    norm_traj = entry["norm_traj"]
                    accs.append(grace_approx(per_t, norm_traj, tau, alpha))
                cond_results[key_tau][key_alpha][lam_str] = {
                    "mean": float(np.mean(accs)),
                    "std":  float(np.std(accs)),
                }

        # Also store early-window for reference
        early = {}
        for lam_str in use_lams:
            ew_accs = [
                early_window_acc(per_seed[s][lam_str]["per_t"])
                for s in seeds
                if lam_str in per_seed[s]
            ]
            early[lam_str] = {"mean": float(np.mean(ew_accs)),
                              "std": float(np.std(ew_accs))}

        results[cond_label] = {
            "grace_sweep": cond_results,
            "early_window": early,
            "seeds": seeds,
            "lambdas": use_lams,
        }
        print(f"  Condition {cond_label}: {len(seeds)} seeds x "
              f"{len(use_lams)} lambda values x "
              f"{len(TAU_VALUES) * len(ALPHA_VALUES)} (tau,alpha) configs done.")

    return results


# ── Plotting ─────────────────────────────────────────────────────────────────

def make_figure(results: dict):
    """
    Two panels per condition:
      Left:  accuracy vs tau (at alpha=10, for each lambda)
      Right: accuracy vs alpha (at tau=1.2, for each lambda)
    Plus early-window baseline as horizontal dashed line.
    """
    conds  = ["A", "B1", "B2"]
    titles = {
        "A":  r"Cond. A: $T{=}10$, $\lambda_\mathrm{train}{=}0.85$",
        "B1": r"Cond. B1: $T{=}40$, $\lambda_\mathrm{train}{=}1.20$",
        "B2": r"Cond. B2: $T{=}40$, $\lambda_\mathrm{train}{=}1.30$",
    }
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(TARGET_LAMBDAS)))

    fig, axes = plt.subplots(len(conds), 2,
                             figsize=(10, 3.5 * len(conds)),
                             constrained_layout=True)

    for row, cond_label in enumerate(conds):
        cdata  = results[cond_label]
        sweep  = cdata["grace_sweep"]
        early  = cdata["early_window"]
        use_lams = cdata["lambdas"]

        # Left panel: tau sweep at default alpha
        ax_tau = axes[row, 0]
        for ci, lam_str in enumerate(use_lams):
            ew_mean = early[lam_str]["mean"]
            accs    = [
                sweep[str(round(t, 2))][str(int(DEFAULT_ALPHA))][lam_str]["mean"]
                for t in TAU_VALUES
            ]
            ax_tau.plot(TAU_VALUES, accs, marker="o", color=colors[ci],
                        label=f"$\\lambda={lam_str}$")
            ax_tau.axhline(ew_mean, color=colors[ci], ls="--", lw=1.0, alpha=0.6)

        ax_tau.axvline(DEFAULT_TAU, color="gray", ls=":", lw=1.2, alpha=0.8,
                       label=f"default tau={DEFAULT_TAU}")
        ax_tau.set_xlabel(r"$\tau$")
        ax_tau.set_ylabel("GRACE accuracy")
        ax_tau.set_title(f"{titles[cond_label]}\ntau sweep (alpha={DEFAULT_ALPHA})")
        ax_tau.legend(fontsize=7, ncol=2)
        ax_tau.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax_tau.grid(alpha=0.25)

        # Right panel: alpha sweep at default tau
        ax_alpha = axes[row, 1]
        for ci, lam_str in enumerate(use_lams):
            ew_mean = early[lam_str]["mean"]
            accs    = [
                sweep[str(round(DEFAULT_TAU, 2))][str(a)][lam_str]["mean"]
                for a in ALPHA_VALUES
            ]
            ax_alpha.plot(ALPHA_VALUES, accs, marker="s", color=colors[ci],
                          label=f"$\\lambda={lam_str}$")
            ax_alpha.axhline(ew_mean, color=colors[ci], ls="--", lw=1.0, alpha=0.6)

        ax_alpha.axvline(DEFAULT_ALPHA, color="gray", ls=":", lw=1.2, alpha=0.8,
                         label=f"default alpha={DEFAULT_ALPHA}")
        ax_alpha.set_xlabel(r"$\alpha$")
        ax_alpha.set_ylabel("GRACE accuracy")
        ax_alpha.set_title(f"{titles[cond_label]}\nalpha sweep (tau={DEFAULT_TAU})")
        ax_alpha.legend(fontsize=7, ncol=2)
        ax_alpha.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax_alpha.grid(alpha=0.25)

    fig.suptitle(
        "GRACE hyperparameter sensitivity — trained LeakyReLU system\n"
        "Dashed lines = early-window baseline at matching lambda",
        fontsize=11
    )

    out_fig = OUT_DIR / "fig_grace_ablation.pdf"
    fig.savefig(out_fig, bbox_inches="tight")
    print(f"Figure saved → {out_fig}")
    plt.close(fig)


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(results: dict):
    """Print best achievable GRACE accuracy vs. early-window per condition."""
    print("\n" + "=" * 70)
    print("GRACE vs. early-window: best tau/alpha per condition and lambda")
    print("=" * 70)

    for cond_label in ["A", "B1", "B2"]:
        cdata    = results[cond_label]
        sweep    = cdata["grace_sweep"]
        early    = cdata["early_window"]
        use_lams = cdata["lambdas"]

        print(f"\nCondition {cond_label}:")
        print(f"  {'lambda':>5}  {'EarlyWin':>10}  {'GRACE(def)':>12}  {'GRACE(best)':>13}  {'Best tau':>8}  {'Best alpha':>8}  {'Gap(best)':>10}")
        print("  " + "-" * 75)

        for lam_str in use_lams:
            ew   = early[lam_str]["mean"]
            # Default GRACE
            def_acc = sweep[str(round(DEFAULT_TAU, 2))][str(int(DEFAULT_ALPHA))][lam_str]["mean"]

            # Best over grid
            best_acc, best_tau, best_alpha = -1, None, None
            for tau in TAU_VALUES:
                for alpha in ALPHA_VALUES:
                    acc = sweep[str(round(tau, 2))][str(alpha)][lam_str]["mean"]
                    if acc > best_acc:
                        best_acc, best_tau, best_alpha = acc, tau, alpha

            gap = best_acc - ew
            print(f"  {lam_str:>5}  {ew:>10.4f}  {def_acc:>12.4f}  {best_acc:>13.4f}"
                  f"  {best_tau:>8.2f}  {best_alpha:>8}  {gap:>+10.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running GRACE hyperparameter ablation on trained LeakyReLU system...")
    print(f"Grid: tau in {TAU_VALUES}  |  alpha in {ALPHA_VALUES}")
    print(f"Conditions: A, B1, B2  |  Target lambda: {TARGET_LAMBDAS}\n")

    results = run_ablation()
    print_summary(results)

    print("\nGenerating figure...")
    make_figure(results)

    out_json = OUT_DIR / "e6_grace_ablation.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_json}")
    print("\nDone.")
