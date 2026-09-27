"""Generate E7 CIFAR-10 figure matching paper style."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent.parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Match paper global style exactly ─────────────────────────────────────
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          13,
    "axes.titlesize":     15,
    "axes.titleweight":   "bold",
    "axes.labelsize":     13,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    12,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.8",
    "lines.linewidth":    2.5,
    "lines.markersize":   8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.15,
    "text.usetex":        False,
})

# Paper colour palette
C = dict(
    final    = "#e05263",
    early    = "#2d8f4e",
    adaptive = "#e07b39",
    grace    = "#2166ac",
)
LABELS = dict(
    final    = "Terminal state",
    early    = "Early window",
    adaptive = "Adaptive stopping",
    grace    = "GRACE",
)
MARKERS = dict(final="o", early="s", adaptive="^", grace="D")

# ── Load data ─────────────────────────────────────────────────────────────
with open(ROOT / "results/e7_cifar10/e7_cifar10.json") as f:
    d = json.load(f)

summary = d["summary"]
radii   = np.array(d["spectral_radii"])

means = {m: np.array([summary[str(round(r,2))][m]["mean"] * 100 for r in radii])
         for m in C}
stds  = {m: np.array([summary[str(round(r,2))][m]["std"]  * 100 for r in radii])
         for m in C}

# ── Figure: two panels ────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

# Panel 1: accuracy vs rho
for m in ("final", "early", "adaptive", "grace"):
    ax1.plot(radii, means[m], color=C[m], marker=MARKERS[m],
             label=LABELS[m], zorder=3)
    ax1.fill_between(radii, means[m] - stds[m], means[m] + stds[m],
                     color=C[m], alpha=0.15)

ax1.axvline(x=1.0, color="black", linestyle=":", linewidth=1.2,
            label="$\\rho = 1$", alpha=0.6)
ax1.set_xlabel("Spectral radius $\\rho$")
ax1.set_ylabel("Test accuracy (%)")
ax1.set_title("Accuracy vs. Spectral Radius")
ax1.set_ylim(0, 48)
# Legend below the plot to avoid overlapping data
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
           ncol=3, framealpha=0.9)

# Panel 2: advantage over final state
gap_max = 0
for m in ("early", "adaptive", "grace"):
    gap = means[m] - means["final"]
    gap_max = max(gap_max, gap.max())
    ax2.plot(radii, gap, color=C[m], marker=MARKERS[m],
             label=f"{LABELS[m]}", zorder=3)

ax2.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.5)
ax2.axvline(x=1.0, color="black", linestyle=":", linewidth=1.2, alpha=0.6,
            label="$\\rho = 1$")
ax2.set_xlabel("Spectral radius $\\rho$")
ax2.set_ylabel("Advantage over terminal (pp)")
ax2.set_title("Advantage over Terminal-State Readout")
ax2.set_ylim(-2, gap_max + 5)
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
           ncol=3, framealpha=0.9)

plt.tight_layout()
fig.subplots_adjust(bottom=0.22)   # room for below-axis legends

# Save
for ext in ("pdf", "png"):
    p = FIG_DIR / f"fig_cifar10_regime.{ext}"
    fig.savefig(p)
    print(f"Saved: {p}")

import shutil
shutil.copy(FIG_DIR / "fig_cifar10_regime.png",
            ROOT / "results/e7_cifar10/fig_e7_cifar10.png")
plt.close()
