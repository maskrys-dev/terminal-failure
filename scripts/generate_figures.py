"""
Generate publication figures from existing JSON results.
No experiment re-runs. Saves PDF + PNG to figures/.
"""
import json, sys
import numpy as np
from pathlib import Path
from collections import defaultdict
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT    = Path(__file__).resolve().parent.parent
RES     = ROOT / "results"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────────────
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

# Colour-blind friendly palette
C = dict(
    final    = "#e05263",   # warm red
    early    = "#2d8f4e",   # forest green
    adaptive = "#e07b39",   # burnt orange
    grace    = "#2166ac",   # strong blue
    oracle   = "#7b2d8b",   # purple
    modrelu  = "#2166ac",
    tanh     = "#e07b39",
    linear   = "#2d8f4e",
)

GRACE_LABEL    = "GRACE"
TERMINAL_LABEL = "Terminal state"
EARLY_LABEL    = "Early window"
ADAPTIVE_LABEL = "Adaptive stopping"


def fig0_precollapse_schematic():
    """Conceptual schematic for the pre-collapse readout regime."""
    print("Generating Fig 0: Conceptual pre-collapse schematic...")
    from matplotlib.patches import Ellipse

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.35), sharex=True, sharey=True)
    green = C["early"]
    red = C["final"]
    purple = C["oracle"]
    grey = "0.55"

    def setup(ax, title):
        ax.set_title(title, fontsize=12, pad=7)
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("0.75")
        ax.axvline(0, color="0.35", lw=1.3, ls=":")
        ax.text(0.03, 0.92, "decision boundary", fontsize=8, color="0.35")
        ax.add_patch(Ellipse((-0.68, 0.08), 0.84, 0.82, angle=8,
                             facecolor=green, edgecolor=green, alpha=0.10, lw=1.3))
        ax.add_patch(Ellipse((0.72, -0.06), 0.86, 0.80, angle=-6,
                             facecolor=purple, edgecolor=purple, alpha=0.08, lw=1.3))
        ax.text(-0.96, 0.62, "class A manifold", fontsize=8.5, color=green)
        ax.text(0.34, -0.63, "other class", fontsize=8.5, color=purple)

    setup(axes[0], "Stable trajectory")
    setup(axes[1], "Pre-collapse trajectory")

    t = np.linspace(0, 1, 18)
    x_stable = -0.98 + 0.48 * t + 0.06 * np.sin(2 * np.pi * t)
    y_stable = -0.38 + 0.78 * t - 0.08 * np.sin(np.pi * t)
    axes[0].plot(x_stable, y_stable, color="0.25", lw=1.7, zorder=3)
    axes[0].scatter(x_stable[:7], y_stable[:7], s=30, color=green, edgecolor="white",
                    lw=0.4, zorder=4)
    axes[0].scatter(x_stable[-1], y_stable[-1], marker="*", s=130, color=green,
                    edgecolor="white", lw=0.7, zorder=5)
    axes[0].annotate("prefix and endpoint agree",
                     xy=(x_stable[-1], y_stable[-1]), xytext=(-1.23, -0.82),
                     fontsize=8.8, color="0.25",
                     arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0))
    axes[0].annotate("time", xy=(-0.43, 0.39), xytext=(-0.88, -0.22),
                     fontsize=8.5, color="0.3",
                     arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0))

    s = np.linspace(0, 1, 22)
    x_pre = -1.02 + 1.86 * s + 0.11 * np.sin(3.4 * np.pi * s)
    y_pre = -0.37 + 0.72 * np.sin(1.1 * np.pi * s) - 0.42 * s
    axes[1].plot(x_pre, y_pre, color="0.25", lw=1.7, zorder=3)
    axes[1].scatter(x_pre[:8], y_pre[:8], s=30, color=green, edgecolor="white",
                    lw=0.4, zorder=4)
    axes[1].scatter(x_pre[8:16], y_pre[8:16], s=22, color=grey, edgecolor="white",
                    lw=0.3, zorder=4)
    axes[1].scatter(x_pre[-1], y_pre[-1], marker="X", s=105, color=red,
                    edgecolor="white", lw=0.8, zorder=5)
    axes[1].annotate("usable prefix",
                     xy=(x_pre[5], y_pre[5]), xytext=(-1.22, 0.74),
                     fontsize=8.8, color=green,
                     arrowprops=dict(arrowstyle="->", color=green, lw=1.0))
    axes[1].annotate("terminal state degraded",
                     xy=(x_pre[-1], y_pre[-1]), xytext=(0.19, -0.86),
                     fontsize=8.8, color=red,
                     arrowprops=dict(arrowstyle="->", color=red, lw=1.0))
    axes[1].annotate("time", xy=(0.38, 0.18), xytext=(-0.65, -0.21),
                     fontsize=8.5, color="0.3",
                     arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0))

    fig.tight_layout(pad=0.7)
    save(fig, "fig0_precollapse_schematic")

# ── Helpers ───────────────────────────────────────────────────────────────────
def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  Saved {name}  ({(FIG_DIR / name).with_suffix('.png').stat().st_size // 1024} KB)")


def copy_result_figure(result_dir, result_stem, paper_stem):
    src_dir = RES / result_dir
    copied = []
    for ext in ("png", "pdf"):
        src = src_dir / f"{result_stem}.{ext}"
        dst = FIG_DIR / f"{paper_stem}.{ext}"
        if src.exists():
            shutil.copyfile(src, dst)
            copied.append(ext)
    if copied:
        print(f"  Staged {paper_stem} from {result_dir} ({', '.join(copied)})")
    else:
        raise FileNotFoundError(f"No saved figure found for {result_dir}/{result_stem}")


def agg_by(records, group_key, val_key):
    d = defaultdict(list)
    for r in records:
        d[r[group_key]].append(r[val_key])
    return {k: (np.mean(v), np.std(v)) for k, v in sorted(d.items())}


def plot_line(ax, data_dict, color, label, marker="o", ls="-", lw=2.5, ms=8, alpha_band=0.15):
    xs  = sorted(data_dict.keys())
    ys  = [data_dict[x][0] for x in xs]
    err = [data_dict[x][1] for x in xs]
    ax.plot(xs, ys, marker=marker, color=color, label=label,
            ls=ls, lw=lw, ms=ms, zorder=4)
    ax.fill_between(xs, [y-e for y,e in zip(ys,err)],
                        [y+e for y,e in zip(ys,err)],
                    alpha=alpha_band, color=color, zorder=3)


def fig12_ginibre_spectrum():
    """Visual diagnostic for the complex Ginibre spectral-radius sweep."""
    rng = np.random.default_rng(2604)
    d = 128
    w0 = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / np.sqrt(2 * d)
    base_radius = np.max(np.abs(np.linalg.eigvals(w0)))

    panels = [
        ("Unscaled draw", None),
        (r"Rescaled to $\rho=0.9$", 0.9),
        (r"Rescaled to $\rho=1.3$", 1.3),
        (r"Rescaled to $\rho=1.6$", 1.6),
    ]
    lim = 1.75
    theta = np.linspace(0, 2 * np.pi, 400)

    fig, axes = plt.subplots(1, 4, figsize=(14.8, 3.6), sharex=True, sharey=True)
    for ax, (title, target) in zip(axes, panels):
        w = w0 if target is None else w0 * (target / base_radius)
        vals = np.linalg.eigvals(w)
        circle_radius = base_radius if target is None else target

        ax.scatter(vals.real, vals.imag, s=10, alpha=0.7, color=C["grace"], edgecolors="none")
        ax.plot(np.cos(theta), np.sin(theta), color="0.6", ls=":", lw=1.2, label="unit circle")
        ax.plot(
            circle_radius * np.cos(theta),
            circle_radius * np.sin(theta),
            color=C["final"],
            ls="--",
            lw=1.5,
            label="spectral radius",
        )
        ax.axhline(0, color="0.75", lw=0.6)
        ax.axvline(0, color="0.75", lw=0.6)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Real")
    axes[0].set_ylabel("Imaginary")
    axes[-1].legend(loc="upper right", fontsize=9, frameon=True)
    fig.suptitle(
        "Complex Ginibre Spectral Clouds Under Radial Rescaling",
        y=1.02,
        fontweight="bold",
    )
    fig.tight_layout(pad=0.5, w_pad=0.9)
    save(fig, "fig12_ginibre_spectrum")


# =============================================================================
# Fig 1 — Phase diagram: accuracy vs spectral radius (10-seed)
# =============================================================================
def fig1_phase_diagram():
    print("Generating Fig 1: Phase diagram...")
    records = json.loads((RES / "overnight_10seed" / "results.json").read_text())

    d_final    = agg_by(records, "spectral_radius", "acc_final")
    d_early    = agg_by(records, "spectral_radius", "acc_early_window")
    d_adaptive = agg_by(records, "spectral_radius", "acc_adaptive")
    d_grace    = agg_by(records, "spectral_radius", "acc_d1_best")

    # Single-axis figure: compact enough not to dominate the main-text page.
    fig, ax = plt.subplots(figsize=(7.2, 4.25))

    plot_line(ax, d_final,    C["final"],    TERMINAL_LABEL,  marker="s", ls="--")
    plot_line(ax, d_early,    C["early"],    EARLY_LABEL,     marker="^")
    plot_line(ax, d_adaptive, C["adaptive"], ADAPTIVE_LABEL,  marker="D", ls="-.")
    plot_line(ax, d_grace,    C["grace"],    GRACE_LABEL,     marker="o")

    ax.axvline(1.0, color="0.5", ls=":", lw=1.5, zorder=2)
    ax.text(1.01, 0.135, "ρ = 1  (stability boundary)",
            fontsize=10, color="0.4", va="bottom", style="italic")

    # shade the "pre-collapse regime"
    ax.axvspan(1.30, 1.65, alpha=0.06, color=C["final"], zorder=1)
    ax.text(1.45, 0.22, "Pre-collapse\nregime", fontsize=9.5,
            color=C["final"], ha="center", style="italic")

    ax.set_xlabel("Spectral radius ρ(W)")
    ax.set_ylabel("Classification accuracy")
    ax.set_ylim(0.12, 0.87)
    ax.set_title("The Pre-Collapse Regime: Accuracy vs Spectral Radius\n"
                 "Complex modReLU system on MNIST  (10 seeds, shading = ±1 std)")
    ax.legend(loc="center left", bbox_to_anchor=(0.01, 0.45), fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    fig.tight_layout()
    save(fig, "fig1_phase_diagram")


# =============================================================================
# Fig 2 — Grace period length vs spectral radius
# =============================================================================
def fig2_grace_period():
    print("Generating Fig 2: True grace period + norm proxy...")
    g_data = json.loads((RES / "grace_period" / "accuracy_grace_period.json").read_text())
    records = json.loads((RES / "overnight_10seed" / "results.json").read_text())
    d_stop = agg_by(records, "spectral_radius", "adaptive_stop_t")

    alpha_cfg = [
        ("0.65", "#2d8f4e", r"$\alpha = 0.65$"),
        ("0.70", "#2166ac", r"$\alpha = 0.70$"),
        ("0.75", "#e05263", r"$\alpha = 0.75$"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), sharey=True)

    ax = axes[0]
    for alpha_key, color, label in alpha_cfg:
        series = g_data["aggregated_G"][alpha_key]
        rhos = sorted(float(r) for r in series.keys())
        means = [series[f"{rho:.2f}"]["mean"] for rho in rhos]
        stds = [series[f"{rho:.2f}"]["std"] for rho in rhos]
        ax.plot(rhos, means, marker="o", color=color, label=label, lw=2.5, ms=8, zorder=4)
        ax.fill_between(
            rhos,
            [max(1, m - s) for m, s in zip(means, stds)],
            [min(40, m + s) for m, s in zip(means, stds)],
            alpha=0.15,
            color=color,
            zorder=3,
        )

    ax.axhline(40, color="0.5", ls=":", lw=1.3)
    ax.axvline(1.0, color="0.5", ls=":", lw=1.3, zorder=2)
    ax.set_xlabel("Spectral radius ρ(W)")
    ax.set_ylabel("Usable prefix length (timesteps)")
    ax.set_ylim(0, 45)
    ax.set_title("Accuracy-Defined Grace Period  $G(\\rho, \\alpha)$")
    ax.legend(loc="lower left")

    ax = axes[1]
    rhos = sorted(d_stop.keys())
    means = [min(40.0, d_stop[r][0]) for r in rhos]
    stds = [d_stop[r][1] for r in rhos]
    ax.plot(rhos, means, "o-", color=C["adaptive"], lw=2.5, ms=8, zorder=4)
    ax.fill_between(
        rhos,
        [max(1, m - s) for m, s in zip(means, stds)],
        [min(40, m + s) for m, s in zip(means, stds)],
        alpha=0.18,
        color=C["adaptive"],
        zorder=3,
    )
    ax.axhline(40, color="0.5", ls=":", lw=1.3)
    ax.axvline(1.0, color="0.5", ls=":", lw=1.3, zorder=2)
    ax.set_xlabel("Spectral radius ρ(W)")
    ax.set_title("Norm-Growth Stopping Proxy")

    fig.suptitle(
        "Grace Period Diagnostics\n"
        "Left: formal accuracy-defined quantity  |  Right: online proxy from norm growth",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig2_grace_period")


# =============================================================================
# Fig 3 — GRACE method: accuracy + weight profiles (Section 5.2)
# =============================================================================
def fig3_grace_method():
    print("Generating Fig 3: GRACE method + weight profiles...")
    records  = json.loads((RES / "overnight_10seed" / "results.json").read_text())
    wp_raw   = json.loads((RES / "d1_cross_system" / "weight_profiles.json").read_text())

    # aggregate accuracy
    d_final = agg_by(records, "spectral_radius", "acc_final")
    d_early = agg_by(records, "spectral_radius", "acc_early_window")
    d_adapt = agg_by(records, "spectral_radius", "acc_adaptive")
    d_grace = agg_by(records, "spectral_radius", "acc_d1_best")

    # weight profiles at rho = 1.0, 1.3, 1.5  (tau=1.2, alpha=10)
    rho_labels = {
        1.0: r"$\rho = 1.0$ (stable)",
        1.3: r"$\rho = 1.3$ (transition)",
        1.5: r"$\rho = 1.5$ (unstable)",
    }
    wp_colors  = {1.0: "#2166ac", 1.3: "#e07b39", 1.5: "#e05263"}
    profiles = {}
    for (sys, rho, tau_x10, alpha) in [(eval(k)) for k in wp_raw.keys()]:
        if sys == "complex_modrelu" and abs(tau_x10 - 1.2) < 0.01 and alpha == 10 and rho in rho_labels:
            profiles[rho] = np.array(wp_raw[str((sys, rho, tau_x10, alpha))])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("GRACE: Regime-Aware Trajectory Readout\n"
                 "Left: accuracy vs spectral radius  |  Right: temporal weight profiles at key operating points",
                 fontsize=15, fontweight="bold", y=1.02)

    # --- Left: accuracy vs rho ---
    ax = axes[0]
    plot_line(ax, d_final, C["final"],    TERMINAL_LABEL,  marker="s", ls="--")
    plot_line(ax, d_early, C["early"],    EARLY_LABEL,     marker="^")
    plot_line(ax, d_adapt, C["adaptive"], ADAPTIVE_LABEL,  marker="D", ls="-.")
    plot_line(ax, d_grace, C["grace"],    GRACE_LABEL,     marker="o")

    ax.axvline(1.0, color="0.5", ls=":", lw=1.5)
    ax.text(1.01, 0.135, "ρ = 1  (stability boundary)", fontsize=10, color="0.4", style="italic")
    ax.axvspan(1.30, 1.65, alpha=0.055, color=C["final"])
    ax.text(1.45, 0.22, "Pre-collapse\nregime", fontsize=9.5, color=C["final"], ha="center", style="italic")

    ax.set_xlabel("Spectral radius ρ(W)")
    ax.set_ylabel("Classification accuracy")
    ax.set_ylim(0.12, 0.87)
    ax.set_title("GRACE Maintains Accuracy Across the Full Spectral Sweep")
    ax.legend(loc="center left", bbox_to_anchor=(0.01, 0.45))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    # --- Right: weight profiles ---
    ax = axes[1]
    timesteps = np.arange(1, 41)
    for rho in [1.0, 1.3, 1.5]:
        if rho in profiles:
            w = profiles[rho]
            color = wp_colors[rho]
            ax.plot(timesteps, w, "o-", color=color, label=rho_labels[rho],
                    lw=2.2, ms=5, zorder=4)
            ax.fill_between(timesteps, 0, w, alpha=0.10, color=color)

    # uniform reference line
    ax.axhline(1/40, color="0.6", ls="--", lw=1.5, label="Uniform weights (1/T)")
    ax.set_xlabel("Trajectory timestep  t")
    ax.set_ylabel("GRACE weight  w_t")
    ax.set_title("Weight Profiles: Front-Loading Increases with Instability")
    leg = ax.legend(loc="upper right", title="Spectral radius")
    leg.get_title().set_fontsize(10)
    ax.set_xlim(0.5, 40.5)

    # annotate the key insight
    ax.annotate("Stable: near-uniform", xy=(20, 1/40),
                xytext=(22, 0.038), fontsize=9.5, color=wp_colors[1.0],
                arrowprops=dict(arrowstyle="->", color=wp_colors[1.0], lw=0.9))
    ax.annotate("Unstable: sharp prefix", xy=(2, profiles[1.5][1] if 1.5 in profiles else 0.25),
                xytext=(8, 0.22), fontsize=9.5, color=wp_colors[1.5],
                arrowprops=dict(arrowstyle="->", color=wp_colors[1.5], lw=0.9))

    fig.tight_layout()
    save(fig, "fig3_grace_method")


# =============================================================================
# Fig 3b — Variance reduction: std(Terminal) vs std(GRACE)  [appendix-ready]
# =============================================================================
def fig3_variance_reduction():
    print("Generating Fig 3: Variance reduction...")
    records = json.loads((RES / "overnight_10seed" / "results.json").read_text())

    d_final = agg_by(records, "spectral_radius", "acc_final")
    d_grace = agg_by(records, "spectral_radius", "acc_d1_best")
    rhos    = sorted(d_final.keys())

    std_f  = [d_final[r][1] for r in rhos]
    std_g  = [d_grace[r][1] for r in rhos]
    ratios = [f / max(g, 1e-6) for f, g in zip(std_f, std_g)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("GRACE Dramatically Reduces Cross-Seed Variance\n"
                 "Complex modReLU system on MNIST  (10 seeds)",
                 fontsize=15, fontweight="bold", y=1.02)

    # --- Left: raw std ---
    ax = axes[0]
    ax.plot(rhos, std_f, "s--", color=C["final"],  label=TERMINAL_LABEL, lw=2.5, ms=8)
    ax.plot(rhos, std_g, "o-",  color=C["grace"],  label=GRACE_LABEL,    lw=2.5, ms=8)
    ax.axvline(1.0, color="0.5", ls=":", lw=1.2)
    ax.set_xlabel("Spectral radius ρ(W)")
    ax.set_ylabel("Std of accuracy across seeds")
    ax.set_title("Cross-Seed Standard Deviation")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # --- Right: variance ratio ---
    ax = axes[1]
    bar_colors = [C["grace"] if r > 2 else "0.72" for r in ratios]
    bars = ax.bar(rhos, ratios, color=bar_colors, width=0.035,
                  edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(1.0, color="0.4", ls="--", lw=1.5, label="No improvement (1×)")
    ax.axvline(1.0, color="0.5", ls=":", lw=1.2)

    # annotate peak
    peak_i = int(np.argmax(ratios))
    ax.annotate(f"{ratios[peak_i]:.0f}×",
                xy=(rhos[peak_i], ratios[peak_i]),
                xytext=(rhos[peak_i] + 0.04, ratios[peak_i] - 2),
                fontsize=13, fontweight="bold", color=C["grace"],
                arrowprops=dict(arrowstyle="->", color=C["grace"], lw=1.2))

    ax.set_xlabel("Spectral radius ρ(W)")
    ax.set_ylabel("Variance ratio  std(Terminal) / std(GRACE)")
    ax.set_title("GRACE Variance Reduction Factor")
    ax.legend()

    fig.tight_layout()
    save(fig, "fig3_variance_reduction")


# =============================================================================
# Fig 4 — Cross-system taxonomy
# =============================================================================
def fig4_cross_system():
    print("Generating Fig 4: Cross-system taxonomy...")
    records = json.loads((RES / "a5_cross_system" / "results.json").read_text())

    sys_cfg = {
        "complex_modrelu": ("Complex modReLU  (unbounded)", C["modrelu"]),
        "real_esn":        ("Real ESN  (tanh, bounded)",    C["tanh"]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Cross-System Taxonomy: Nonlinearity Governs Regime Severity\n"
                 "Complex modReLU vs. Real ESN (tanh)  |  3 seeds; shading = \u00b11 std",
                 fontsize=15, fontweight="bold", y=1.02)

    # --- Left: acc vs rho for each system ---
    ax = axes[0]
    for key, (label, color) in sys_cfg.items():
        recs = [r for r in records if r.get("system") == key]
        if not recs:
            continue
        d_f = agg_by(recs, "spectral_radius", "acc_final")
        d_e = agg_by(recs, "spectral_radius", "acc_early_window")
        plot_line(ax, d_f, color, f"{label}  [terminal]", marker="s", ls="--", ms=7)
        plot_line(ax, d_e, color, f"{label}  [early]",    marker="o", ls="-",  ms=7)

    ax.axvline(1.0, color="0.5", ls=":", lw=1.2)
    ax.set_xlabel("Spectral radius ρ(W)")
    ax.set_ylabel("Classification accuracy")
    ax.set_ylim(0.10, 0.90)
    ax.set_title("Accuracy vs Spectral Radius by System")
    ax.legend(fontsize=10, ncol=1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    # --- Right: advantage bar at rho=1.6 ---
    ax = axes[1]
    rho_t = 1.6
    x_pos, ht, er, bcolors, xlabels = [], [], [], [], []
    for i, (key, (label, color)) in enumerate(sys_cfg.items()):
        recs = [r for r in records
                if r.get("system") == key and abs(r["spectral_radius"] - rho_t) < 0.01]
        if not recs:
            continue
        gaps = [(r["acc_early_window"] - r["acc_final"]) * 100 for r in recs]
        x_pos.append(i)
        ht.append(np.mean(gaps))
        er.append(np.std(gaps))
        bcolors.append(color)
        xlabels.append(label)

    bars = ax.bar(x_pos, ht, yerr=er, color=bcolors, width=0.5,
                  capsize=8, linewidth=0, zorder=3, alpha=0.88,
                  error_kw=dict(elinewidth=2, ecolor="0.3", capthick=2))
    ax.axhline(0, color="0.3", lw=1.2)
    for bar, h, e in zip(bars, ht, er):
        sign = "+" if h >= 0 else ""
        ax.text(bar.get_x() + bar.get_width()/2,
                h + e + 0.8, f"{sign}{h:.1f} pp",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylabel("Early window advantage over terminal state (pp)")
    ax.set_title(f"Grace-Period Advantage at ρ = {rho_t}")

    fig.tight_layout()
    save(fig, "fig4_cross_system")


# =============================================================================
# Fig 5 — Noise robustness: GRACE vs adaptive stopping
# =============================================================================
def fig5_noise_robustness():
    print("Generating Fig 5: Noise robustness...")
    records = json.loads((RES / "c4_noise" / "results.json").read_text())

    noise_levels = [0.0, 0.01, 0.05, 0.10]
    rho_vals     = [1.4, 1.5]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle("GRACE is Robust to Dynamical Noise; Adaptive Stopping Degrades\n"
                 "5 seeds; shading = ±1 std",
                 fontsize=15, fontweight="bold", y=1.02)

    for ax, rho in zip(axes, rho_vals):
        recs = [r for r in records if abs(r["spectral_radius"] - rho) < 0.01]

        mg, sg, ma, sa = [], [], [], []
        for ns in noise_levels:
            sub = [r for r in recs if abs(r["noise_std"] - ns) < 1e-6]
            g = [r["acc_grace"]    for r in sub]
            a = [r["acc_adaptive"] for r in sub]
            mg.append(np.mean(g)); sg.append(np.std(g))
            ma.append(np.mean(a)); sa.append(np.std(a))

        noise_pct = [n * 100 for n in noise_levels]

        ax.plot(noise_pct, mg, "o-",  color=C["grace"],    label=GRACE_LABEL,    lw=2.5, ms=9)
        ax.fill_between(noise_pct, [m-s for m,s in zip(mg,sg)],
                                   [m+s for m,s in zip(mg,sg)],
                        alpha=0.18, color=C["grace"])

        ax.plot(noise_pct, ma, "s--", color=C["adaptive"], label=ADAPTIVE_LABEL, lw=2.5, ms=9)
        ax.fill_between(noise_pct, [m-s for m,s in zip(ma,sa)],
                                   [m+s for m,s in zip(ma,sa)],
                        alpha=0.18, color=C["adaptive"])

        # annotate gap at highest noise
        gap = (mg[-1] - ma[-1]) * 100
        ax.annotate(f"+{gap:.1f} pp\n(GRACE advantage)",
                    xy=(noise_pct[-1], mg[-1]),
                    xytext=(noise_pct[-1] - 6, mg[-1] + 0.004),
                    fontsize=10, color=C["grace"], ha="right",
                    arrowprops=dict(arrowstyle="->", color=C["grace"], lw=1.1))

        ax.set_xlabel("Injected noise std (%)")
        ax.set_ylabel("Classification accuracy")
        ax.set_title(f"Spectral radius ρ = {rho}")
        ax.set_xticks(noise_pct)
        ax.set_xticklabels([f"{n:.0f}%" if n > 0 else "0%" for n in noise_pct])
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        ax.legend(loc="lower left")

    fig.tight_layout()
    save(fig, "fig5_noise_robustness")


# =============================================================================
# Fig 6 — E6: Training at operating point postpones but does not suppress
#             the pre-collapse regime  (Sec 6.2)
# =============================================================================
def fig6_trained_system():
    print("Generating Fig 6: Trained system — 3-condition E6 figure...")

    CONDITIONS = [
        dict(name="cond_A",  lam_train=0.85, T_train=10,
             label="A: Short-horizon reference\n($T_{\\mathrm{train}}$=10, $\\lambda_{\\mathrm{train}}$=0.85)"),
        dict(name="cond_B1", lam_train=1.20, T_train=40,
             label="B1: Trained at operating point\n($T_{\\mathrm{train}}$=40, $\\lambda_{\\mathrm{train}}$=1.20)"),
        dict(name="cond_B2", lam_train=1.30, T_train=40,
             label="B2: Trained at operating point\n($T_{\\mathrm{train}}$=40, $\\lambda_{\\mathrm{train}}$=1.30)"),
    ]

    METRICS = ["final", "early", "adaptive", "grace"]

    # Load all three conditions
    cond_data = []
    for cond in CONDITIONS:
        d = json.loads((RES / "e6_trained_at_op" / f"{cond['name']}.json").read_text())
        s = d["summary"]
        lams = sorted(float(k) for k in s.keys())
        cond_data.append((lams, s))

    global_final = []
    global_early = []
    for lams, s in cond_data:
        global_final.extend([s[str(lam)]["final"]["mean"] for lam in lams])
        global_early.extend([s[str(lam)]["early"]["mean"] for lam in lams])
    global_y_min = max(0.55, min(global_final) - 0.02)
    global_y_max = min(1.01, max(global_early) + 0.015)

    # The paper scales this to text width. Use a wide intrinsic canvas and
    # remove duplicate y-axis text so the three columns have real plotting room.
    fig, axes = plt.subplots(
        2, 3,
        figsize=(12.6, 6.7),
        gridspec_kw={"height_ratios": [1.0, 1.05]},
    )

    method_cfg = [
        ("final",    TERMINAL_LABEL,  C["final"],    "s", "--"),
        ("early",    EARLY_LABEL,     C["early"],    "^", "-"),
        ("adaptive", ADAPTIVE_LABEL,  C["adaptive"], "D", "-."),
        ("grace",    GRACE_LABEL,     C["grace"],    "o", "-"),
    ]

    for col, (cond, (lams, s)) in enumerate(zip(CONDITIONS, cond_data)):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        # ── Top: accuracy vs lambda ────────────────────────────────────────
        for mkey, label, color, marker, ls in method_cfg:
            ys  = [s[str(lam)][mkey]["mean"] for lam in lams]
            err = [s[str(lam)][mkey]["std"]  for lam in lams]
            ax_top.plot(lams, ys, marker=marker, color=color, label=label,
                        ls=ls, lw=2.0, ms=5.4, zorder=4)
            ax_top.fill_between(lams,
                                [y - e for y, e in zip(ys, err)],
                                [y + e for y, e in zip(ys, err)],
                                alpha=0.12, color=color, zorder=3)

        # Training lambda marker
        ax_top.axvline(cond["lam_train"], color="0.35", ls=":", lw=1.45)
        ax_top.text(cond["lam_train"] + 0.03,
                    min(s[str(lams[0])]["final"]["mean"] for _ in [1]) + 0.005,
                    "$\\lambda_{\\mathrm{train}}$", fontsize=10.0,
                    color="0.35", style="italic", va="bottom")

        # Shade supercritical region
        ax_top.axvspan(1.0, max(lams) + 0.05, alpha=0.04, color=C["final"])

        ax_top.set_xlim(min(lams) - 0.05, max(lams) + 0.05)
        ax_top.set_ylim(global_y_min, global_y_max)

        ax_top.set_title(cond["label"], fontsize=13.2, fontweight="bold")
        ax_top.set_xlabel("")
        ax_top.set_ylabel("Classification accuracy" if col == 0 else "", fontsize=12.0)
        ax_top.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax_top.tick_params(axis="both", labelsize=11.0)
        ax_top.tick_params(axis="x", labelbottom=False)
        if col > 0:
            ax_top.tick_params(axis="y", labelleft=False)
        if col == 0:
            ax_top.legend(loc="lower left", fontsize=11.0, framealpha=0.9)

        # ── Bottom: advantage bar (early – final) ─────────────────────────
        gaps = [(s[str(lam)]["early"]["mean"] - s[str(lam)]["final"]["mean"]) * 100
                for lam in lams]
        gerr = [np.sqrt(s[str(lam)]["early"]["std"]**2 + s[str(lam)]["final"]["std"]**2) * 100
                for lam in lams]
        bar_cols = [C["early"] if g >= 0 else "#e05263" for g in gaps]

        ax_bot.bar(range(len(lams)), gaps, yerr=gerr,
                   color=bar_cols, width=0.68, capsize=5, alpha=0.85,
                   linewidth=0, error_kw=dict(elinewidth=1.55, ecolor="0.4",
                   capthick=1.55), zorder=3)
        ax_bot.axhline(0, color="0.3", lw=1.3)

        # Training lambda marker on bar chart
        if cond["lam_train"] in lams:
            ti = lams.index(cond["lam_train"])
            ax_bot.axvline(ti + 0.5, color="0.35", ls=":", lw=1.35)

        # Annotate maximum gap
        mx = int(np.argmax(gaps))
        if gaps[mx] > 0.5:
            ax_bot.annotate(f"+{gaps[mx]:.1f} pp",
                            xy=(mx, gaps[mx]),
                            xytext=(mx - 0.8, gaps[mx] + 1.0),
                            fontsize=10.2, fontweight="bold", color=C["early"],
                            arrowprops=dict(arrowstyle="->", color=C["early"], lw=1.0))

        ax_bot.set_xticks(range(len(lams)))
        ax_bot.set_xticklabels([str(l) for l in lams], fontsize=10.4, rotation=45)
        ax_bot.set_xlabel("Test-time gain  $\\lambda$", fontsize=12.0)
        ax_bot.set_ylabel("Early window - Final (pp)" if col == 0 else "", fontsize=12.0)
        y_lim = max(30, max(abs(g) for g in gaps) + 3)
        ax_bot.set_ylim(-y_lim * 0.35, y_lim)
        ax_bot.tick_params(axis="y", labelsize=11.0)
        if col > 0:
            ax_bot.tick_params(axis="y", labelleft=False)
        ax_bot.grid(alpha=0.25, axis="y")

    fig.subplots_adjust(left=0.055, right=0.998, bottom=0.155, top=0.865,
                        wspace=0.07, hspace=0.12)
    save(fig, "fig6_trained_system")


def fig7_k_sensitivity():
    print("Generating Fig 7: Early-window K sensitivity...")
    data = json.loads((RES / "revision_readout_audits" / "readout_audits.json").read_text())

    rho_cfg = [
        ("1.30", "#2166ac", r"$\rho = 1.30$"),
        ("1.45", "#e07b39", r"$\rho = 1.45$"),
        ("1.60", "#e05263", r"$\rho = 1.60$"),
    ]
    ks = [int(k) for k in data["k_summary"].keys()]

    fig, ax = plt.subplots(figsize=(8.5, 5.7))
    for rho_key, color, label in rho_cfg:
        means = [data["k_summary"][str(k)][rho_key]["mean"] for k in ks]
        stds = [data["k_summary"][str(k)][rho_key]["std"] for k in ks]
        ax.plot(ks, means, marker="o", color=color, label=label, lw=2.5, ms=8, zorder=4)
        ax.fill_between(
            ks,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.14,
            color=color,
            zorder=3,
        )

    ax.axvline(10, color="0.45", ls=":", lw=1.4)
    ax.text(10.3, 0.756, "Main-text default  $K=10$", fontsize=10, color="0.4", style="italic")
    ax.set_xlabel("Early-window size  $K$")
    ax.set_ylabel("Classification accuracy")
    ax.set_ylim(0.75, 0.835)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("Early-Window Sensitivity in the Unstable Regime")
    ax.legend(loc="lower left")

    fig.tight_layout()
    save(fig, "fig7_k_sensitivity")


def fig8_separability_over_time():
    print("Generating Fig 8: Separability over time...")
    data = json.loads((RES / "pilot" / "separability.json").read_text())
    cfg = [
        ("contractive", "#2d8f4e", "Contractive"),
        ("near_critical", "#2166ac", "Near-critical"),
        ("unstable", "#e05263", "Unstable"),
    ]

    fig, ax = plt.subplots(figsize=(8.7, 5.7))
    t = np.arange(1, 41)
    for key, color, label in cfg:
        rho, curve = data[key]
        ax.plot(t, curve, color=color, lw=2.5, label=f"{label}  ($\\rho={rho:.1f}$)")

    ax.set_xlabel("Timestep  $t$")
    ax.set_ylabel("Per-timestep probe accuracy")
    ax.set_xlim(1, 40)
    ax.set_ylim(0.18, 0.86)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("Separability Persists Early and Collapses Late in the Unstable Regime")
    ax.legend(loc="lower left")

    fig.tight_layout()
    save(fig, "fig8_separability_over_time")


def fig9_deq_relevance():
    print("Generating Fig 9: DEQ relevance check...")
    data = json.loads((RES / "e2_deq_bifurcation" / "e2_deq_bifurcation.json").read_text())
    summary = data["summary"]
    lams = sorted(float(k) for k in summary.keys())

    fig, ax = plt.subplots(figsize=(8.6, 5.7))
    series_cfg = [
        ("final_probe", C["final"], "s", "--", "Final iterate"),
        ("early", C["early"], "^", "-", f"Early window (K={data['config']['early_K']})"),
        ("adaptive", C["adaptive"], "D", "-.", ADAPTIVE_LABEL),
        ("grace", C["grace"], "o", "-", GRACE_LABEL),
    ]
    for key, color, marker, ls, label in series_cfg:
        means = [summary[str(lam)][key]["mean"] for lam in lams]
        stds = [summary[str(lam)][key]["std"] for lam in lams]
        ax.plot(lams, means, marker=marker, color=color, ls=ls, lw=2.5, ms=8, label=label, zorder=4)
        ax.fill_between(
            lams,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.12,
            color=color,
            zorder=3,
        )

    ax.set_xlabel("Test-time gain  $\\lambda$")
    ax.set_ylabel("Classification accuracy")
    ax.set_ylim(0.89, 0.925)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("DEQ-Style Relevance Check Near the Solver Stability Boundary")
    ax.legend(loc="lower left")

    fig.tight_layout()
    save(fig, "fig9_deq_relevance")


def fig10_operator_diagnostics():
    print("Generating Fig 10: Operator diagnostics...")
    from transient_geometry.experiments import e10_operator_diagnostics as e10

    modrelu_records = json.loads(
        (RES / "e10_operator_diagnostics" / "modrelu_per_run.json").read_text()
    )
    cross_records = json.loads(
        (RES / "e10_operator_diagnostics" / "cross_system_per_run.json").read_text()
    )
    summary = json.loads((RES / "e10_operator_diagnostics" / "summary.json").read_text())

    old_results_dir = e10.RESULTS_DIR
    e10.RESULTS_DIR = str(FIG_DIR)
    try:
        e10.plot_figure(modrelu_records, cross_records, summary)
    finally:
        e10.RESULTS_DIR = old_results_dir

    for ext in ("png", "pdf"):
        src = FIG_DIR / f"e10_operator_diagnostics.{ext}"
        dst = FIG_DIR / f"fig10_operator_diagnostics.{ext}"
        if dst.exists():
            dst.unlink()
        src.replace(dst)


def fig11_cocycle_envelope():
    print("Generating Fig 11: Cocycle envelope...")
    from transient_geometry.experiments import e11_cocycle_envelope as e11

    records = json.loads(
        (RES / "e11_cocycle_envelope" / "real_esn_per_run.json").read_text()
    )
    summary = json.loads((RES / "e11_cocycle_envelope" / "summary.json").read_text())

    old_results_dir = e11.RESULTS_DIR
    e11.RESULTS_DIR = str(FIG_DIR)
    try:
        e11.plot_summary(records, summary["by_rho"])
    finally:
        e11.RESULTS_DIR = old_results_dir

    for ext in ("png", "pdf"):
        src = FIG_DIR / f"e11_cocycle_envelope.{ext}"
        dst = FIG_DIR / f"fig11_cocycle_envelope.{ext}"
        if dst.exists():
            dst.unlink()
        src.replace(dst)




# =============================================================================
if __name__ == "__main__":
    print(f"\nOutput directory: {FIG_DIR}\n{'-'*55}")
    fig0_precollapse_schematic()
    fig1_phase_diagram()
    fig2_grace_period()
    fig3_grace_method()
    fig3_variance_reduction()
    fig4_cross_system()
    fig5_noise_robustness()
    fig6_trained_system()
    fig7_k_sensitivity()
    fig8_separability_over_time()
    fig9_deq_relevance()
    fig10_operator_diagnostics()
    fig11_cocycle_envelope()
    fig12_ginibre_spectrum()
    print(f"{'-'*55}\nAll figures done.")
