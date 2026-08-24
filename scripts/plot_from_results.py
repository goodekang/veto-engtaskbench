#!/usr/bin/env python
"""Redraw paper statistical figures from results/ tables."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
RES = ROOT / "results" / "tables"

MM = 1.0 / 25.4
FIG_W = 140 * MM

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "font.size": 8.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#3C4650",
        "axes.labelcolor": "#1A1A1A",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.color": "#3C4650",
        "ytick.color": "#3C4650",
        "xtick.labelcolor": "#1A1A1A",
        "ytick.labelcolor": "#1A1A1A",
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.minor.size": 1.4,
        "ytick.minor.size": 1.4,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.1,
        "legend.labelspacing": 0.35,
        "legend.borderaxespad": 0.25,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.6,
        "lines.markeredgewidth": 0.0,
        "grid.color": "#DDE3E9",
        "grid.linewidth": 0.5,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": True,
    }
)

M_REACT = "#E69F00"
M_PIPELINE = "#CC79A7"
M_MAS = "#009E73"
M_VETO = "#0072B2"
M_DOMAIN = "#6E7B8B"
N_DARK = "#33546E"
N_MID = "#7DA0BE"
N_RULE = "#8C97A2"
N_TEXT = "#4A5560"


def save(fig, name):
    fig.canvas.draw()
    fig.savefig(OUT / f"{name}.pdf", pad_inches=0)
    fig.savefig(OUT / f"{name}.png", pad_inches=0)
    plt.close(fig)
    print(f"saved {name}")


def panel_labels(fig, axes, letters=("(a)", "(b)"), pad=0.014):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse = fig.transFigure.inverted()
    boxes = [ax.get_tightbbox(renderer).transformed(inverse) for ax in axes]
    top = max(box.y1 for box in boxes) + pad
    for box, letter in zip(boxes, letters):
        fig.text(box.x0, top, letter, fontsize=9.5, fontweight="bold", ha="left", va="bottom")


def value_grid(ax, axis="y"):
    ax.grid(axis=axis, zorder=0)
    ax.set_axisbelow(True)


def halo(size=1.6):
    return [pe.withStroke(linewidth=size, foreground="white")]


def fig5():
    df = pd.read_csv(RES / "main_results.csv")
    cross = df[df["overall"].notna()].copy()
    methods = cross["method"].tolist()
    bim = cross["bim"].to_numpy()
    cad = cross["cad"].to_numpy()
    overall = cross["overall"].to_numpy()
    ci95 = cross["ci95"].to_numpy()
    ours = len(methods) - 1

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(FIG_W, 3.05), sharey=True, gridspec_kw={"wspace": 0.09}
    )
    fig.subplots_adjust(left=0.160, right=0.980, bottom=0.145, top=0.865)
    y = np.arange(len(methods))
    for ax in (ax1, ax2):
        ax.axhspan(ours - 0.5, ours + 0.5, color="#EEF3F8", zorder=0)
        ax.set_xlim(15, 100)
        ax.set_xticks(np.arange(20, 101, 20))
        ax.set_xticks(np.arange(20, 101, 10), minor=True)
        value_grid(ax, "x")
        ax.tick_params(axis="y", right=False)

    ax1.hlines(y, cad, bim, color="#B9C4CE", linewidth=1.4, zorder=2)
    ax1.plot(bim, y, "o", color=N_DARK, markersize=4.2, zorder=3, label="BIM-RCC")
    ax1.plot(
        cad,
        y,
        "s",
        markerfacecolor="white",
        markeredgecolor=N_DARK,
        markeredgewidth=0.9,
        markersize=3.8,
        zorder=3,
        label="CAD-PC",
        linestyle="none",
    )
    ax1.set_yticks(y, methods)
    ax1.invert_yaxis()
    ax1.set_xlabel("Task success rate (%)")
    ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)

    veto_lo, veto_hi = overall[ours] - ci95[ours], overall[ours] + ci95[ours]
    ax2.axvspan(veto_lo, veto_hi, color=M_VETO, alpha=0.09, zorder=0)
    ax2.axvline(overall[ours], color=M_VETO, linestyle=(0, (4, 2)), linewidth=0.7, zorder=1)
    colors = [M_VETO if i == ours else "#5E6B78" for i in range(len(methods))]
    for yi, (value, half, color) in enumerate(zip(overall, ci95, colors)):
        ax2.errorbar(
            value,
            yi,
            xerr=half,
            fmt="o",
            color=color,
            markersize=4.4 if yi == ours else 3.8,
            elinewidth=0.9,
            capsize=2.2,
            capthick=0.9,
            zorder=3,
        )
        ax2.text(
            value + half + 1.6,
            yi,
            f"{value:.1f}",
            va="center",
            ha="left",
            fontsize=7.2,
            color=color,
            path_effects=halo(),
        )
    ax2.set_xlabel("Overall task success rate (%)")
    panel_labels(fig, (ax1, ax2))
    save(fig, "fig05_main_results")


def fig6():
    df = pd.read_csv(RES / "backbones.csv")
    full = df["backbone"].tolist()
    veto = df["veto"].to_numpy()
    mas = df["mas"].to_numpy()
    react = df["react"].to_numpy()
    cost = df["cost"].to_numpy()
    open_weight = df["open_weight"].astype(bool).to_numpy()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(FIG_W, 3.15), gridspec_kw={"wspace": 0.30, "width_ratios": [1.12, 1.0]}
    )
    fig.subplots_adjust(left=0.098, right=0.988, bottom=0.185, top=0.855)
    x = np.arange(len(full))
    width = 0.27
    ax1.bar(x - width, veto, width, color=M_VETO, label="VETO (ours)", zorder=2)
    ax1.bar(x, mas, width, color=M_MAS, label="MAS w/o verifier", zorder=2)
    ax1.bar(x + width, react, width, color=M_REACT, label="ReAct", zorder=2)
    ax1.set_xticks(x, full, rotation=35, ha="right", rotation_mode="anchor")
    ax1.tick_params(axis="x", labelsize=7.2, pad=1.5)
    ax1.set_xlim(-0.62, len(full) - 0.38)
    ax1.set_ylim(0, 100)
    ax1.set_yticks(np.arange(0, 101, 20))
    ax1.set_yticks(np.arange(0, 101, 10), minor=True)
    ax1.set_ylabel("Overall task success rate (%)")
    ax1.tick_params(axis="x", top=False)
    value_grid(ax1, "y")
    ax1.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3)

    frontier = df[df["frontier"] == 1][["cost", "veto"]].to_numpy()
    frontier = frontier[np.argsort(frontier[:, 0])]
    ax2.plot(frontier[:, 0], frontier[:, 1], color=N_RULE, linestyle=(0, (4, 2)), linewidth=0.8, zorder=2)
    ax2.plot(
        cost[~open_weight],
        veto[~open_weight],
        "o",
        color=M_VETO,
        markersize=4.6,
        linestyle="none",
        zorder=3,
        label="Closed-weight API",
    )
    ax2.plot(
        cost[open_weight],
        veto[open_weight],
        "o",
        markerfacecolor="white",
        markeredgecolor=M_VETO,
        markeredgewidth=1.0,
        markersize=4.6,
        linestyle="none",
        zorder=3,
        label="Open-weight",
    )
    offsets = [
        (0.0, 1.2, "center", "bottom"),
        (0.006, 0.3, "left", "bottom"),
        (0.0, -1.3, "center", "top"),
        (0.006, -0.6, "left", "top"),
        (0.005, -0.5, "left", "top"),
        (0.006, 0.0, "left", "center"),
        (0.0, 1.1, "center", "bottom"),
    ]
    for x0, y0, name, (dx, dy, ha, va) in zip(cost, veto, full, offsets):
        ax2.text(x0 + dx, y0 + dy, name, fontsize=7.0, ha=ha, va=va, color="#1A1A1A", path_effects=halo())
    ax2.set_xlim(0.015, 0.275)
    ax2.set_ylim(60, 92)
    ax2.set_xticks(np.arange(0.05, 0.26, 0.05))
    ax2.set_yticks(np.arange(60, 91, 10))
    ax2.set_yticks(np.arange(60, 92, 5), minor=True)
    ax2.set_xlabel("Mean API cost per task (USD)")
    ax2.set_ylabel("Overall task success rate (%)")
    value_grid(ax2, "both")
    ax2.legend(loc="lower right", handletextpad=0.4, labelspacing=0.3, borderaxespad=0.5)
    panel_labels(fig, (ax1, ax2))
    save(fig, "fig06_backbone_robustness")


def fig8():
    ab = pd.read_csv(RES / "ablation.csv")
    rp = pd.read_csv(RES / "repair.csv")
    components = ab["component"].tolist()
    losses = ab["delta"].to_numpy()
    iterations = rp["iteration"].to_numpy()
    recovery = rp["cumulative"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, 2.85), gridspec_kw={"wspace": 0.34})
    fig.subplots_adjust(left=0.196, right=0.988, bottom=0.165, top=0.925)
    y = np.arange(len(components))
    colors = [N_DARK] + [N_MID] * (len(components) - 1)
    ax1.barh(y, losses, height=0.58, color=colors, zorder=2)
    for yi, loss in enumerate(losses):
        ax1.text(loss + 0.3, yi, f"{loss:.1f}", va="center", ha="left", fontsize=7.2, color="#1A1A1A")
    ax1.set_yticks(y, components)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 13)
    ax1.set_xticks(np.arange(0, 13, 4))
    ax1.set_xticks(np.arange(0, 13, 2), minor=True)
    ax1.set_xlabel("Loss in overall TSR (points)")
    ax1.tick_params(axis="y", right=False)
    value_grid(ax1, "x")

    ax2.fill_between(iterations, recovery, color=M_VETO, alpha=0.10, zorder=1)
    ax2.plot(iterations, recovery, color=M_VETO, marker="o", markersize=3.8, linewidth=1.5, zorder=3)
    k3 = float(rp.loc[rp["iteration"] == 3, "cumulative"].iloc[0])
    ax2.plot([3, 3], [0, k3], color=N_RULE, linestyle=(0, (2, 2)), linewidth=0.7, zorder=2)
    ax2.text(3.18, 74.5, "default $K$ = 3", fontsize=7.0, color=N_TEXT, ha="left", va="center")
    ax2.set_xlim(-0.12, 5.12)
    ax2.set_ylim(0, 82)
    ax2.set_xticks(iterations)
    ax2.set_yticks(np.arange(0, 81, 20))
    ax2.set_yticks(np.arange(0, 81, 10), minor=True)
    ax2.set_xlabel("Verifier-driven repair iteration")
    ax2.set_ylabel("Cumulative recovery (%)")
    value_grid(ax2, "y")
    panel_labels(fig, (ax1, ax2))
    save(fig, "fig08_ablation_recovery")


def fig9():
    bim_df = pd.read_csv(RES / "tiers_bim.csv")
    cad_df = pd.read_csv(RES / "tiers_cad.csv")
    tiers_bim = ["T1", "T2", "T3", "T4"]
    tiers_cad = ["C1", "C2", "C3", "C4"]
    bim = {row.method: [row.T1, row.T2, row.T3, row.T4] for row in bim_df.itertuples()}
    cad = {row.method: [row.C1, row.C2, row.C3, row.C4] for row in cad_df.itertuples()}
    styles = {
        "ReAct": (M_REACT, "s", "-", 1.2),
        "Fixed pipeline": (M_PIPELINE, "^", "-", 1.2),
        "MAS w/o verifier": (M_MAS, "D", "-", 1.2),
        "ACC agent (BIM only)": (M_DOMAIN, "v", (0, (4, 2)), 1.1),
        "Visual CAD agent (CAD only)": (M_DOMAIN, "X", (0, (4, 2)), 1.1),
        "VETO (ours)": (M_VETO, "o", "-", 1.7),
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, 3.35), sharey=True, gridspec_kw={"wspace": 0.07})
    fig.subplots_adjust(left=0.105, right=0.988, bottom=0.275, top=0.935)
    for ax, tiers, data, xlabel in (
        (ax1, tiers_bim, bim, "BIM-RCC complexity tier"),
        (ax2, tiers_cad, cad, "CAD-PC complexity tier"),
    ):
        pos = np.arange(len(tiers))
        veto = np.array(data["VETO (ours)"], dtype=float)
        best = np.max([v for k, v in data.items() if k != "VETO (ours)"], axis=0)
        ax.fill_between(pos, best, veto, color=M_VETO, alpha=0.10, zorder=1)
        for label, values in data.items():
            color, marker, ls, lw = styles[label]
            ax.plot(
                pos,
                values,
                color=color,
                marker=marker,
                linestyle=ls,
                linewidth=lw,
                markersize=3.8 if label != "VETO (ours)" else 4.4,
                zorder=4 if label == "VETO (ours)" else 3,
            )
        ax.set_xticks(pos, tiers)
        ax.set_xlim(-0.28, len(tiers) - 0.72)
        ax.set_ylim(20, 100)
        ax.set_yticks(np.arange(20, 101, 20))
        ax.set_yticks(np.arange(20, 101, 10), minor=True)
        ax.set_xlabel(xlabel)
        value_grid(ax, "y")
    ax1.set_ylabel("Task success rate (%)")
    order = [
        "ReAct",
        "Fixed pipeline",
        "MAS w/o verifier",
        "ACC agent (BIM only)",
        "Visual CAD agent (CAD only)",
        "VETO (ours)",
    ]
    handles = [
        Line2D([], [], color=styles[k][0], marker=styles[k][1], linestyle=styles[k][2], linewidth=styles[k][3], markersize=3.8)
        for k in order
    ]
    handles.append(Line2D([], [], color=M_VETO, alpha=0.20, linewidth=6, linestyle="-"))
    fig.legend(handles, order + ["VETO margin over best baseline"], loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3)
    panel_labels(fig, (ax1, ax2))
    save(fig, "fig09_complexity")


def fig10():
    checks = pd.read_csv(RES / "defect_shares.csv")
    sweep = pd.read_csv(RES / "broker_sweep.csv")
    names = checks["check"].tolist()
    shares = checks["share"].to_numpy()
    exposure = sweep["k"].to_numpy()
    grounded = sweep["with_broker"].to_numpy()
    no_broker = sweep["no_broker"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, 2.8), gridspec_kw={"wspace": 0.34})
    fig.subplots_adjust(left=0.226, right=0.988, bottom=0.170, top=0.925)
    y = np.arange(len(names))
    ax1.barh(y, shares, height=0.6, color=N_MID, zorder=2)
    for yi, value in enumerate(shares):
        ax1.text(value + 0.7, yi, str(int(value)), va="center", ha="left", fontsize=7.2, color="#1A1A1A")
    ax1.set_yticks(y, names)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 40)
    ax1.set_xticks(np.arange(0, 41, 10))
    ax1.set_xticks(np.arange(0, 41, 5), minor=True)
    ax1.set_xlabel("Share of caught defects (%)")
    ax1.tick_params(axis="y", right=False)
    value_grid(ax1, "x")

    ax2.fill_between(exposure, no_broker, grounded, color=M_VETO, alpha=0.10, zorder=1)
    ax2.plot(exposure, grounded, color=M_VETO, marker="o", markersize=3.8, linewidth=1.5, zorder=3, label="With broker grounding")
    ax2.plot(
        exposure,
        no_broker,
        color=N_RULE,
        marker="s",
        markersize=3.6,
        linewidth=1.2,
        linestyle=(0, (4, 2)),
        zorder=3,
        label="No broker",
    )
    ax2.plot([15, 15], [76, 85.9], color=N_RULE, linestyle=(0, (2, 2)), linewidth=0.7, zorder=2)
    ax2.text(15.8, 76.6, "default $k$ = 15", fontsize=7.0, color=N_TEXT, ha="left", va="bottom")
    ax2.set_xlim(3, 44)
    ax2.set_ylim(76, 88)
    ax2.set_xticks(exposure)
    ax2.set_yticks(np.arange(76, 89, 4))
    ax2.set_yticks(np.arange(76, 89, 2), minor=True)
    ax2.set_xlabel("Tools exposed per executor step, $k$")
    ax2.set_ylabel("Overall task success rate (%)")
    value_grid(ax2, "y")
    ax2.legend(loc="upper right", handletextpad=0.4)
    panel_labels(fig, (ax1, ax2))
    save(fig, "fig10_mechanism_profile")


def fig_seeds():
    df = pd.read_csv(ROOT / "results" / "curves" / "veto_seed_runs.csv")
    fig, ax = plt.subplots(figsize=(FIG_W, 2.4))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.90)
    for run, g in df.groupby("run"):
        ax.plot(g["epoch"], g["val_tsr"], label=str(run), linewidth=1.3)
    ax.set_xlabel("Replay epoch")
    ax.set_ylabel("Validation TSR")
    ax.set_ylim(0.55, 0.92)
    value_grid(ax, "y")
    ax.legend(ncol=3, loc="lower right")
    save(fig, "veto_seed_val_tsr")


def main():
    fig5()
    fig6()
    fig8()
    fig9()
    fig10()
    fig_seeds()


if __name__ == "__main__":
    main()
