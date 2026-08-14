#!/usr/bin/env python3
"""Seed-0 seasonal comparison bars for the Applied Energy draft."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
    }
)

# Order matches Table tab:main narrative
SEASONS = ["Winter", "Transition", "Summer"]
# Reward
REW = {
    "B0": [54.3, 8.4, 22.1],
    "linprog": [21.9, -16.2, 8.0],
    "PSO": [7.9, 49.1, 35.7],
    "SAC": [102.9, 18.6, 79.3],
    "TD3": [-1.3, -0.3, 97.9],
    "HMSD": [113.6, 50.1, 61.2],
}
# Jgen 1e6 CNY
JGEN = {
    "B0": [6.27, -0.89, 1.21],
    "linprog": [4.56, -1.17, -0.28],
    "PSO": [1.52, 8.08, 6.37],
    "SAC": [14.82, 3.78, 11.19],
    "TD3": [0.29, 0.13, 13.86],
    "HMSD": [15.99, 8.43, 7.94],
}
TH = {"TD3": [398, 1350, 9025], "HMSD": [9700, 9421, 15624]}
BAT = {"TD3": [188, 900, 644], "HMSD": [7499, 9253, 4328]}

COLORS = {
    "B0": "#B0BEC5",
    "linprog": "#8C8C8C",
    "PSO": "#E9C46A",
    "SAC": "#2A9D8F",
    "TD3": "#0072B2",
    "HMSD": "#E76F51",
}


def grouped_bars(data: dict, ylabel: str, fname: str, hline: float | None = None) -> None:
    methods = list(data)
    x = np.arange(len(SEASONS))
    n = len(methods)
    width = 0.78 / n
    fig, ax = plt.subplots(figsize=(6.9, 2.85))
    for i, m in enumerate(methods):
        off = (i - n / 2 + 0.5) * width
        ax.bar(
            x + off,
            data[m],
            width * 0.92,
            label=m,
            color=COLORS[m],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
    if hline is not None:
        ax.axhline(hline, color="#444", lw=0.6, ls="--", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(SEASONS)
    ax.set_ylabel(ylabel)
    ax.legend(ncol=3, loc="upper right")
    fig.savefig(OUT / f"{fname}.pdf")
    fig.savefig(OUT / f"{fname}.png")
    plt.close(fig)


def mechanism() -> None:
    x = np.arange(len(SEASONS))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.7))
    pairs = [
        (axes[0], TH, "Thermal energy (MWh)"),
        (axes[1], BAT, "Battery throughput (MWh)"),
    ]
    for ax, data, ylab in pairs:
        ax.bar(x - w / 2, data["TD3"], w, label="TD3", color=COLORS["TD3"], edgecolor="white")
        ax.bar(x + w / 2, data["HMSD"], w, label="HMSD", color=COLORS["HMSD"], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(SEASONS)
        ax.set_ylabel(ylab)
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_mechanism.pdf")
    fig.savefig(OUT / "fig_mechanism.png")
    plt.close(fig)


if __name__ == "__main__":
    grouped_bars(REW, "Episode reward", "fig_main_reward_bars")
    grouped_bars(JGEN, r"$J^{\mathrm{gen}}$ ($10^{6}$ CNY)", "fig_jgen_bars")
    mechanism()
    print("wrote", OUT / "fig_main_reward_bars.pdf")
    print("wrote", OUT / "fig_jgen_bars.pdf")
    print("wrote", OUT / "fig_mechanism.pdf")
