#!/usr/bin/env python
"""Holdout online decision time vs weekly CC (paper_min summaries)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path(__file__).resolve().parent
DATA = json.loads((FIG / "_holdout" / "runtime_kpis.json").read_text(encoding="utf-8"))

COLORS = {
    "Rule": "#0072B2",
    "MILP": "#E69F00",
    "PC-HybridTD3": "#D55E00",
}
MARKERS = {12: "o", 25: "s", 38: "^", 51: "D"}


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(7.16, 2.65))
    weeks = list(DATA["weeks"])
    for name, block in DATA["methods"].items():
        cc = np.asarray(block["cc"], dtype=float) / 1e6
        dt = np.asarray(block["dt_ms"], dtype=float)
        color = COLORS[name]
        for w, x, y in zip(weeks, dt, cc):
            ax.scatter(
                x,
                y,
                s=36,
                color=color,
                marker=MARKERS[w],
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )
        ax.scatter(
            float(dt.mean()),
            float(cc.mean()),
            s=110,
            color=color,
            marker="*",
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
            label=name,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Online decision time (ms/step)", fontsize=9)
    ax.set_ylabel(r"Weekly $CC$ ($10^6$ CNY/week)", fontsize=9)
    ax.set_xlim(0.45, 500)
    ax.set_ylim(-20.5, 0.5)
    ax.grid(True, which="both", color="#E5E7EB", lw=0.5)
    ax.tick_params(labelsize=8)
    handles_w = [
        plt.Line2D(
            [0],
            [0],
            marker=MARKERS[w],
            color="#6B7280",
            linestyle="None",
            markersize=6,
            label=f"Week {w}",
        )
        for w in weeks
    ]
    mean_h = plt.Line2D(
        [0],
        [0],
        marker="*",
        color="#6B7280",
        linestyle="None",
        markersize=10,
        label="Four-week mean",
    )
    leg_m = ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.add_artist(leg_m)
    ax.legend(
        handles=handles_w + [mean_h],
        frameon=False,
        fontsize=7.5,
        loc="lower left",
    )
    fig.tight_layout(pad=0.35)
    fig.savefig(FIG / "fig_runtime.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "fig_runtime.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", FIG / "fig_runtime.pdf")


if __name__ == "__main__":
    main()
