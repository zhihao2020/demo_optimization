"""Grouped weekly CC bars for Rule / MILP / PSO / PC-HybridTD3."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path(__file__).resolve().parent
DATA = json.loads((FIG / "_holdout" / "runtime_kpis.json").read_text(encoding="utf-8"))

ORDER = ("Rule", "MILP", "PSO", "PC-HybridTD3")
COLORS = {
    "Rule": "#0072B2",
    "MILP": "#E69F00",
    "PSO": "#009E73",
    "PC-HybridTD3": "#D55E00",
}
WEEK_LABELS = ("12\n(win.)", "25\n(trans.)", "38\n(sum.)", "51\n(aut.)")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    weeks = list(DATA["weeks"])
    n_w = len(weeks)
    n_m = len(ORDER)
    x = np.arange(n_w, dtype=float)
    width = 0.18
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    for i, name in enumerate(ORDER):
        cc = np.asarray(DATA["methods"][name]["cc"], dtype=float) / 1e6
        offset = (i - (n_m - 1) / 2.0) * width
        ax.bar(
            x + offset,
            cc,
            width=width * 0.92,
            color=COLORS[name],
            edgecolor="white",
            linewidth=0.3,
            label=name,
            zorder=3,
        )
    ax.axhline(0.0, color="#D1D5DB", lw=0.6, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(WEEK_LABELS, fontsize=7)
    ax.set_ylabel(r"Weekly $CC$ ($10^6$ CNY/week)", fontsize=8)
    ax.set_ylim(-20.5, 0.5)
    ax.grid(True, axis="y", color="#E5E7EB", lw=0.5, zorder=0)
    ax.tick_params(labelsize=7)
    ax.legend(
        frameon=False,
        fontsize=6.5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=2,
        columnspacing=0.8,
        handlelength=1.2,
    )
    fig.tight_layout(pad=0.25)
    fig.subplots_adjust(top=0.82)
    fig.savefig(FIG / "fig_cc_weeks.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "fig_cc_weeks.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", FIG / "fig_cc_weeks.pdf")


if __name__ == "__main__":
    main()
