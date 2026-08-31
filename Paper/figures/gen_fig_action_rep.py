#!/usr/bin/env python
"""Three-panel CAES action-representation schematic for Fig. action-rep."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle

FIG = Path(__file__).resolve().parent
DIS_LO, DIS_HI = -1.0, -0.33
CHG_LO, CHG_HI = 0.86, 1.0
C_DIS = "#0072B2"
C_IDLE = "#6B7280"
C_CHG = "#D55E00"
C_GAP = "#9CA3AF"
C_INK = "#1F2937"
C_EX = "#B91C1C"


def _number_line(ax) -> None:
    ax.plot([-1.12, 1.12], [0, 0], color="#111827", lw=1.15, zorder=2, solid_capstyle="butt")
    h = 0.28
    ax.add_patch(Rectangle((DIS_LO, -h / 2), DIS_HI - DIS_LO, h, facecolor=C_DIS, alpha=0.90, lw=0, zorder=3))
    ax.add_patch(Rectangle((-0.05, -h / 2), 0.10, h, facecolor=C_IDLE, alpha=0.95, lw=0, zorder=4))
    ax.add_patch(Rectangle((CHG_LO, -h / 2), CHG_HI - CHG_LO, h, facecolor=C_CHG, alpha=0.90, lw=0, zorder=3))
    ax.add_patch(
        Rectangle(
            (DIS_HI + 0.02, -h / 2),
            CHG_LO - DIS_HI - 0.07,
            h,
            facecolor="#F9FAFB",
            edgecolor=C_GAP,
            hatch="////",
            linewidth=0.5,
            zorder=3,
        )
    )
    ax.set_xlim(-1.22, 1.28)
    ax.set_ylim(-0.62, 0.78)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([-1, -0.33, 0, 1])
    ax.set_xticklabels(["-1", "-0.33", "0", "1"], fontsize=7.5)
    ax.set_xlabel(r"$u_{\mathrm{caes}}$", fontsize=9)
    ax.tick_params(length=3, color="#9CA3AF")


def _arrow(ax, x0, y0, x1, y1, color: str) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=9,
            lw=1.2,
            color=color,
            zorder=5,
        )
    )


def main() -> Path:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))

    ax = axes[0]
    _number_line(ax)
    ax.set_title("(a) Continuous-projection TD3", loc="left", fontsize=8.5, color=C_INK, pad=4)
    _arrow(ax, 0.28, 0.48, 0.0, 0.18, C_EX)
    ax.plot(0.28, 0.48, "o", ms=4.5, color=C_EX, zorder=6)
    ax.plot(0.0, 0.16, "o", ms=5.5, color=C_EX, zorder=6)
    ax.text(0.28, 0.58, r"$z$", fontsize=9, color=C_EX, ha="center")

    ax = axes[1]
    _number_line(ax)
    ax.set_title("(b) Component-support Hybrid TD3", loc="left", fontsize=8.5, color=C_INK, pad=4)
    _arrow(ax, -0.665, 0.52, -0.665, 0.18, C_DIS)
    _arrow(ax, 0.93, 0.52, 0.93, 0.18, C_CHG)

    ax = axes[2]
    _number_line(ax)
    ax.set_title(r"(c) PC-HybridTD3", loc="left", fontsize=8.5, color=C_INK, pad=4)
    lo, hi = -0.72, -0.40
    ax.add_patch(Rectangle((DIS_LO, -0.14), DIS_HI - DIS_LO, 0.28, facecolor="#FFFFFF", alpha=0.62, lw=0, zorder=4))
    ax.add_patch(Rectangle((lo, -0.16), hi - lo, 0.32, facecolor=C_DIS, alpha=1.0, lw=0, zorder=5))
    ax.text((lo + hi) / 2, 0.42, r"$\mathcal{M}_k(s)$", ha="center", fontsize=9, color=C_DIS)

    handles = [
        Patch(facecolor=C_DIS, edgecolor="none", label="discharge"),
        Patch(facecolor=C_IDLE, edgecolor="none", label="idle"),
        Patch(facecolor=C_CHG, edgecolor="none", label="charge"),
        Patch(facecolor="#F9FAFB", edgecolor=C_GAP, hatch="////", label="illegal gap"),
        Line2D([0], [0], marker="o", color=C_EX, lw=0, markersize=5, label=r"example $z$"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.04),
        handlelength=1.4,
        columnspacing=1.2,
    )
    fig.tight_layout(w_pad=0.85, rect=(0, 0.08, 1, 1))
    png = FIG / "fig_action_rep.png"
    pdf = FIG / "fig_action_rep.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print("wrote", png)
    return png


if __name__ == "__main__":
    main()
