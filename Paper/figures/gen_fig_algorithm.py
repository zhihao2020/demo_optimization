#!/usr/bin/env python
"""Journal-style PC-HybridTD3 closed-loop schematic (no in-figure title)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

FIG = Path(__file__).resolve().parent
C_NAVY = "#1A3A5C"
C_TEAL = "#0072B2"
C_ORANGE = "#D55E00"
C_GREEN = "#009E73"
C_SLATE = "#5B6570"
C_BG = "#F6F7F8"
C_BOX = "#FFFFFF"
C_EDGE = "#B8BFC7"


def _band(ax, y, h):
    ax.add_patch(Rectangle((0.18, y), 11.64, h, facecolor=C_BG, edgecolor="none", zorder=0))


def _box(ax, x, y, w, h, title, body, *, tc=C_NAVY):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.05",
            facecolor=C_BOX,
            edgecolor=C_EDGE,
            linewidth=1.05,
            zorder=2,
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.28,
        title,
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=tc,
        zorder=3,
    )
    ax.text(x + w / 2, y + 0.38, body, ha="center", va="center", fontsize=7.0, color="#333333", zorder=3)


def _arr(ax, p0, p1, *, color=C_TEAL, ls="-", rad=0.0, lw=1.2, head=True):
    if head:
        ax.add_patch(
            FancyArrowPatch(
                p0,
                p1,
                arrowstyle="-|>",
                mutation_scale=9,
                lw=lw,
                color=color,
                linestyle=ls,
                connectionstyle=f"arc3,rad={rad}",
                zorder=4,
            )
        )
        return
    ax.add_line(Line2D([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, ls=ls, zorder=4, solid_capstyle="butt"))


def _label(ax, x, y, text, color, *, ha="center"):
    ax.text(x, y, text, fontsize=6.6, color=color, ha=ha, va="center", zorder=5)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.55))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.35)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    _band(ax, 5.55, 1.45)
    ax.text(0.32, 6.82, "Plant twin", fontsize=7.2, fontweight="bold", color=C_NAVY, va="top")
    _box(ax, 1.70, 5.68, 3.85, 1.12, "Sysplorer FMU", r"thermal-battery-CAES DAE, $\tau_s=1$ h", tc=C_NAVY)
    _box(ax, 7.00, 5.68, 3.85, 1.12, r"state $s_t$", "inventories, powers, 24 h TOU/forecast", tc=C_NAVY)

    _band(ax, 3.55, 1.72)
    ax.text(0.32, 5.08, "Actor on " + r"$\mathcal{A}_f(s)$", fontsize=7.2, fontweight="bold", color=C_TEAL, va="top")
    _box(ax, 0.70, 3.68, 3.20, 1.32, "Feasibility oracle", r"$\mathcal{K}(s)$, $\mathcal{M}_k(s)$, grid slice", tc=C_TEAL)
    _box(ax, 4.35, 3.68, 3.30, 1.32, "PC-HybridTD3 actor", r"$(u^{\mathrm{th}},u^{\mathrm{bat}},m,z)$ decoded on $\mathcal{A}_f(s)$", tc=C_TEAL)
    _box(ax, 8.10, 3.68, 3.10, 1.32, "Greedy inference", r"$m=\arg\max$ mode logits", tc=C_TEAL)

    _band(ax, 1.95, 1.35)
    ax.text(0.32, 3.12, "Screen", fontsize=7.2, fontweight="bold", color=C_ORANGE, va="top")
    _box(
        ax,
        2.35,
        2.08,
        7.30,
        1.05,
        "GiveSafe (adopted residual check)",
        r"train: up to $N_{\mathrm{try}}=64$; evaluation: one attempt",
        tc=C_ORANGE,
    )

    _band(ax, 0.12, 1.58)
    ax.text(0.32, 1.52, "Replay", fontsize=7.2, fontweight="bold", color=C_GREEN, va="top")
    _box(ax, 0.70, 0.22, 3.20, 1.22, r"$\mathcal{D}_{\mathrm{B}}$ economic", r"accepted $(s,a,r,s')$ only", tc=C_GREEN)
    _box(ax, 4.35, 0.22, 3.30, 1.22, r"$\mathcal{D}_{\mathrm{S}}$ audit", "rejected hours; no Bellman update", tc=C_GREEN)
    _box(ax, 8.10, 0.22, 3.10, 1.22, "TD3 target", r"$\arg\max m'$; noise on $z'$ only", tc=C_GREEN)

    _arr(ax, (5.55, 6.24), (7.00, 6.24))
    _label(ax, 6.28, 6.42, r"next $s$", C_TEAL)

    _arr(ax, (8.92, 5.68), (8.92, 5.42), head=False)
    _arr(ax, (8.92, 5.42), (2.30, 5.42), head=False)
    _arr(ax, (2.30, 5.42), (2.30, 5.00))
    _label(ax, 5.60, 5.55, r"build $\mathcal{A}_f(s)$", C_TEAL)

    _arr(ax, (3.90, 4.34), (4.35, 4.34))
    _arr(ax, (7.65, 4.34), (8.10, 4.34))
    _arr(ax, (9.65, 3.68), (7.70, 3.13), color=C_ORANGE)
    _label(ax, 9.15, 3.28, r"decoded $(m,z)$", C_ORANGE, ha="left")

    _arr(ax, (2.35, 2.60), (0.48, 2.60), color=C_TEAL, head=False)
    _arr(ax, (0.48, 2.60), (0.48, 6.24), color=C_TEAL, head=False)
    _arr(ax, (0.48, 6.24), (1.70, 6.24), color=C_TEAL)
    _label(ax, 0.02, 4.55, "accept", C_TEAL, ha="left")

    _arr(ax, (6.00, 2.08), (6.00, 1.44), color=C_ORANGE, ls="--")
    _label(ax, 6.55, 1.78, "reject", C_ORANGE, ha="left")

    _arr(ax, (1.70, 5.90), (0.32, 5.90), color=C_GREEN, head=False)
    _arr(ax, (0.32, 5.90), (0.32, 0.83), color=C_GREEN, head=False)
    _arr(ax, (0.32, 0.83), (0.70, 0.83), color=C_GREEN)
    _label(ax, 0.05, 2.20, "physical\ntransition", C_GREEN, ha="left")

    _arr(ax, (3.90, 0.83), (4.35, 0.83), color=C_SLATE)
    _arr(ax, (7.65, 0.83), (8.10, 0.83), color=C_SLATE)
    _arr(ax, (11.20, 0.83), (11.55, 0.83), color=C_SLATE, head=False)
    _arr(ax, (11.55, 0.83), (11.55, 5.22), color=C_SLATE, head=False)
    _arr(ax, (11.55, 5.22), (6.00, 5.22), color=C_SLATE, head=False)
    _arr(ax, (6.00, 5.22), (6.00, 5.00), color=C_SLATE)
    _label(ax, 11.68, 3.00, r"$\theta$", C_SLATE, ha="left")

    out = FIG / "fig_algorithm.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "fig_algorithm.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
