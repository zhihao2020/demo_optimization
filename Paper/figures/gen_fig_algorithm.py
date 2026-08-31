#!/usr/bin/env python
"""Vector PC-HybridTD3 loop. Band layout from the Image-model sketch; arrows match Alg. 1."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

FIG = Path(__file__).resolve().parent
C_TEAL = "#264653"
C_TEAL2 = "#2A9D8F"
C_GOLD = "#C8963E"
C_ORANGE = "#E76F51"
C_SLATE = "#6B7280"
C_BG = "#F4F6F5"
C_BOX = "#FFFFFF"
C_EDGE = "#D0D5DD"


def _band(ax, y, h, color, label):
    ax.add_patch(Rectangle((0.12, y), 11.76, h, facecolor=C_BG, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0.12, y), 0.16, h, facecolor=color, edgecolor="none", zorder=1))
    ax.text(0.38, y + h - 0.18, label, fontsize=7.5, fontweight="bold", color=color, va="top")


def _box(ax, x, y, w, h, title, body, *, tc=C_TEAL):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.07",
            facecolor=C_BOX,
            edgecolor=C_EDGE,
            linewidth=1.0,
            zorder=2,
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.26,
        title,
        ha="center",
        va="center",
        fontsize=8.4,
        fontweight="bold",
        color=tc,
        zorder=3,
    )
    ax.text(x + w / 2, y + 0.36, body, ha="center", va="center", fontsize=7.0, color="#333333", zorder=3)


def _arr(ax, p0, p1, *, color=C_TEAL2, ls="-", rad=0.0, lw=1.25, head=True):
    if head:
        ax.add_patch(
            FancyArrowPatch(
                p0,
                p1,
                arrowstyle="-|>",
                mutation_scale=10,
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
    ax.text(x, y, text, fontsize=6.8, color=color, ha=ha, va="center", zorder=5)


def main() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 6.55))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(6.0, 7.95, "PC-HybridTD3 closed loop", ha="center", fontsize=13, fontweight="bold", color=C_TEAL)

    _band(ax, 6.22, 1.42, C_TEAL2, "TWIN")
    _box(ax, 1.85, 6.35, 3.7, 1.08, "Sysplorer FMU, 1 h", "thermal--BESS--CAES DAE", tc=C_TEAL)
    _box(ax, 7.05, 6.35, 3.7, 1.08, "state $s$", "SoC, thermo, power, monthly TOU", tc=C_TEAL)

    _band(ax, 4.22, 1.72, C_TEAL2, "SUPPORT + ACTOR")
    _box(ax, 0.85, 4.35, 3.05, 1.28, "oracle", r"$A(s)=K(s)\times M_k(s)$", tc=C_TEAL2)
    _box(ax, 4.35, 4.35, 3.25, 1.28, "PC-HybridTD3 actor", "mode + magnitude;\ndecode on $A_f(s)$", tc=C_TEAL2)
    _box(ax, 8.05, 4.35, 3.05, 1.28, "inference", r"eval: $\arg\max$ mode", tc=C_TEAL2)

    _band(ax, 2.52, 1.42, C_GOLD, "SCREEN")
    _box(
        ax,
        2.55,
        2.64,
        6.9,
        1.08,
        "GiveSafe (adopted, fallback off)",
        r"$N_{\mathrm{try}}=64$; only accepted hours step the FMU",
        tc=C_GOLD,
    )

    _band(ax, 0.18, 2.02, C_ORANGE, "REPLAY + UPDATE")
    _box(ax, 0.85, 0.38, 3.05, 1.48, r"$D_B$ physical Bellman", r"accepted $(s,a,r,s')$", tc=C_ORANGE)
    _box(ax, 4.35, 0.38, 3.25, 1.48, r"$D_S$ safety audit only", "never a Bellman self-loop", tc=C_ORANGE)
    _box(ax, 8.05, 0.38, 3.05, 1.48, "TD3 target + delayed actor", r"$\arg\max m'$; noise on $z'$", tc=C_ORANGE)

    # Twin: FMU -> state
    _arr(ax, (5.55, 6.89), (7.05, 6.89))
    _label(ax, 6.30, 7.08, r"next $s$", C_TEAL2)

    # state s -> oracle (down then left in the gap under TWIN)
    _arr(ax, (8.90, 6.35), (8.90, 6.10), head=False)
    _arr(ax, (8.90, 6.10), (2.38, 6.10), head=False)
    _arr(ax, (2.38, 6.10), (2.38, 5.63))
    _label(ax, 5.60, 6.24, r"build $A(s)$", C_TEAL2)

    # oracle -> actor -> inference
    _arr(ax, (3.90, 4.99), (4.35, 4.99))
    _arr(ax, (7.60, 4.99), (8.05, 4.99))

    # inference -> GiveSafe
    _arr(ax, (9.58, 4.35), (7.80, 3.72), color=C_GOLD)
    _label(ax, 9.20, 3.92, r"decoded $(m,z)$", C_GOLD, ha="left")

    # ACCEPT: GiveSafe left -> left gutter -> FMU
    _arr(ax, (2.55, 3.18), (0.55, 3.18), color=C_TEAL2, head=False)
    _arr(ax, (0.55, 3.18), (0.55, 6.89), color=C_TEAL2, head=False)
    _arr(ax, (0.55, 6.89), (1.85, 6.89), color=C_TEAL2)
    _label(ax, 0.68, 5.05, "ACCEPT\nstep 1 h", C_TEAL2, ha="left")

    # REJECT: GiveSafe bottom -> D_F
    _arr(ax, (5.98, 2.64), (5.98, 1.86), color=C_ORANGE, ls="--")
    _label(ax, 6.55, 2.22, "REJECT", C_ORANGE, ha="left")

    # physical transition: FMU down left gutter to D_B
    _arr(ax, (1.85, 6.55), (0.42, 6.55), color="#E09A4A", head=False)
    _arr(ax, (0.42, 6.55), (0.42, 1.12), color="#E09A4A", head=False)
    _arr(ax, (0.42, 1.12), (0.85, 1.12), color="#E09A4A")
    _label(ax, 0.05, 2.55, "physical\ntransition", "#C45C26", ha="left")

    # D_B and D_F independently into V(s) (not a chain through D_F)
    _arr(ax, (2.38, 0.38), (2.38, 0.22), color=C_SLATE, head=False)
    _arr(ax, (2.38, 0.22), (9.58, 0.22), color=C_SLATE, head=False)
    _arr(ax, (9.58, 0.22), (9.58, 0.38), color=C_SLATE)
    _arr(ax, (7.60, 1.12), (8.05, 1.12), color=C_SLATE)
    _label(ax, 5.60, 0.08, "both buffers", C_SLATE)

    # theta: V(s) right gutter, over SUPPORT boxes, into actor top
    _arr(ax, (11.10, 1.12), (11.55, 1.12), color=C_SLATE, head=False)
    _arr(ax, (11.55, 1.12), (11.55, 5.78), color=C_SLATE, head=False)
    _arr(ax, (11.55, 5.78), (5.98, 5.78), color=C_SLATE, head=False)
    _arr(ax, (5.98, 5.78), (5.98, 5.63), color=C_SLATE)
    _label(ax, 11.68, 3.35, r"$\theta$", C_SLATE, ha="left")

    out = FIG / "fig_algorithm.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "fig_algorithm.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
