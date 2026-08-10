"""Publication figure: HMSD data flow + network layer dimensions."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "Paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 11,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

C = {
    "hi": "#E8F1FB",
    "hi_b": "#2F6FED",
    "lo": "#E8F7EF",
    "lo_b": "#1B8A5A",
    "gs": "#FFF3E0",
    "gs_b": "#E67E22",
    "env": "#F3E8FF",
    "env_b": "#7C3AED",
    "buf": "#F5F5F5",
    "buf_b": "#6B7280",
    "ink": "#1F2937",
    "muted": "#6B7280",
    "arrow": "#374151",
    "learn": "#DC2626",
}


def rbox(ax, x, y, w, h, text, fc, ec, fs=8, lw=1.2, bold=False):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=2,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=C["ink"],
        fontweight="bold" if bold else "normal",
        zorder=3,
        multialignment="center",
    )
    return p


def arrow(ax, x1, y1, x2, y2, color=None, lw=1.3, rad=0.0, label=None, ls="-"):
    color = color or C["arrow"]
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=10,
        color=color,
        lw=lw,
        linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
        zorder=4,
        shrinkA=1,
        shrinkB=1,
    )
    ax.add_patch(arr)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(
            mx,
            my + 0.12,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            color=color,
            fontweight="medium",
            zorder=5,
        )


def draw_mlp(ax, x0, y0, layers, title, accent, note=None):
    n = len(layers)
    max_h = 2.6
    gap = 0.55
    ax.text(
        x0 + (n * 0.72 + (n - 1) * gap) / 2,
        y0 + max_h + 0.35,
        title,
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=accent,
    )
    xs = []
    for i, (lab, nh) in enumerate(layers):
        h = max(0.55, min(2.4, 0.45 + np.log1p(nh) * 0.28))
        w = 0.72
        x = x0 + i * (w + gap)
        y = y0 + (max_h - h) / 2
        rbox(ax, x, y, w, h, lab, "#FFFFFF", accent, fs=6.8, lw=1.3)
        xs.append((x + w / 2, y + h / 2, x + w, y + h / 2))
        if i > 0:
            px = xs[i - 1][2]
            py = xs[i - 1][1]
            arrow(ax, px, py, x, y + h / 2, accent, lw=1.0)
    if note:
        ax.text(
            x0 + (n * 0.72 + (n - 1) * gap) / 2,
            y0 - 0.25,
            note,
            ha="center",
            va="top",
            fontsize=6.8,
            color=C["muted"],
        )


def main():
    fig = plt.figure(figsize=(11.2, 8.6))

    # ---------- Panel A ----------
    ax1 = fig.add_axes([0.04, 0.52, 0.92, 0.45])
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 6.2)
    ax1.axis("off")
    ax1.set_title(
        "(a) HMSD data flow — follow solid arrows hour by hour",
        loc="left",
        fontsize=11,
        fontweight="bold",
        pad=6,
    )

    ax1.add_patch(
        FancyBboxPatch(
            (0.15, 0.35),
            7.55,
            5.55,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor="#F8FAFC",
            edgecolor="#94A3B8",
            lw=1.0,
            ls="--",
            zorder=0,
        )
    )
    ax1.text(0.35, 5.65, "AGENT", fontsize=9, fontweight="bold", color="#334155")

    ax1.add_patch(
        FancyBboxPatch(
            (7.95, 0.35),
            3.85,
            5.55,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor="#FDF4FF",
            edgecolor="#C084FC",
            lw=1.0,
            ls="--",
            zorder=0,
        )
    )
    ax1.text(8.15, 5.65, "ENVIRONMENT", fontsize=9, fontweight="bold", color="#6B21A8")

    rbox(
        ax1,
        0.4,
        3.9,
        2.4,
        1.35,
        "High-level TD3\n$\\mu^{hi}(s)\\!\\to\\!g\\in\\mathbb{R}^5$\n+ MSGP prior",
        C["hi"],
        C["hi_b"],
        fs=8,
        bold=True,
    )
    rbox(
        ax1,
        3.1,
        3.9,
        2.5,
        1.35,
        "Low-level TD3\n$\\pi_{lo}(s_n,\\kappa g)\\!\\to\\!\\tilde a$\n$\\kappa\\!=\\!4$",
        C["lo"],
        C["lo_b"],
        fs=8,
        bold=True,
    )
    rbox(
        ax1,
        5.85,
        4.05,
        1.6,
        1.05,
        "GiveSafe\n$a=\\Pi_{\\mathcal{F}(s)}(\\tilde a)$",
        C["gs"],
        C["gs_b"],
        fs=7.5,
        bold=True,
    )

    rbox(
        ax1,
        0.4,
        1.7,
        2.4,
        1.35,
        "High buffer $\\mathcal{D}^{hi}$\n$(s,g,R^{ext},s_{t+c})$\nMS-HER relabel $g$",
        C["buf"],
        C["buf_b"],
        fs=7.5,
    )
    rbox(
        ax1,
        3.1,
        1.7,
        2.5,
        1.35,
        "Low buffer $\\mathcal{D}^{lo}$\n$(s,g,a,r^{int},s',g')$\nevery hour",
        C["buf"],
        C["buf_b"],
        fs=7.5,
    )
    rbox(
        ax1,
        5.75,
        1.7,
        1.75,
        1.35,
        "F-MLE\nwarm-start\n(before RL)",
        "#FEF3C7",
        "#D97706",
        fs=7.5,
    )

    rbox(
        ax1,
        8.25,
        3.9,
        3.3,
        1.35,
        "Sysplorer FMU twin\nthermal + battery + CAES\n1 h step",
        C["env"],
        C["env_b"],
        fs=8,
        bold=True,
    )
    rbox(
        ax1,
        8.25,
        1.7,
        3.3,
        1.35,
        "Boundaries\nTOU price, wind/PV/load\n$\\to s',\\, r^{ext},\\, r^{int}$",
        C["env"],
        C["env_b"],
        fs=8,
    )

    arrow(ax1, 2.8, 4.55, 3.1, 4.55, C["hi_b"], label="g every c=8")
    arrow(ax1, 5.6, 4.55, 5.85, 4.55, C["lo_b"], label=r"$\tilde a$")
    arrow(ax1, 7.45, 4.55, 8.25, 4.55, C["gs_b"], label="a*")
    arrow(ax1, 9.9, 3.9, 9.9, 3.05, C["env_b"])
    arrow(ax1, 8.25, 2.35, 7.5, 2.35, C["env_b"], label="r, s'")
    arrow(ax1, 7.5, 2.35, 5.6, 2.35, C["muted"], lw=1.0)
    arrow(ax1, 5.6, 2.35, 4.35, 3.05, C["muted"], lw=1.0, rad=-0.05)
    arrow(ax1, 7.5, 2.5, 1.6, 3.05, C["muted"], lw=1.0, rad=0.15)

    arrow(ax1, 1.6, 3.05, 1.6, 3.9, C["learn"], ls="--", lw=1.1, label="TD3 update")
    arrow(ax1, 4.35, 3.05, 4.35, 3.9, C["learn"], ls="--", lw=1.1)
    arrow(ax1, 6.6, 3.05, 4.35, 3.9, "#D97706", ls=":", lw=1.0, rad=0.2)

    ax1.text(
        3.1,
        3.55,
        "within c-window: residual  $g\\leftarrow h(s,g,s')$",
        fontsize=7,
        color=C["lo_b"],
        style="italic",
    )
    ax1.text(
        0.35,
        0.55,
        "Solid = interact every hour   |   Dashed red = train from buffer   |   Dotted = F-MLE only once at start",
        fontsize=7.2,
        color=C["muted"],
    )

    # ---------- Panel B ----------
    ax2 = fig.add_axes([0.04, 0.03, 0.92, 0.46])
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 6.0)
    ax2.axis("off")
    ax2.set_title(
        "(b) Neural nets — both levels share the same TD3 skeleton (2 x 256 MLP)",
        loc="left",
        fontsize=11,
        fontweight="bold",
        pad=4,
    )

    draw_mlp(
        ax2,
        0.25,
        3.15,
        [("s\n163", 163), ("256\nReLU", 256), ("256\nReLU", 256), ("g\n5", 5)],
        r"High Actor  $\mu^{hi}$",
        C["hi_b"],
        "s -> goal every c steps",
    )
    draw_mlp(
        ax2,
        0.25,
        0.35,
        [("[s,g]\n168", 168), ("256\nReLU", 256), ("256\nReLU", 256), ("Q\n1", 1)],
        r"High Critic  $Q^{hi}$  (x2 twin)",
        C["hi_b"],
        "each twin is this stack",
    )
    draw_mlp(
        ax2,
        6.2,
        3.15,
        [("[s,kg]\n168", 168), ("256\nReLU", 256), ("256\nReLU", 256), ("heads\nhybrid a", 6)],
        r"Low Actor  $\pi_{lo}$",
        C["lo_b"],
        "heads: u_tp, u_bat, mode(3), mag",
    )
    draw_mlp(
        ax2,
        6.2,
        0.35,
        [("[s,g,a]\n174", 174), ("256\nReLU", 256), ("256\nReLU", 256), ("Q\n1", 1)],
        r"Low Critic  $Q^{lo}$  (x2 twin)",
        C["lo_b"],
        "a = [u_tp, u_bat, mode_oh(3), mag]  (6-D)",
    )

    ax2.add_patch(
        FancyBboxPatch(
            (4.55, 1.8),
            1.35,
            2.0,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="#FFFBEB",
            edgecolor="#F59E0B",
            lw=1.0,
            zorder=2,
        )
    )
    ax2.text(5.22, 3.5, "Same depth", ha="center", fontsize=7.5, fontweight="bold", color="#B45309")
    ax2.text(
        5.22,
        3.05,
        "2 hidden\nlayers\nwidth 256\n+\noutput head",
        ha="center",
        fontsize=7,
        color=C["ink"],
    )
    ax2.text(5.22, 2.05, "+ target nets\n(soft tau)", ha="center", fontsize=6.8, color=C["muted"])

    pdf = OUT / "fig_network_structure.pdf"
    png = OUT / "fig_network_structure.png"
    fig.savefig(pdf)
    fig.savefig(png)
    print(f"saved {png}")
    print(f"saved {pdf}")
    plt.close()


if __name__ == "__main__":
    main()
