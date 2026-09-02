#!/usr/bin/env python
"""Overlay journal labels on the plant topology illustration.

The live Fig. 1 is `fig_topology.png` (illustration with Wind/PV labels).
Do not run this overlay on that file; it was written for an unlabeled source.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, TextArea
from PIL import Image

FIG = Path(__file__).resolve().parent
SRC = FIG / "fig_topology.png"


def _tag(ax, xy, text, *, color="#1A3A5C"):
    ta = TextArea(
        text,
        textprops=dict(
            fontsize=8.0,
            color=color,
            fontfamily="serif",
            fontweight="bold",
            ha="left",
            va="center",
        ),
    )
    ab = AnnotationBbox(
        ta,
        xy,
        xycoords="data",
        box_alignment=(0.0, 0.5),
        pad=0.18,
        frameon=True,
        bboxprops=dict(facecolor="white", edgecolor=color, linewidth=0.8, boxstyle="round,pad=0.22"),
        zorder=6,
    )
    ax.add_artist(ab)


def main() -> None:
    im = Image.open(SRC)
    w, h = im.size
    fig, ax = plt.subplots(figsize=(7.2, 7.2 * h / w))
    ax.imshow(im)
    ax.axis("off")
    # Pixel coordinates on 1672 x 941.
    _tag(ax, (40, 70), "Supply: grid / wind / PV", color="#1A3A5C")
    _tag(ax, (70, 210), r"Grid interface  TOU buy/sell", color="#0072B2")
    _tag(ax, (980, 70), "Wind + PV", color="#009E73")
    _tag(ax, (40, 360), "Conversion", color="#D55E00")
    _tag(ax, (70, 455), r"Thermal  $u_{\mathrm{tp}}$", color="#D55E00")
    _tag(ax, (620, 330), r"CAES  $(m,z)$", color="#0072B2")
    _tag(ax, (40, 560), "Storage", color="#1A3A5C")
    _tag(ax, (70, 640), r"Battery  $u_{\mathrm{bat}}$, SoC", color="#0072B2")
    _tag(ax, (560, 700), "CAES inventories  gas / hot / cold", color="#D55E00")
    _tag(ax, (40, 880), "Demand + settlement: electric load", color="#333333")
    _tag(ax, (1320, 250), "Modelica FMU twin\n1 h state/command loop", color="#009E73")
    _tag(ax, (1180, 820), "Solid: power / mass flow\nDashed: observe / control", color="#555555")
    fig.subplots_adjust(0, 0, 1, 1)
    out_png = FIG / "fig_topology.png"
    out_pdf = FIG / "fig_topology.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print("wrote", out_png, im.size)


if __name__ == "__main__":
    main()
