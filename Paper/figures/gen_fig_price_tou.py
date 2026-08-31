#!/usr/bin/env python3
"""Fig.2: monthly Shandong 110 kV two-part TOU — winter eval day vs Jan/Aug."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CSV = ROOT / "data" / "price_tou.csv"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
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
# Okabe–Ito
COL = {
    "S": "#D55E00",
    "P": "#E69F00",
    "F": "#0072B2",
    "V": "#009E73",
    "D": "#56B4E9",
    "jan": "#0072B2",
    "aug": "#D55E00",
}


def load_buy() -> tuple[np.ndarray, list[str]]:
    buy, band = [], []
    with CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            buy.append(float(row["buy_yuan_per_kwh"]))
            band.append(str(row.get("band", "F")))
    return np.asarray(buy, dtype=np.float64), band


def shade_bands(ax, hours, bands):
    h0 = 0
    for h in range(1, 25):
        if h == 24 or bands[h] != bands[h0]:
            ax.axvspan(h0, h, color=COL.get(bands[h0], "#ccc"), alpha=0.12, lw=0)
            h0 = h


def main() -> None:
    buy, bands = load_buy()
    # winter held-out week 5 starts at hour 840 (5*168)
    w0 = 5 * 168
    hours = np.arange(24)
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.55), sharey=False)

    ax = axes[0]
    y = buy[w0 : w0 + 24]
    b = bands[w0 : w0 + 24]
    shade_bands(ax, hours, b)
    ax.step(hours, y, where="post", color="#264653", lw=1.7)
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xlabel("Hour")
    ax.set_ylabel(r"Buy price (CNY/kWh)")
    ax.set_title("(a) Winter held-out week, day 1", loc="left", fontsize=9)
    ax.set_ylim(0, 1.35)
    handles = [
        Patch(facecolor=COL["S"], alpha=0.35, edgecolor="none", label="Sharp"),
        Patch(facecolor=COL["P"], alpha=0.35, edgecolor="none", label="Peak"),
        Patch(facecolor=COL["F"], alpha=0.35, edgecolor="none", label="Flat"),
        Patch(facecolor=COL["V"], alpha=0.35, edgecolor="none", label="Valley"),
        Patch(facecolor=COL["D"], alpha=0.35, edgecolor="none", label="Deep"),
    ]
    ax.legend(handles=handles, loc="upper left", ncol=2, fontsize=7, frameon=False)

    ax = axes[1]
    jan = buy[0:24]
    aug = buy[212 * 24 : 212 * 24 + 24]  # 1 Aug = day 212 of 2026
    ax.step(hours, jan, where="post", color=COL["jan"], lw=1.7, label="January (O)")
    ax.step(hours, aug, where="post", color=COL["aug"], lw=1.7, ls="--", label="August (O)")
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xlabel("Hour")
    ax.set_title("(b) Same clock, different months", loc="left", fontsize=9)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.35)

    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / "fig_price_tou.pdf")
    fig.savefig(OUT / "fig_price_tou.png", dpi=300)
    print("wrote", OUT / "fig_price_tou.pdf")


if __name__ == "__main__":
    main()
