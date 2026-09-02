#!/usr/bin/env python
"""Placeholder 168 h dispatch panels for Fig. 5 (internal draft; replace with Stage D)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent
T = np.arange(168)
H = T % 24


def _tou(h: np.ndarray) -> np.ndarray:
    p = np.full(h.shape, 0.38)
    p[(h >= 8) & (h < 11)] = 0.92
    p[(h >= 16) & (h < 21)] = 1.12
    p[(h >= 11) & (h < 16)] = 0.64
    p[(h < 7) | (h >= 23)] = 0.22
    return p


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    load = 210 + 45 * np.sin(2 * np.pi * (H - 7) / 24) + 18 * np.sin(4 * np.pi * T / 168)
    pv = np.clip(155 * np.sin(np.pi * np.clip(H - 6, 0, 12) / 12), 0, None)
    pv[(H < 6) | (H > 18)] = 0
    wind = 90 + 55 * np.sin(2 * np.pi * T / 84 + 0.7) + 20 * np.sin(2 * np.pi * H / 24)
    wind = np.clip(wind, 15, 220)
    tou = _tou(H)
    net = load - pv - wind
    thermal = np.clip(85 + 0.22 * net, 50, 150)
    # Battery: charge valley, discharge peak.
    bat = np.where(tou >= 0.9, 70.0, np.where(tou <= 0.25, -55.0, 8.0 * np.sin(2 * np.pi * H / 24)))
    caes = np.where(tou >= 1.0, -90.0, np.where(tou <= 0.25, 70.0, np.where(tou >= 0.85, -40.0, 0.0)))
    residual = load - pv - wind - thermal - bat - caes
    grid = np.clip(residual, -200, 200)
    soc_b = 0.45 + np.cumsum(-bat) / 500.0
    soc_b = 0.25 + 0.45 * (soc_b - soc_b.min()) / (soc_b.max() - soc_b.min() + 1e-9)
    soc_g = 0.55 + np.cumsum(-caes) / 1800.0
    soc_g = 0.35 + 0.40 * (soc_g - soc_g.min()) / (soc_g.max() - soc_g.min() + 1e-9)
    soc_h = 0.50 + 0.12 * np.sin(2 * np.pi * T / 168)
    soc_c = 0.50 - 0.12 * np.sin(2 * np.pi * T / 168)

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.15), sharex=True)
    ax = axes[0, 0]
    ax.plot(T, load, color="#111827", lw=1.15, label="Load")
    ax.plot(T, pv, color="#D55E00", lw=1.05, label="PV")
    ax.plot(T, wind, color="#0072B2", lw=1.05, label="Wind")
    ax.set_ylabel("Power (MW)", fontsize=8)
    ax.set_title("(a) Load and renewable availability", loc="left", fontsize=8.5, pad=3)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper right")

    ax = axes[0, 1]
    ax.plot(T, thermal, color="#B45309", lw=1.1, label="Thermal")
    ax.plot(T, bat, color="#7C3AED", lw=1.1, label="BESS")
    ax.plot(T, caes, color="#009E73", lw=1.1, label="CAES")
    ax.plot(T, grid, color="#6B7280", lw=1.0, label="Grid")
    ax.axhline(0, color="#D1D5DB", lw=0.6)
    ax.set_ylabel("Power (MW)", fontsize=8)
    ax.set_title("(b) Thermal, BESS, CAES and grid", loc="left", fontsize=8.5, pad=3)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")

    ax = axes[1, 0]
    ax.plot(T, tou, color="#C2410C", lw=1.15, label="TOU buy")
    ax.set_ylabel("Price (CNY/kWh)", fontsize=8, color="#C2410C")
    ax.tick_params(axis="y", labelcolor="#C2410C")
    ax2 = ax.twinx()
    ax2.plot(T, net, color="#374151", lw=1.05, label="Net load")
    ax2.set_ylabel("Net load (MW)", fontsize=8)
    ax.set_title("(c) TOU price and net load", loc="left", fontsize=8.5, pad=3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="upper right")

    ax = axes[1, 1]
    ax.plot(T, soc_b, color="#7C3AED", lw=1.15, label="Battery")
    ax.plot(T, soc_g, color="#009E73", lw=1.15, label="CAES gas")
    ax.plot(T, soc_h, color="#D55E00", lw=1.05, label="Hot tank")
    ax.plot(T, soc_c, color="#0072B2", lw=1.05, label="Cold tank")
    ax.set_ylim(0.15, 0.95)
    ax.set_ylabel("SOC (p.u.)", fontsize=8)
    ax.set_title("(d) Storage inventories", loc="left", fontsize=8.5, pad=3)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower right")

    for ax in axes.ravel():
        ax.set_xlim(0, 167)
        ax.tick_params(labelsize=7.5)
        ax.grid(True, color="#F3F4F6", lw=0.5)
        ax.spines["top"].set_visible(False)
    axes[1, 0].set_xlabel("Hour", fontsize=8)
    axes[1, 1].set_xlabel("Hour", fontsize=8)
    fig.tight_layout(h_pad=0.45, w_pad=0.55)
    fig.savefig(FIG / "fig_dispatch_week.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "fig_dispatch_week.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
