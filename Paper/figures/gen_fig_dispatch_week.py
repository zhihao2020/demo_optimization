#!/usr/bin/env python
"""Week-12 (winter) PC-HybridTD3 seed-0 dispatch from holdout eval.csv."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path(__file__).resolve().parent
CSV = FIG / "_holdout" / "pc_s0_w12_eval.csv"


def _mw(row: dict, key: str, *, flip: bool = False) -> float:
    v = float(row[key]) / 1e6
    return -v if flip else v


def main() -> None:
    with CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    t = np.arange(len(rows), dtype=float)
    load = np.array([_mw(r, "obs_p_load_actual") for r in rows])
    wind = np.array([_mw(r, "obs_p_wind_available", flip=True) for r in rows])
    pv = np.array([_mw(r, "obs_p_pv_available", flip=True) for r in rows])
    thermal = np.array([_mw(r, "obs_p_thermal", flip=True) for r in rows])
    # Charge-positive storage; import-positive grid (FMU generation/export is negative).
    bat = np.array([_mw(r, "obs_p_battery") for r in rows])
    caes = np.array([_mw(r, "obs_p_caes") for r in rows])
    grid = np.array([_mw(r, "obs_p_grid", flip=True) for r in rows])
    tou = np.array([float(r["rt_market_buy_yuan_per_kwh"]) for r in rows])
    net = load - wind - pv
    soc_b = np.array([float(r["obs_battery_soc"]) for r in rows])
    soc_g = np.array([float(r["obs_caes_gas_soc"]) for r in rows])
    soc_h = np.array([float(r["obs_caes_hot_soc"]) for r in rows])
    soc_c = np.array([float(r["obs_caes_cold_soc"]) for r in rows])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.25), sharex=True)
    day = np.arange(24, 168, 24)

    ax = axes[0, 0]
    ax.plot(t, load, color="#000000", lw=1.15, label="Load")
    ax.plot(t, pv, color="#E69F00", lw=1.05, label="PV")
    ax.plot(t, wind, color="#0072B2", lw=1.05, label="Wind")
    ax.set_ylabel("Power (MW)", fontsize=8)
    ax.set_title("(a) Load and renewable availability", loc="left", fontsize=8.5, pad=3)
    ax.legend(frameon=False, fontsize=7, ncol=1, loc="center left")

    ax = axes[0, 1]
    ax.plot(t, thermal, color="#D55E00", lw=1.15, label="Thermal")
    ax.plot(t, bat, color="#CC79A7", lw=1.05, label="BESS")
    ax.plot(t, caes, color="#009E73", lw=1.05, label="CAES")
    ax.plot(t, grid, color="#0072B2", lw=0.95, alpha=0.9, label="Grid")
    ax.axhline(0.0, color="#D1D5DB", lw=0.6)
    ax.set_ylabel("Power (MW)", fontsize=8)
    ax.set_title("(b) Thermal, BESS, CAES and grid", loc="left", fontsize=8.5, pad=3)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower left")

    ax = axes[1, 0]
    ax.step(t, tou, color="#D55E00", lw=1.15, where="post")
    ax.set_ylabel("Price (CNY/kWh)", fontsize=8, color="#D55E00")
    ax.tick_params(axis="y", labelcolor="#D55E00")
    ax2 = ax.twinx()
    ax2.plot(t, net, color="#000000", lw=1.05)
    ax2.set_ylabel("Net load (MW)", fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax.set_title("(c) TOU price and net load", loc="left", fontsize=8.5, pad=3)

    ax = axes[1, 1]
    ax.plot(t, soc_b, color="#CC79A7", lw=1.15, label="Battery")
    ax.plot(t, soc_g, color="#009E73", lw=1.15, label="CAES gas")
    ax.plot(t, soc_h, color="#E69F00", lw=1.05, label="Hot tank")
    ax.plot(t, soc_c, color="#0072B2", lw=1.05, label="Cold tank")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("SOC (p.u.)", fontsize=8)
    ax.set_title("(d) Storage inventories", loc="left", fontsize=8.5, pad=3)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="center right")

    for ax in axes.ravel():
        ax.set_xlim(0, 167)
        ax.tick_params(labelsize=7.5)
        ax.grid(True, color="#F3F4F6", lw=0.5)
        for x in day:
            ax.axvline(x, color="#E5E7EB", lw=0.45, zorder=0)
    axes[1, 0].set_xlabel("Hour", fontsize=8)
    axes[1, 1].set_xlabel("Hour", fontsize=8)
    fig.tight_layout(h_pad=0.45, w_pad=0.55)
    fig.savefig(FIG / "fig_dispatch_week.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(
        FIG / "fig_dispatch_week.png", dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)
    print("wrote", FIG / "fig_dispatch_week.pdf")


if __name__ == "__main__":
    main()
