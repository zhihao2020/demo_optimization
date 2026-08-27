#!/usr/bin/env python3
"""Fig.3: held-out weekly wind / irradiance / load / monthly TOU (weeks 5, 18, 31)."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
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

COL = {
    "wind": "#009E73",
    "irr": "#E69F00",
    "load": "#0072B2",
    "tou": "#000000",
}

# Fair protocol held-out weeks: winter 5 (Feb), transition 18 (May), summer 31 (Aug).
SEASONS = (("Winter", 5 * 168), ("Transition", 18 * 168), ("Summer", 31 * 168))


def _col(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    keys = [k for k in rows[0] if k.lower() not in ("time", "hour", "t", "index")]
    key = keys[0]
    return np.asarray([float(r[key]) for r in rows], dtype=np.float64)


def main() -> None:
    wind = _col(ROOT / "data" / "winds.csv")
    irr = _col(ROOT / "data" / "Gstc.csv")
    load_mw = _col(ROOT / "data" / "load.csv") / 1.0e6
    buy = _col(ROOT / "data" / "price_tou.csv")  # buy_yuan_per_kwh is first non-time column

    t = np.arange(168)
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 5.5), sharex="col", sharey="row")
    axp_last = None
    for j, (name, s0) in enumerate(SEASONS):
        sl = slice(s0, s0 + 168)
        axes[0, j].set_title(name)
        axes[0, j].plot(t, wind[sl], color=COL["wind"], lw=0.95)
        axes[1, j].plot(t, irr[sl], color=COL["irr"], lw=0.95)
        axes[2, j].plot(t, load_mw[sl], color=COL["load"], lw=0.95)
        axp = axes[2, j].twinx()
        axp.plot(t, buy[sl], color=COL["tou"], lw=0.85, alpha=0.85)
        axp.spines["top"].set_visible(False)
        axp.set_ylim(0.0, 1.35)
        if j < 2:
            axp.set_yticklabels([])
        else:
            axp_last = axp
        axes[2, j].set_xlabel("Hour in held-out week")
        axes[2, j].set_xlim(0, 167)
    axes[0, 0].set_ylabel(r"Wind (m/s)")
    axes[1, 0].set_ylabel(r"Irradiance (W/m$^2$)")
    axes[2, 0].set_ylabel("Load (MW)")
    if axp_last is not None:
        axp_last.set_ylabel("TOU buy (CNY/kWh)")

    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    fig.savefig(OUT / "fig_seasonal_boundary.pdf")
    fig.savefig(OUT / "fig_seasonal_boundary.png", dpi=300)
    print("wrote", OUT / "fig_seasonal_boundary.pdf")


if __name__ == "__main__":
    main()
