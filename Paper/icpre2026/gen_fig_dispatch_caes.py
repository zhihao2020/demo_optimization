#!/usr/bin/env python
"""IEEE Fig.3: FS-HSAC vs linprog/rule CAES dispatch, one held-out week."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEASON = "transition"
OUT = Path(__file__).resolve().parent / "figures"
OKABE = {
    "th": "#0072B2",
    "grid": "#E69F00",
    "caes": "#009E73",
    "zero": "#999999",
}


def load_powers(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    out = pd.DataFrame()
    out["h"] = np.arange(len(df))
    out["p_th"] = df["obs_p_thermal"].to_numpy(float) * 1e-6
    out["p_grid"] = df["obs_p_grid"].to_numpy(float) * 1e-6
    out["p_caes"] = df["obs_p_caes"].to_numpy(float) * 1e-6
    return out


def pick_contrast() -> tuple[str, Path | None]:
    lp = ROOT / f"runs/seasonal_tou2026/{SEASON}/linprog_s0/trajectories/eval.csv"
    rule = ROOT / f"runs/seasonal_tou2026/{SEASON}/fs_hsac_s0/trajectories/rule.csv"
    if lp.exists():
        return "Rolling linprog", lp
    return "missing", None


def panel(ax, df: pd.DataFrame, title: str):
    h = df["h"].to_numpy()
    ax.plot(h, df["p_th"], color=OKABE["th"], lw=1.2, label="Thermal")
    ax.plot(h, df["p_grid"], color=OKABE["grid"], lw=1.1, label="Grid")
    ax.plot(h, df["p_caes"], color=OKABE["caes"], lw=1.6, label="CAES")
    ax.axhline(0.0, color=OKABE["zero"], lw=0.6, ls="--")
    ax.set_xlim(0, max(len(df) - 1, 1))
    ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
    ax.set_ylabel("Power (MW)")
    ax.grid(True, alpha=0.18)


def main():
    fs = ROOT / f"runs/seasonal_tou2026/{SEASON}/fs_hsac_s0/trajectories/eval.csv"
    if not fs.exists():
        raise SystemExit(f"missing {fs}")
    label, cpath = pick_contrast()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    n = 2 if cpath is not None else 1
    fig, axes = plt.subplots(n, 1, figsize=(7.16, 2.15 * n), sharex=True)
    if n == 1:
        axes = [axes]
    panel(axes[0], load_powers(fs), "FS-HSAC (greedy)")
    if cpath is not None:
        panel(axes[1], load_powers(cpath), label)
        axes[1].set_xlabel("Hour in held-out week")
    else:
        axes[0].set_xlabel("Hour in held-out week")
    axes[0].legend(ncol=3, loc="upper right")
    fig.tight_layout(h_pad=0.35)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_dispatch_caes.pdf")
    fig.savefig(OUT / "fig_dispatch_caes.png")
    print("wrote", OUT / "fig_dispatch_caes.pdf", "contrast", label)


if __name__ == "__main__":
    main()
