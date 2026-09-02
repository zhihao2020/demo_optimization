#!/usr/bin/env python
"""Placeholder validation-cost curves for Fig. 4 (internal draft; replace with Stage D)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent
HOURS = np.linspace(0.0, 4.0e5, 81)
# Terminal means match tab:kpi CC (1e6 CNY).
ENDS = {"proj": 16.81, "hybrid": 15.94, "pc": 14.28}
MILP = 15.41
C = {"proj": "#D55E00", "hybrid": "#0072B2", "pc": "#009E73"}


def _curve(end: float, start: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    tau = 1.15e5
    mean = end + (start - end) * np.exp(-HOURS / tau)
    wobble = 0.18 * np.exp(-HOURS / 1.6e5) * np.sin(HOURS / 2.2e4)
    mean = mean + wobble
    std = 0.55 * np.exp(-HOURS / 1.8e5) + 0.12
    noise = 0.04 * rng.normal(size=HOURS.size)
    return mean + noise, std


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    rng = np.random.default_rng(4)
    fig, ax = plt.subplots(figsize=(7.16, 2.35))
    series = [
        ("Continuous-projection TD3", "proj", 21.6),
        ("Component-support hybrid TD3", "hybrid", 20.8),
        ("PC-HybridTD3", "pc", 20.2),
    ]
    for label, key, start in series:
        mean, std = _curve(ENDS[key], start, rng)
        ax.plot(HOURS / 1e5, mean, color=C[key], lw=1.35, label=label)
        ax.fill_between(HOURS / 1e5, mean - std, mean + std, color=C[key], alpha=0.16, lw=0)
    ax.axhline(MILP, color="#6B7280", ls="--", lw=1.0, label="Rolling MILP (eval.)")
    ax.set_xlim(0, 4.0)
    ax.set_ylim(13.0, 22.4)
    ax.set_xlabel("Environment hours ($\\times 10^{5}$)", fontsize=9)
    ax.set_ylabel(r"Validation $CC$ ($10^{6}$ CNY)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=1)
    ax.grid(True, color="#E5E7EB", lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.25)
    fig.savefig(FIG / "fig_training.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "fig_training.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
