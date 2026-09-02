#!/usr/bin/env python
"""Seed-0 PC-HybridTD3 training log (0903, 400k physical steps)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path(__file__).resolve().parent
LOG = FIG / "_holdout" / "pc_s0_step_log.json"


def _smooth(y: np.ndarray, w: int = 21) -> np.ndarray:
    if y.size < 3:
        return y
    k = min(w, y.size if y.size % 2 else y.size - 1)
    if k < 3:
        return y
    ker = np.ones(k) / k
    return np.convolve(y, ker, mode="same")


def main() -> None:
    rows = json.loads(LOG.read_text(encoding="utf-8"))
    steps = np.array([float(r["valid_step"]) for r in rows])
    rew = np.array([float(r["reward"]) for r in rows])
    c_steps, c_loss = [], []
    for r in rows:
        if r.get("critic_loss") is None:
            continue
        c_steps.append(float(r["valid_step"]))
        c_loss.append(float(r["critic_loss"]))
    c_steps = np.array(c_steps)
    c_loss = np.clip(np.array(c_loss), 1e-6, None)
    c_log = np.log10(c_loss)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.45))

    ax = axes[0]
    ax.plot(steps / 1e5, rew, color="#56B4E9", lw=0.55, alpha=0.55, zorder=1)
    ax.plot(
        steps / 1e5,
        _smooth(rew),
        color="#0072B2",
        lw=1.45,
        label="Moving average",
        zorder=2,
    )
    ax.set_xlabel(r"Environment hours ($\times 10^{5}$)", fontsize=9)
    ax.set_ylabel("Step reward", fontsize=9)
    ax.set_title("(a) Training reward", loc="left", fontsize=8.5, pad=3)
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(-5.4, 4.8)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.grid(True, color="#E5E7EB", lw=0.5)
    ax.tick_params(labelsize=8)

    ax = axes[1]
    ax.plot(c_steps / 1e5, c_loss, color="#E69F00", lw=0.55, alpha=0.55, zorder=1)
    ax.plot(
        c_steps / 1e5,
        np.power(10.0, _smooth(c_log)),
        color="#D55E00",
        lw=1.45,
        label="Moving average",
        zorder=2,
    )
    ax.set_xlabel(r"Environment hours ($\times 10^{5}$)", fontsize=9)
    ax.set_ylabel("Critic loss", fontsize=9)
    ax.set_title("(b) Critic loss", loc="left", fontsize=8.5, pad=3)
    ax.set_xlim(0.0, 4.0)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.grid(True, which="both", color="#E5E7EB", lw=0.5)
    ax.tick_params(labelsize=8)

    fig.tight_layout(pad=0.35)
    fig.savefig(FIG / "fig_training.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "fig_training.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", FIG / "fig_training.pdf")


if __name__ == "__main__":
    main()
