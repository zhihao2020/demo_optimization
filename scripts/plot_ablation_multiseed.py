#!/usr/bin/env python
"""Plot matched-budget multi-seed ablation bars for the paper."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    d = json.loads((ROOT / "runs/paper_ablation_multiseed_15k.json").read_text(encoding="utf-8"))
    order = [
        ("full", "Full HMSD"),
        ("noprior", "w/o MSGP"),
        ("noher", "w/o MS-HER"),
        ("nofmle", "w/o F-MLE"),
    ]
    labels, means, stds, socs = [], [], [], []
    for k, lab in order:
        v = d["variants"][k]
        labels.append(lab)
        means.append(v["reward_mean_of_seed_means"])
        stds.append(v["reward_std_of_seed_means"])
        soc = n = 0
        for row in v["seeds"].values():
            for se in row["by_season"]:
                n += 1
                if se.get("soc"):
                    soc += 1
        socs.append(100.0 * soc / n)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(labels))
    colors = ["#2ca02c", "#1f77b4", "#d62728", "#ff7f0e"]
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="k", lw=0.4, width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Three-season mean episode reward\n(mean ± std over seeds)")
    ax.set_title(r"Matched-budget ablations ($1.5\times10^{4}$ steps, seeds 0–2)")
    ax.grid(True, axis="y", alpha=0.3)
    for i, (m, s, sc) in enumerate(zip(means, stds, socs)):
        ax.text(i, m + s + 2.5, f"SoC {sc:.0f}%", ha="center", fontsize=8)
    ax.set_ylim(0, max(means) + max(stds) + 18)
    fig.tight_layout()
    out = ROOT / "Paper/figures/fig_ablation_bars.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
    for lab, m, s, sc in zip(labels, means, stds, socs):
        print(f"{lab}: {m:.2f}±{s:.2f}, SoC {sc:.0f}%")


if __name__ == "__main__":
    main()
