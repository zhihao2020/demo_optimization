#!/usr/bin/env python
"""Generate paper figures from current abs / TD3-scratch runs only (no Hybrid/ares)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "Paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# clean style
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def plot_main_bars() -> Path:
    """Full method family: B0, linprog, PSO, TD3-scratch, Safe Market-GHTD3."""
    base = json.loads((ROOT / "runs/paper_baselines_or_pso.json").read_text(encoding="utf-8"))
    seasons = ["winter", "transition", "summer"]
    labels = ["Winter", "Transition", "Summer"]
    methods = [
        ("B0", "#9aa0a6", False),
        ("linprog", "#f6bd16", False),
        ("PSO", "#e8684a", False),
        ("TD3_scratch", "#5b8ff9", True),
        ("GHTD3", "#5ad8a6", True),
    ]
    legend_names = {
        "B0": "B0 (rule)",
        "linprog": "linprog MPC",
        "PSO": "PSO",
        "TD3_scratch": "TD3-scratch",
        "GHTD3": "Safe Market-GHTD3",
    }
    x = np.arange(len(seasons))
    n = len(methods)
    w = 0.15
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    for i, (key, color, has_err) in enumerate(methods):
        means, errs = [], []
        for s in seasons:
            block = base["seasons"][s][key]
            if has_err:
                means.append(float(block["mean"]))
                errs.append(float(block["std"]))
            else:
                means.append(float(block["reward"]))
                errs.append(0.0)
        offset = (i - (n - 1) / 2) * w
        if has_err:
            ax.bar(x + offset, means, w, yerr=errs, capsize=2, label=legend_names[key], color=color)
        else:
            ax.bar(x + offset, means, w, label=legend_names[key], color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Episode reward")
    ax.set_title("Closed-loop weekly reward: rules, OR, PSO, single-layer TD3, hierarchical GHTD3")
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    out = FIG / "fig_main_reward_bars.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _load_step_log(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "steps" in data:
        data = data["steps"]
    df = pd.DataFrame(data)
    # common keys
    step_col = "valid_step" if "valid_step" in df.columns else "step"
    if "r_ext" in df.columns:
        rew = df["r_ext"]
    elif "reward" in df.columns:
        rew = df["reward"]
    else:
        rew = df.iloc[:, 1]
    return pd.DataFrame({"step": df[step_col].astype(float), "reward": rew.astype(float)})


def plot_training() -> Path | None:
    """Multi-seed mean ± std band (plus faint individual seeds)."""
    series = {"TD3-scratch": [], "Safe Market-GHTD3": []}
    for seed in (0, 1, 2):
        gp = ROOT / f"runs/ghtd3_abs_s{seed}_35k/train/step_log.json"
        tp = ROOT / f"runs/td3_scratch_s{seed}_35k/train/step_log.json"
        if not (gp.is_file() and tp.is_file()):
            continue
        g = _load_step_log(gp).sort_values("step")
        t = _load_step_log(tp).sort_values("step")
        # clip extreme outliers for display only (TD3 s1 spikes)
        t = t.copy()
        t["reward"] = t["reward"].clip(-5, 5)
        g = g.copy()
        g["reward"] = g["reward"].clip(-5, 5)
        series["TD3-scratch"].append(t)
        series["Safe Market-GHTD3"].append(g)
    if not series["TD3-scratch"]:
        return None

    def aligned_matrix(dfs: list[pd.DataFrame], k: int = 15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # interpolate each seed onto common step grid
        steps = None
        mats = []
        for df in dfs:
            y = df["reward"].rolling(k, min_periods=1).mean().to_numpy()
            x = df["step"].to_numpy()
            if steps is None:
                steps = x
            # resample to first seed's steps if lengths match protocol
            if len(x) == len(steps) and np.allclose(x, steps):
                mats.append(y)
            else:
                mats.append(np.interp(steps, x, y))
        M = np.stack(mats, axis=0)
        return steps, M.mean(axis=0), M.std(axis=0)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    colors = {"TD3-scratch": "#5b8ff9", "Safe Market-GHTD3": "#5ad8a6"}
    for name, dfs in series.items():
        # faint individuals
        for df in dfs:
            y = df["reward"].rolling(15, min_periods=1).mean()
            ax.plot(df["step"], y, color=colors[name], alpha=0.18, lw=0.9)
        steps, mu, sd = aligned_matrix(dfs)
        ax.plot(steps, mu, color=colors[name], lw=1.8, label=name)
        ax.fill_between(steps, mu - sd, mu + sd, color=colors[name], alpha=0.22)
    ax.set_xlabel("Valid steps")
    ax.set_ylabel("Step reward (rolling mean)")
    ax.set_title("Training curves: 3-seed mean $\\pm$ std (faint lines = individual seeds)")
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Write both formats: pdflatex prefers .pdf over .png for \\includegraphics{fig_training}
    out_png = FIG / "fig_training.png"
    out_pdf = FIG / "fig_training.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_pdf


def plot_algorithm_schematic() -> Path:
    """Simple block diagram without hybrid teacher residual."""
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#e8f5e9"):
        from matplotlib.patches import FancyBboxPatch

        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15", facecolor=fc, edgecolor="#333", lw=1.2)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    box(0.3, 2.4, 2.2, 1.1, "High-level TD3\n$\\mu^{hi}(s)$ + MSGP", "#e3f2fd")
    box(3.0, 2.4, 2.0, 1.1, "Goal $g\\in\\mathbb{R}^5$\nMS-HER buffer", "#fff3e0")
    box(5.5, 2.4, 2.2, 1.1, "Low-level TD3\n$\\pi_{lo}(s_n,\\kappa g)$", "#e8f5e9")
    box(8.0, 2.4, 1.7, 1.1, "GiveSafe\n$\\Pi_{\\mathcal{F}}$", "#fce4ec")
    box(5.5, 0.5, 2.2, 1.1, "FMU twin\nthermal/bat/CAES", "#f3e5f5")
    box(0.3, 0.5, 2.2, 1.1, "F-MLE warm-start\n(rule feasible)", "#eceff1")
    box(3.0, 0.5, 2.0, 1.1, "Market reward\n$r^{ext}$, SoC gate", "#e0f7fa")

    # arrows
    ax.annotate("", xy=(3.0, 2.95), xytext=(2.5, 2.95), arrowprops=dict(arrowstyle="->", color="#333"))
    ax.annotate("", xy=(5.5, 2.95), xytext=(5.0, 2.95), arrowprops=dict(arrowstyle="->", color="#333"))
    ax.annotate("", xy=(8.0, 2.95), xytext=(7.7, 2.95), arrowprops=dict(arrowstyle="->", color="#333"))
    ax.annotate("", xy=(6.6, 1.6), xytext=(6.6, 2.4), arrowprops=dict(arrowstyle="->", color="#333"))
    ax.annotate("", xy=(8.8, 1.6), xytext=(8.8, 2.4), arrowprops=dict(arrowstyle="->", color="#333"))
    ax.text(5.0, 3.7, "Safe Market-GHTD3 (absolute goal-conditioned, no hybrid teacher residual)", ha="center", fontsize=10)
    out = FIG / "fig_algorithm.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_soc_from_csv() -> Path | None:
    """B0 rule vs GHTD3 eval SoC if columns exist."""
    seasons = []
    # use winter eval if available
    gh = ROOT / "runs/ghtd3_abs_s1_35k/trajectories/eval.csv"
    rule = ROOT / "runs/ghtd3_abs_s1_35k/trajectories/rule.csv"
    td3 = ROOT / "runs/td3_scratch_s1_35k/trajectories/eval.csv"
    if not gh.is_file():
        return None

    def load_soc(path: Path):
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        bat = None
        gas = None
        for k, c in cols.items():
            if "battery" in k and "soc" in k:
                bat = df[c].astype(float).values
            if ("gas" in k or "caes_gas" in k) and "soc" in k:
                gas = df[c].astype(float).values
        t = np.arange(len(df))
        return t, bat, gas

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.5), sharex=True)
    for path, name, color in [
        (rule, "B0", "#9aa0a6"),
        (td3 if td3.is_file() else None, "TD3-scratch", "#5b8ff9"),
        (gh, "Safe Market-GHTD3", "#5ad8a6"),
    ]:
        if path is None or not path.is_file():
            continue
        t, bat, gas = load_soc(path)
        if bat is not None:
            axes[0].plot(t, bat, label=name, color=color, lw=1.1)
        if gas is not None:
            axes[1].plot(t, gas, label=name, color=color, lw=1.1)
    axes[0].set_ylabel("Battery SoC")
    axes[1].set_ylabel("CAES gas SoC")
    axes[1].set_xlabel("Hour in evaluation week")
    axes[0].legend(frameon=False, ncol=3)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("SoC trajectories (seed 1 weekly eval, FMU closed loop)", y=1.01)
    out = FIG / "fig_soc_compare.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_delta_bars() -> Path:
    multi = json.loads((ROOT / "runs/ghtd3_abs/multi_seed_summary_paired.json").read_text(encoding="utf-8"))
    seasons = ["winter", "transition", "summer"]
    labels = ["Winter", "Transition", "Summer"]
    d = [multi["multi_seed"][s]["delta_mean"] for s in seasons]
    ds = [multi["multi_seed"][s]["delta_std"] for s in seasons]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    x = np.arange(len(seasons))
    colors = ["#5ad8a6" if v >= 0 else "#e57373" for v in d]
    ax.bar(x, d, yerr=ds, capsize=3, color=colors, edgecolor="#333", lw=0.5)
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("$\\Delta$ reward (GHTD3 $-$ TD3-scratch)")
    ax.set_title("Paired multi-seed improvement over TD3-scratch")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    out = FIG / "fig_delta_vs_td3.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    outs = []
    outs.append(plot_main_bars())
    outs.append(plot_delta_bars())
    outs.append(plot_algorithm_schematic())
    tr = plot_training()
    if tr:
        outs.append(tr)
    soc = plot_soc_from_csv()
    if soc:
        outs.append(soc)
    print("wrote:")
    for p in outs:
        if p:
            print(" ", p)


if __name__ == "__main__":
    main()
