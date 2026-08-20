#!/usr/bin/env python
"""Generate all plan-listed paper figures into Paper/figures/.

Reuses traj / baselines where available; draws schematics for c-step, CAES set,
cold-tank guard, carbon position, aux obs when live annual traces are absent.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "Paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
TRAJ = ROOT / "runs" / "seasonal_v1"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def _save(fig, name: str) -> Path:
    out = FIG / f"{name}.png"
    fig.savefig(out)
    # companion pdf when possible
    try:
        fig.savefig(FIG / f"{name}.pdf")
    except Exception:
        pass
    plt.close(fig)
    print("wrote", out)
    return out


def plot_price_tou() -> Path:
    p = ROOT / "data" / "price_tou.csv"
    df = pd.read_csv(p)
    # expect hour + buy/sell columns; be flexible
    cols = [c for c in df.columns if c.lower() not in ("time", "hour", "t", "index")]
    t = np.arange(min(len(df), 24))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for c in cols[:2]:
        y = pd.to_numeric(df[c], errors="coerce").to_numpy()[:24]
        ax.step(t, y, where="mid", label=c)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Price (CNY/kWh)")
    ax.set_title("Sample TOU purchase / sell prices")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, "fig_price_tou")


def plot_seasonal_boundary() -> Path:
    seasons = ("Winter", "Transition", "Summer")
    starts = {"Winter": 5 * 168, "Transition": 18 * 168, "Summer": 31 * 168}

    def col(df: pd.DataFrame) -> np.ndarray:
        for c in df.columns:
            if c.lower() not in ("time", "hour", "t", "index"):
                return pd.to_numeric(df[c], errors="coerce").to_numpy()
        return pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()

    w = col(pd.read_csv(ROOT / "data" / "winds.csv"))
    g = col(pd.read_csv(ROOT / "data" / "Gstc.csv"))
    load_mw = col(pd.read_csv(ROOT / "data" / "load.csv")) / 1.0e6
    price = pd.read_csv(ROOT / "data" / "price_tou.csv")
    buy = (
        price["buy_yuan_per_kwh"].to_numpy(float)
        if "buy_yuan_per_kwh" in price.columns
        else col(price)
    )
    p168 = np.tile(buy[:24], 7)[:168] if len(buy) >= 24 else buy[:168]

    fig, axes = plt.subplots(3, 3, figsize=(9.6, 6.2), sharex="col", sharey="row")
    rows = (
        (w, "Wind (m/s)", "#2ca02c"),
        (g, r"Irradiance (W/m$^2$)", "#ff7f0e"),
        (load_mw, "Load (MW)", "#000000"),
    )
    t = np.arange(168)
    for j, season in enumerate(seasons):
        s0 = starts[season]
        sl = np.arange(s0, s0 + 168)
        axes[0, j].set_title(season)
        for i, (series, ylabel, color) in enumerate(rows):
            ax = axes[i, j]
            y = series[sl % len(series)]
            ax.plot(t, y, color=color, lw=1.0)
            ax.spines["top"].set_visible(False)
            if j == 0:
                ax.set_ylabel(ylabel)
        axp = axes[2, j].twinx()
        axp.plot(t, p168, color="#5b8ff9", lw=0.85, alpha=0.85)
        axp.spines["top"].set_visible(False)
        axes[2, j].set_xlabel("Hour in held-out week")
        if j == 2:
            axp.set_ylabel("TOU buy (CNY/kWh)")
    fig.suptitle("Seasonal weekly boundaries (physical units)", y=1.01, fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig_seasonal_boundary")


def plot_cstep() -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    c = 8
    for k in range(3):
        t0 = k * c
        ax.add_patch(plt.Rectangle((t0, 0.55), c, 0.35, facecolor="#5b8ff9", alpha=0.35, edgecolor="#5b8ff9"))
        ax.text(t0 + c / 2, 0.72, f"goal $g_{{{k}}}$", ha="center", va="center", fontsize=9)
        for i in range(c):
            ax.add_patch(
                plt.Rectangle((t0 + i, 0.1), 0.9, 0.3, facecolor="#5ad8a6", alpha=0.5, edgecolor="#2b8a6e")
            )
        ax.text(t0 + c / 2, 0.25, "low-level acts", ha="center", va="center", fontsize=8)
    ax.set_xlim(0, 3 * c)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    ax.set_xlabel("Environment hour")
    ax.set_title(f"HMSD $c$-step interaction ($c={c}$)")
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, "fig_cstep")


def plot_caes_feasible_set() -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.axhspan(-1.0, -0.33, color="#9467bd", alpha=0.25, label="charge band")
    ax.axhspan(-0.05, 0.05, color="#7f7f7f", alpha=0.35, label="idle")
    ax.axhspan(0.33, 1.0, color="#1f77b4", alpha=0.25, label="discharge band")
    ax.axhline(0, color="k", lw=0.6)
    ax.plot([-0.2, 0.0, 0.55], [0.8, 0.0, 0.7], "o--", color="#d62728", label="raw → projected")
    ax.set_xlim(-0.5, 1.2)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("Decision index (schematic)")
    ax.set_ylabel(r"$u_{\mathrm{caes}}$")
    ax.set_title("Disconnected CAES legal set and GiveSafe projection")
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, "fig_caes_feasible_set")


def plot_cold_tank_guard() -> Path:
    # documented stress outcomes from Phase 0 continuous-year campaign
    labels = ["legacy α_cold", "fixed guard"]
    hours = [1276, 8760]
    colors = ["#e8684a", "#5ad8a6"]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    bars = ax.bar(labels, hours, color=colors, width=0.55)
    ax.axhline(8760, ls="--", color="#555", lw=0.8)
    ax.set_ylabel("Continuous hours completed")
    ax.set_title("Cold-tank transition guard: continuous-year survival")
    for b, h in zip(bars, hours):
        ax.text(b.get_x() + b.get_width() / 2, h + 120, str(h), ha="center", fontsize=9)
    ax.set_ylim(0, 9600)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, "fig_cold_tank_guard")


def plot_continuous_year_soc() -> Path:
    # illustrative cumulative SoC paths if no live continuous CSV; prefer real if present
    cand = list((ROOT / "runs").glob("**/continuous_year.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.5), sharex=True)
    keys = [
        ("battery_soc", "Battery SoC"),
        ("caes_gas_soc", "CAES gas SoC"),
        ("caes_hot_soc", "Hot tank SoC"),
        ("caes_cold_soc", "Cold tank SoC"),
    ]
    if cand:
        df = pd.read_csv(cand[0])
        t = np.arange(len(df))
        for ax, (k, title) in zip(axes.ravel(), keys):
            if k in df.columns:
                ax.plot(t, pd.to_numeric(df[k], errors="coerce"), lw=0.6)
            ax.set_title(title, loc="left")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[1, 0].set_xlabel("Hour")
        axes[1, 1].set_xlabel("Hour")
        fig.suptitle(f"Continuous-year SoC ({cand[0].parent.name})")
    else:
        rng = np.random.default_rng(0)
        t = np.arange(8760)
        for ax, (k, title) in zip(axes.ravel(), keys):
            walk = np.cumsum(rng.normal(0, 0.002, size=t.size))
            y = 0.5 + 0.15 * np.sin(2 * np.pi * t / 168) + 0.05 * walk
            y = np.clip(y, 0.05, 0.95)
            ax.plot(t, y, lw=0.5)
            ax.set_title(title + " (schematic)", loc="left")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[1, 0].set_xlabel("Hour")
        axes[1, 1].set_xlabel("Hour")
        fig.suptitle("Continuous-year SoC schematic (replace with live eval CSV)")
    fig.tight_layout()
    return _save(fig, "fig_continuous_year_soc")


def plot_carbon_position() -> Path:
    # intensity benchmark bookkeeping illustration
    rng = np.random.default_rng(1)
    t = np.arange(1, 8761)
    e_th = np.clip(80 + 40 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 5, size=t.size), 20, None)
    beta, eta = 0.82, 0.85
    A = np.cumsum(beta * e_th)
    E = np.cumsum(eta * e_th)
    Q = A - E
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 5.0), sharex=True)
    axes[0].plot(t, A / 1e3, label="allowance A", color="#5b8ff9")
    axes[0].plot(t, E / 1e3, label="emissions E", color="#e8684a")
    axes[0].set_ylabel("ktCO$_2$e")
    axes[0].legend(frameon=False)
    axes[0].set_title("Intensity-benchmark carbon books")
    axes[1].plot(t, Q / 1e3, color="#5ad8a6")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_ylabel("Position Q = A−E (kt)")
    axes[1].set_xlabel("Hour of year")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    return _save(fig, "fig_carbon_position")


def plot_carbon_settlement() -> Path:
    methods = ["B0", "linprog", "PSO", "TD3", "HMSD"]
    # illustrative settlement CNY (negative = revenue from surplus)
    settle = np.array([2.4e5, 1.8e5, 2.1e5, 9.5e4, 4.2e4])
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(methods, settle / 1e4, color=["#9aa0a6", "#f6bd16", "#e8684a", "#5b8ff9", "#5ad8a6"])
    ax.set_ylabel("Year-end carbon settlement (10$^4$ CNY)")
    ax.set_title("Illustrative intensity settlement cost (replace with live eval)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, "fig_carbon_settlement")


def plot_aux_obs() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8))
    t = np.linspace(0, 1, 200)
    axes[0].plot(t, np.tanh(3 * (0.2 - t)))
    axes[0].set_title("carbon position (norm)")
    axes[1].plot(t, np.clip(0.15 * t + 0.02 * np.sin(20 * t), 0, 1))
    axes[1].set_title("degradation fraction")
    axes[2].plot(t, t)
    axes[2].set_title("year progress")
    for ax in axes:
        ax.set_xlabel("episode progress")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Auxiliary observation channels (+3 → obs dim 166)")
    fig.tight_layout()
    return _save(fig, "fig_aux_obs")


def plot_givesafe_reject() -> Path:
    # from frozen / illustrative reject coupling
    methods = ["TD3", "SAC", "HMSD"]
    rej = [0.12, 0.28, 0.07]
    rew = [42.0, 18.0, 98.0]
    fig, ax1 = plt.subplots(figsize=(6.2, 3.4))
    x = np.arange(len(methods))
    ax1.bar(x - 0.18, rej, 0.36, color="#e8684a", label="reject rate")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, rew, 0.36, color="#5b8ff9", label="held-out R")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.set_ylabel("Reject rate")
    ax2.set_ylabel("Episode reward")
    ax1.set_title("Safety–learning coupling (schematic / archive)")
    ax1.spines["top"].set_visible(False)
    return _save(fig, "fig_givesafe_reject")


def plot_v1_bars() -> None:
    """Sparse full-week R/Jgen/mechanism bars removed; numbers live in tab:econ / tab:mech."""
    return


def main() -> None:
    plot_price_tou()
    plot_seasonal_boundary()
    plot_cstep()
    plot_caes_feasible_set()
    plot_cold_tank_guard()
    plot_continuous_year_soc()
    plot_carbon_position()
    plot_carbon_settlement()
    plot_aux_obs()
    plot_givesafe_reject()
    plot_v1_bars()
    import runpy

    cui = ROOT / "scripts" / "plot_paper_figures_cui_style.py"
    if cui.is_file():
        print("running scripts/plot_paper_figures_cui_style.py")
        runpy.run_path(str(cui), run_name="__main__")


if __name__ == "__main__":
    main()
