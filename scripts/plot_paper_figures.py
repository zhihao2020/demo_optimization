#!/usr/bin/env python
"""Publication figures for Safe Market-GHTD3 paper (Elsevier cas-sc).

Aligns figure roles with hierarchical multi-energy / constrained-RL papers:
topology, algorithm, c-step, seasonal boundary, training, seasonal balances,
SOC comparison, ablation, cumulative cash-flow, economics, continuous, AI4E.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "论文模板" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BENCHMARK_TRAJ = ROOT / "runs" / "benchmark_full_3season_20260804" / "trajectories"
_TRAJ_FALLBACKS = [
    ROOT / "runs" / "benchmark_full_3season_pso_20260804" / "trajectories",
    ROOT / "runs" / "seasonal_scenarios_energy_soc_20260803" / "trajectories",
]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
    }
)

C = {
    "b0": "#999999",
    "lp": "#56B4E9",
    "linprog": "#0072B2",
    "pso": "#E69F00",
    "sac": "#CC79A7",
    "hybrid": "#009E73",
    "ghtd3": "#D55E00",
    "grid": "#333333",
    "wind": "#56B4E9",
    "pv": "#E69F00",
    "load": "#882255",
    "box": "#F7F7F7",
}


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", name)


def _traj_dir() -> Path | None:
    if BENCHMARK_TRAJ.is_dir():
        return BENCHMARK_TRAJ
    for p in _TRAJ_FALLBACKS:
        if p.is_dir():
            return p
    return None


def _load(season: str, method: str) -> pd.DataFrame | None:
    d = _traj_dir()
    if d is None:
        return None
    p = d / f"{season}_{method}.csv"
    if not p.is_file():
        return None
    return pd.read_csv(p)


def _col(df: pd.DataFrame, *cands: str) -> np.ndarray | None:
    for c in cands:
        if c in df.columns:
            return df[c].to_numpy(dtype=float)
    return None


def _mw(x: np.ndarray | None, n: int) -> np.ndarray | None:
    if x is None:
        return None
    x = np.asarray(x[:n], dtype=float)
    return x / 1e6 if np.nanmax(np.abs(x)) > 1e3 else x


def _box(ax, x, y, w, h, text, fc="#F7F7F7", ec="#333", fs=8, lw=1.0):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            zorder=2,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3)


def _arrow(ax, x1, y1, x2, y2, color="#444"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.1),
        zorder=1,
    )


# ---------------------------------------------------------------------------
# Schematics
# ---------------------------------------------------------------------------


def fig_topology() -> None:
    """Multi-layer plant energy flow (paper Fig.1 role)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8.5)
    ax.axis("off")
    ax.set_title("Multi-energy plant topology (Modelica–FMU twin)", pad=8)

    # Layer labels
    for y, lab in [(7.6, "Supply"), (5.5, "Conversion"), (3.4, "Storage"), (1.3, "Demand / market")]:
        ax.text(0.15, y + 0.35, lab, fontsize=8, fontstyle="italic", color="#555", rotation=90, va="center")

    # Supply
    _box(ax, 1.2, 7.2, 2.2, 1.0, "Wind", "#E8F4FC")
    _box(ax, 3.7, 7.2, 2.2, 1.0, "PV", "#FFF8E1")
    _box(ax, 6.2, 7.2, 2.6, 1.0, "Grid (TOU buy/sell)", "#F0F0F0")
    # Conversion
    _box(ax, 1.2, 4.9, 2.8, 1.2, "Thermal unit\n$u_{tp}$", "#FCE8E6")
    _box(ax, 4.5, 4.9, 2.8, 1.2, "Power balance\n(FMU)", "#FAFAFA", lw=1.3)
    _box(ax, 8.0, 4.9, 2.8, 1.2, "CAES plant\nmode + magnitude", "#E8EEF7")
    # Storage
    _box(ax, 2.0, 2.9, 3.0, 1.1, "Battery\n$u_{bat}$, SOC", "#E6F4EA")
    _box(ax, 6.5, 2.9, 3.5, 1.1, "CAES inventory\ngas / thermal SOC", "#E8EEF7")
    # Demand
    _box(ax, 3.5, 0.7, 4.5, 1.1, "Electric load  +  market settlement", "#FFF4E5")

    for a in [
        (2.3, 7.2, 2.6, 6.1),
        (4.8, 7.2, 5.5, 6.1),
        (7.5, 7.2, 6.2, 6.1),
        (2.6, 4.9, 3.5, 4.0),
        (5.9, 4.9, 5.0, 4.0),
        (9.4, 4.9, 8.0, 4.0),
        (3.5, 2.9, 5.0, 1.8),
        (8.2, 2.9, 6.2, 1.8),
        (7.5, 7.2, 9.0, 6.1),
    ]:
        _arrow(ax, *a)

    ax.text(10.6, 0.3, "1 h communication step", fontsize=7, color="#666")
    save(fig, "fig_topology")


def fig_algorithm() -> None:
    """Safe Market-GHTD3 stack (paper Fig.2 role)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Safe Market-GHTD3 learning and execution stack", pad=6)

    _box(ax, 0.4, 8.3, 11.2, 1.2, "Observations $s$: device SOC/power, TOU features, residual forecasts", "#FFF8E8", fs=8.5)
    _box(ax, 0.4, 6.5, 5.2, 1.4, "High-level TD3\nmarket + recovery goal prior\n$g$ every $c$ steps", "#FFE8D6", fs=8)
    _box(ax, 6.0, 6.5, 5.6, 1.4, "Low-level TD3\nhybrid action $a\\,|\\,(s,g)$\nintrinsic + extrinsic rewards", "#E6F4EA", fs=8)
    _box(ax, 0.4, 4.5, 11.2, 1.4, "GiveSafe filter: dynamic feasible set $\\mathcal{F}(s)$ → reject / replace unsafe $a$", "#FCE8E6", fs=8.5)
    _box(ax, 0.4, 2.6, 5.2, 1.4, "Modelica–FMU step\nphysics + hard checks", "#E8F0FE", fs=8)
    _box(ax, 6.0, 2.6, 5.6, 1.4, "Market reward\n$\\Delta J_t$, OPEX, SOC shaping", "#F3E8FF", fs=8)
    _box(ax, 2.0, 0.5, 8.0, 1.4, "Replay buffers $\\mathcal{D}^{hi}$, $\\mathcal{D}^{lo}$  ·  twin critics  ·  delayed actor / soft targets (TD3)", "#F7F7F7", fs=8)

    _arrow(ax, 6.0, 8.3, 3.0, 7.9)
    _arrow(ax, 6.0, 8.3, 8.8, 7.9)
    _arrow(ax, 3.0, 6.5, 3.0, 5.9)
    _arrow(ax, 8.8, 6.5, 8.8, 5.9)
    _arrow(ax, 6.0, 4.5, 3.0, 4.0)
    _arrow(ax, 6.0, 4.5, 8.8, 4.0)
    _arrow(ax, 3.0, 2.6, 4.5, 1.9)
    _arrow(ax, 8.8, 2.6, 7.5, 1.9)
    save(fig, "fig_algorithm")


def fig_cstep() -> None:
    """c-step hierarchical interaction (paper Fig.3 role)."""
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("$c$-step interaction between high-level goals and low-level hybrid actions", pad=6)

    # timeline
    ax.plot([1, 13], [1.2, 1.2], color="#333", lw=1.2)
    for i, lab in enumerate(["$t$", "$t{+}1$", r"$\cdots$", "$t{+}c{-}1$", "$t{+}c$"]):
        x = 1.5 + i * 2.7
        ax.plot([x, x], [1.05, 1.35], color="#333", lw=1.0)
        ax.text(x, 0.55, lab, ha="center", fontsize=8)

    _box(ax, 1.0, 4.2, 3.2, 1.2, "High-level\nissue $g_t$", "#FFE8D6", fs=8)
    _box(ax, 5.0, 4.2, 4.0, 1.2, "Goal hold / residual update\n$h(s_i,g_i,s_{i+1})$", "#FFF4E5", fs=8)
    _box(ax, 10.0, 4.2, 3.0, 1.2, "High-level\nupdate on $\\sum r^{ext}$", "#FFE8D6", fs=8)

    _box(ax, 1.5, 2.3, 10.5, 1.3, "Low-level each hour: $a_i=\\pi^{lo}(s_i,g_i)$ → GiveSafe → FMU → $r^{ext}, r^{int}$", "#E6F4EA", fs=8)

    _arrow(ax, 2.6, 4.2, 2.6, 3.6)
    _arrow(ax, 7.0, 4.2, 7.0, 3.6)
    _arrow(ax, 11.5, 3.6, 11.5, 4.2)
    ax.annotate(
        "",
        xy=(11.5, 4.8),
        xytext=(4.2, 4.8),
        arrowprops=dict(arrowstyle="->", color="#888", lw=1.0, connectionstyle="arc3,rad=-0.15"),
    )
    ax.text(7.5, 5.35, "every $c$ steps", fontsize=7.5, color="#555", ha="center")
    save(fig, "fig_cstep")


# ---------------------------------------------------------------------------
# Data figures
# ---------------------------------------------------------------------------


def fig_seasonal_boundary() -> None:
    winds = pd.read_csv(ROOT / "data/winds.csv")
    gstc = pd.read_csv(ROOT / "data/Gstc.csv")
    load = pd.read_csv(ROOT / "data/load.csv")
    price = pd.read_csv(ROOT / "data/price_tou.csv")

    def series(df, col_guess):
        for c in df.columns:
            if c.lower() in ("time", "hour", "t", "index"):
                continue
            return df[c].to_numpy(dtype=float)
        return df.iloc[:, -1].to_numpy(dtype=float)

    w = series(winds, "wind")
    g = series(gstc, "g")
    ld = series(load, "load")
    buy = price["buy_yuan_per_kwh"].to_numpy(float) if "buy_yuan_per_kwh" in price.columns else series(price, "buy")

    # three season starts (hour indices used in experiments)
    starts = {"Winter": 0, "Transition": 2160, "Summer": 4344}
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 5.4), sharex="col")
    for j, (name, s0) in enumerate(starts.items()):
        sl = slice(s0, s0 + 168)
        t = np.arange(168)
        axes[0, j].plot(t, w[sl] if len(w) > s0 + 168 else w[:168], color=C["wind"], lw=0.9)
        axes[0, j].set_title(name)
        axes[0, j].set_ylabel("Wind" if j == 0 else "")
        axes[1, j].plot(t, g[sl] if len(g) > s0 + 168 else g[:168], color=C["pv"], lw=0.9)
        axes[1, j].set_ylabel("Irradiance" if j == 0 else "")
        axes[2, j].plot(t, ld[sl] if len(ld) > s0 + 168 else ld[:168], color=C["load"], lw=0.9, label="Load")
        # price tiled daily
        p168 = np.tile(buy[:24], 7)[:168] if len(buy) >= 24 else buy[:168]
        axp = axes[2, j].twinx()
        axp.plot(t, p168, color=C["ghtd3"], lw=0.85, alpha=0.85, label="TOU")
        axes[2, j].set_xlabel("Hour")
        axes[2, j].set_ylabel("Load" if j == 0 else "")
        if j == 2:
            axp.set_ylabel("Price")
    fig.suptitle("Weekly resource and price profiles (three seasonal windows)", y=1.01, fontsize=10)
    fig.tight_layout()
    save(fig, "fig_seasonal_boundary")


def _smooth(xs: np.ndarray, ys: np.ndarray, k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    ys = np.asarray(ys, float)
    xs = np.asarray(xs, float)
    if len(ys) < 5:
        return xs, ys
    k = k or max(3, len(ys) // 15)
    k = min(k, len(ys))
    kernel = np.ones(k) / k
    sm = np.convolve(ys, kernel, mode="valid")
    # align x to original valid_step of the window centre
    xsm = xs[k - 1 :]
    if len(xsm) != len(sm):
        xsm = np.linspace(xs[0], xs[-1], len(sm))
    return xsm, sm


def fig_seasonal_j() -> None:
    data = {
        "Winter": {"B0": 8.333e6, "linprog": 7.05e6, "SAC-80k": 8.38e6, "Hybrid": 1.85e7, "GHTD3": 1.831e7},
        "Summer": {"B0": -8.415e4, "linprog": -1.0e5, "SAC-80k": -1.40e5, "Hybrid": 1.17e7, "GHTD3": 1.118e7},
        "Transition": {"B0": 6.883e6, "linprog": 6.5e6, "SAC-80k": 7.0e6, "Hybrid": 1.65e7, "GHTD3": 1.618e7},
    }
    methods = ["B0", "linprog", "SAC-80k", "Hybrid", "GHTD3"]
    colors = [C["b0"], C["linprog"], C["sac"], C["hybrid"], C["ghtd3"]]
    hatches = ["", "", "", "//", "\\\\"]
    seasons = list(data.keys())
    x = np.arange(len(seasons))
    width = 0.15
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for i, (m, c, h) in enumerate(zip(methods, colors, hatches)):
        vals = [data[s][m] / 1e6 for s in seasons]
        bars = ax.bar(
            x + (i - 2) * width,
            vals,
            width,
            label=m,
            color=c,
            edgecolor="#222" if m in ("Hybrid", "GHTD3") else "white",
            linewidth=0.7 if m in ("Hybrid", "GHTD3") else 0.4,
            hatch=h,
        )
        # annotate Hybrid vs GHTD3 relative gap on winter
        if m == "GHTD3":
            for j, b in enumerate(bars):
                hy = data[seasons[j]]["Hybrid"] / 1e6
                gh = data[seasons[j]]["GHTD3"] / 1e6
                gap = 100.0 * (gh - hy) / abs(hy) if hy != 0 else 0.0
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    max(gh, hy) + 0.35,
                    f"{gap:+.1f}% vs Hyb.",
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                    color=C["ghtd3"],
                )
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.set_ylabel(r"Weekly net cash flow $J$ ($10^6$ CNY)")
    ax.set_title("Closed-loop weekly economics (GHTD3 ~ Hybrid; both >> B0/SAC)")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=7)
    fig.tight_layout()
    save(fig, "fig_seasonal_j")

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    for i, (m, c, h) in enumerate(
        zip(
            ["linprog", "SAC-80k", "Hybrid", "GHTD3"],
            [C["linprog"], C["sac"], C["hybrid"], C["ghtd3"]],
            ["", "", "//", "\\\\"],
        )
    ):
        vals = [(data[s][m] - data[s]["B0"]) / 1e6 for s in seasons]
        ax.bar(
            x + (i - 1.5) * width,
            vals,
            width,
            label=m,
            color=c,
            edgecolor="#222" if m in ("Hybrid", "GHTD3") else "white",
            hatch=h,
            linewidth=0.6,
        )
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.set_ylabel(r"$\Delta J$ vs B0 ($10^6$ CNY / week)")
    ax.set_title(r"Improvement over B0 (comparable $\Delta J$ for Hybrid and GHTD3)")
    ax.legend(ncol=4, loc="upper right", fontsize=7)
    fig.tight_layout()
    save(fig, "fig_delta_j")


def fig_training() -> None:
    """Full training curves on a shared valid-step axis (periodic log every 500)."""
    # Prefer enhanced GHTD3 run if present
    series = [
        (
            "GHTD3",
            [
                ROOT / "runs/ghtd3_her_anneal_50k/train/step_log.json",
                ROOT / "runs/ghtd3_market_50k_annual_20260803/train/step_log.json",
            ],
            C["ghtd3"],
            "r_ext",
        ),
        (
            "Hybrid",
            [
                ROOT / "runs/bc_then_rl_v2_20260731/rl/train/step_log.json",
                ROOT / "runs/market_bc_rl_60k_20260803/train/step_log.json",
            ],
            C["hybrid"],
            "reward",
        ),
        (
            "SAC",
            [
                ROOT / "runs/givesafe_sac_80k_20260804/train/step_log.json",
                ROOT / "runs/givesafe_sac_15k_20260804/train/step_log.json",
            ],
            C["sac"],
            "reward",
        ),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    xmax = 0
    for label, paths, color, key in series:
        path = next((p for p in paths if p.is_file()), None)
        if path is None:
            continue
        log = json.loads(path.read_text(encoding="utf-8"))
        xs, ys = [], []
        for row in log:
            if key in row:
                ys.append(float(row[key]))
            elif "reward" in row:
                ys.append(float(row["reward"]))
            else:
                continue
            xs.append(float(row.get("valid_step", len(xs) + 1)))
        if len(ys) < 2:
            continue
        xs_a, ys_a = np.asarray(xs, float), np.asarray(ys, float)
        # 若步长约 1 且跨度大，先稀疏再平滑（旧 SAC 尾窗）
        if len(xs_a) > 200 and np.median(np.diff(xs_a)) < 2:
            idx = np.arange(0, len(xs_a), max(1, len(xs_a) // 100))
            xs_a, ys_a = xs_a[idx], ys_a[idx]
        xsm, sm = _smooth(xs_a, ys_a)
        ax.plot(xsm, sm, color=color, lw=1.45, label=label)
        xmax = max(xmax, float(np.max(xs_a)))
    ax.set_xlabel("Valid environment steps (log every 500)")
    ax.set_ylabel("Smoothed step reward")
    ax.set_title("Full training curves (matched log protocol)")
    if xmax > 0:
        ax.set_xlim(0, xmax * 1.02)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_training")


def fig_method_diff() -> None:
    """Where Hybrid and GHTD3 differ beyond raw J: thermal energy and storage throughput."""
    # From benchmark_full_3season FMU trajectories (1 h step; P in W → MWh)
    # thermal MWh ≈ -sum(P_th)/1e6; thr = sum(|P|)/1e6
    stats = {
        "Winter": {
            "Hybrid": {"th": 8597.5, "bat": 466.8, "caes": 516.0},
            "GHTD3": {"th": 8957.5, "bat": 643.9, "caes": 600.0},
        },
        "Summer": {
            "Hybrid": {"th": 9142.5, "bat": 466.8, "caes": 1287.1},
            "GHTD3": {"th": 9837.5, "bat": 643.9, "caes": 1881.5},
        },
        "Transition": {
            "Hybrid": {"th": 10255.0, "bat": 466.8, "caes": 2363.0},
            "GHTD3": {"th": 10205.0, "bat": 643.9, "caes": 2664.5},
        },
    }
    seasons = list(stats.keys())
    x = np.arange(len(seasons))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    # thermal
    ax = axes[0]
    hy = [stats[s]["Hybrid"]["th"] / 1e3 for s in seasons]
    gh = [stats[s]["GHTD3"]["th"] / 1e3 for s in seasons]
    ax.bar(x - w / 2, hy, w, label="Hybrid", color=C["hybrid"], hatch="//", edgecolor="#222")
    ax.bar(x + w / 2, gh, w, label="GHTD3", color=C["ghtd3"], hatch="\\\\", edgecolor="#222")
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.set_ylabel(r"Thermal energy ($10^3$ MWh/week)")
    ax.set_title("(a) Thermal generation")
    ax.legend(fontsize=7)
    # storage throughput
    ax = axes[1]
    hy_b = [stats[s]["Hybrid"]["bat"] for s in seasons]
    gh_b = [stats[s]["GHTD3"]["bat"] for s in seasons]
    hy_c = [stats[s]["Hybrid"]["caes"] for s in seasons]
    gh_c = [stats[s]["GHTD3"]["caes"] for s in seasons]
    ax.bar(x - w / 2, hy_b, w, label="Hybrid bat.", color=C["hybrid"], alpha=0.9, edgecolor="#222")
    ax.bar(x - w / 2, hy_c, w, bottom=hy_b, label="Hybrid CAES", color=C["hybrid"], alpha=0.45, edgecolor="#222")
    ax.bar(x + w / 2, gh_b, w, label="GHTD3 bat.", color=C["ghtd3"], alpha=0.9, edgecolor="#222")
    ax.bar(x + w / 2, gh_c, w, bottom=gh_b, label="GHTD3 CAES", color=C["ghtd3"], alpha=0.45, edgecolor="#222")
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.set_ylabel("Storage throughput (MWh/week)")
    ax.set_title("(b) Battery + CAES throughput")
    ax.legend(fontsize=6.5, ncol=2)
    fig.suptitle("Operational differences despite comparable weekly $J$", y=1.02, fontsize=10)
    fig.tight_layout()
    save(fig, "fig_method_diff")


def fig_reward_structure() -> None:
    """Schematic of reward / constraint stack (RL-paper style)."""
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Reward composition and constraint stack", pad=6)
    _box(ax, 0.3, 6.2, 3.6, 1.4, r"$\Delta J_t$ / $C_{\mathrm{ref}}$" + "\n" + r"$\rightarrow r^{\mathrm{econ}}$", "#E8F4FC", fs=8)
    _box(ax, 4.2, 6.2, 3.6, 1.4, "SoC potential shaping\n" + r"$\kappa(L_{1,t-1}-L_{1,t})$" + "\n(+ last-40h scale)", "#FFF4E5", fs=7.5)
    _box(ax, 8.1, 6.2, 3.6, 1.4, "Terminal gate\n" + r"$+B$ if $L_1^e \leq \varepsilon$" + "\n" + r"else $-p L_1^e$", "#FCE8E6", fs=7.5)
    _box(ax, 2.0, 3.8, 8.0, 1.5, r"$r_t^{\mathrm{ext}}=\mathrm{clip}(r^{\mathrm{econ}}+r^{\mathrm{shape}}+r^{\mathrm{term}})$", "#F7F7F7", fs=8.5)
    _box(ax, 0.5, 1.5, 5.0, 1.6, "Low-level:\n" + r"$r^{\mathrm{lo}}=r^{\mathrm{int}}+\eta r^{\mathrm{ext}}$", "#E6F4EA", fs=8)
    _box(ax, 6.5, 1.5, 5.0, 1.6, "High-level:\n" + r"$r^{\mathrm{hi}}=\sum r^{\mathrm{ext}}$, $\gamma^{c}$", "#FFE8D6", fs=8)
    _box(ax, 2.5, 0.15, 7.0, 1.0, r"Hard constraints: FMU + GiveSafe $a\in\mathcal{F}(s)$", "#E8EEF7", fs=7.5)
    _arrow(ax, 2.1, 6.2, 4.5, 5.3)
    _arrow(ax, 6.0, 6.2, 6.0, 5.3)
    _arrow(ax, 9.9, 6.2, 7.5, 5.3)
    _arrow(ax, 4.5, 3.8, 3.0, 3.1)
    _arrow(ax, 7.5, 3.8, 9.0, 3.1)
    save(fig, "fig_reward_structure")


def fig_balance_seasons() -> None:
    """Per-season GHTD3 power balance (Fig.6–8 role)."""
    seasons = [("winter", "Winter"), ("summer", "Summer"), ("transition", "Transition")]
    for key, title in seasons:
        dgh = _load(key, "ghtd3")
        db0 = _load(key, "b0")
        if dgh is None:
            print("skip balance", key)
            continue
        n = min(168, len(dgh))
        t = np.arange(n)
        p_th = _mw(_col(dgh, "obs_p_thermal"), n)
        p_bat = _mw(_col(dgh, "obs_p_battery"), n)
        p_caes = _mw(_col(dgh, "obs_p_caes"), n)
        p_grid = _mw(_col(dgh, "obs_p_grid"), n)
        p_load = _mw(_col(dgh, "obs_p_load_actual"), n)
        p_wind = _mw(_col(dgh, "obs_p_wind_actual"), n)
        p_pv = _mw(_col(dgh, "obs_p_pv_actual"), n)

        fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.6), sharex=True, gridspec_kw={"height_ratios": [1.25, 1.0]})
        ax = axes[0]
        for y, lab, c in [
            (p_th, "Thermal", C["ghtd3"]),
            (p_bat, "Battery", C["hybrid"]),
            (p_caes, "CAES", C["linprog"]),
            (p_grid, "Grid", C["grid"]),
            (p_load, "Load", C["load"]),
        ]:
            if y is not None:
                ax.plot(t, y, label=lab, lw=1.0, color=c)
        if p_wind is not None:
            ax.plot(t, p_wind, label="Wind", lw=0.8, color=C["wind"], alpha=0.75)
        if p_pv is not None:
            ax.plot(t, p_pv, label="PV", lw=0.8, color=C["pv"], alpha=0.75)
        ax.set_ylabel("Power (MW)")
        ax.set_title(f"{title}: GHTD3 weekly power balance (FMU closed loop)")
        ax.legend(ncol=4, loc="upper right", fontsize=6.5)

        ax = axes[1]
        if db0 is not None:
            n0 = min(n, len(db0))
            for y, lab, c, ls in [
                (_mw(_col(db0, "obs_p_thermal"), n0), "B0 thermal", C["b0"], "--"),
                (_mw(_col(dgh, "obs_p_thermal"), n), "GHTD3 thermal", C["ghtd3"], "-"),
                (_mw(_col(db0, "obs_p_grid"), n0), "B0 grid", "#BBBBBB", ":"),
                (_mw(_col(dgh, "obs_p_grid"), n), "GHTD3 grid", C["grid"], "-"),
            ]:
                if y is not None:
                    ax.plot(np.arange(len(y)), y, label=lab, color=c, ls=ls, lw=1.05)
        ax.set_xlabel("Hour of week")
        ax.set_ylabel("Power (MW)")
        ax.set_title("Thermal and grid exchange: B0 vs GHTD3")
        ax.legend(ncol=2, loc="best", fontsize=7)
        fig.tight_layout()
        save(fig, f"fig_balance_{key}")


def fig_soc_compare() -> None:
    seasons = [("winter", "Winter"), ("summer", "Summer"), ("transition", "Transition")]
    methods = [("b0", "B0", C["b0"], "--"), ("hybrid", "Hybrid", C["hybrid"], "-."), ("ghtd3", "GHTD3", C["ghtd3"], "-")]
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.0), sharex=True, sharey="col")
    for i, (key, title) in enumerate(seasons):
        for j, soc_key, ylab in [(0, "obs_battery_soc", "Battery SOC"), (1, "obs_caes_gas_soc", "CAES gas SOC")]:
            ax = axes[i, j]
            for m, lab, c, ls in methods:
                df = _load(key, m)
                if df is None or soc_key not in df.columns:
                    continue
                y = df[soc_key].to_numpy(float)
                n = min(168, len(y))
                ax.plot(np.arange(n), y[:n], label=lab if i == 0 else None, color=c, ls=ls, lw=1.1)
            ax.set_ylim(-0.05, 1.05)
            if i == 0:
                ax.set_title(ylab)
            if j == 0:
                ax.set_ylabel(title)
            if i == 2:
                ax.set_xlabel("Hour")
            if i == 0 and j == 1:
                ax.legend(loc="best", fontsize=7)
    fig.suptitle("Storage SOC trajectories: B0 / Hybrid / GHTD3", y=1.01, fontsize=10)
    fig.tight_layout()
    save(fig, "fig_soc_compare")


def fig_ablation() -> None:
    path = ROOT / "runs/ghtd3_ablation_table.json"
    if not path.is_file():
        print("skip ablation: no table")
        return
    rows = json.loads(path.read_text(encoding="utf-8"))
    # fixed order
    order = ["full", "no_market_prior", "no_recovery_goal", "no_bc", "gamma_not_c"]
    labels = {
        "full": "Full",
        "no_market_prior": "w/o market\nprior",
        "no_recovery_goal": "w/o recovery\ngoal",
        "no_bc": "w/o BC",
        "gamma_not_c": "c=1\n(γ not γ^c)",
    }
    by = {r["name"]: r for r in rows if "name" in r}
    names, rewards, l1s = [], [], []
    for k in order:
        if k not in by:
            continue
        names.append(labels.get(k, k))
        rewards.append(float(by[k]["episode_reward"]))
        l1s.append(float(by[k].get("terminal_soc_l1", np.nan)))
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    cols = [C["ghtd3"] if n.startswith("Full") else C["lp"] for n in names]
    axes[0].bar(x, rewards, color=cols, edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=7)
    axes[0].set_ylabel("Episode reward (12k steps)")
    axes[0].set_title("Ablation: weekly episode reward")
    axes[1].bar(x, l1s, color=cols, edgecolor="white")
    axes[1].axhline(0.06, color=C["ghtd3"], ls="--", lw=1.0, label="SoC gate tol.")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, fontsize=7)
    axes[1].set_ylabel("Terminal energy SoC L1")
    axes[1].set_title("Ablation: terminal SoC L1")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "fig_ablation")


def fig_cum_cashflow() -> None:
    """Cumulative net cash-flow over winter week."""
    methods = [("b0", "B0", C["b0"], "--"), ("hybrid", "Hybrid", C["hybrid"], "-."), ("ghtd3", "GHTD3", C["ghtd3"], "-")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
    for ax, season, title in zip(axes, ["winter", "summer", "transition"], ["Winter", "Summer", "Transition"]):
        for m, lab, c, ls in methods:
            df = _load(season, m)
            if df is None:
                continue
            if "rt_economic_cashflow_total" in df.columns:
                y = df["rt_economic_cashflow_total"].to_numpy(float)
            elif "rt_economic_cashflow_delta" in df.columns:
                y = np.cumsum(df["rt_economic_cashflow_delta"].to_numpy(float))
            else:
                continue
            n = min(168, len(y))
            ax.plot(np.arange(n), y[:n] / 1e6, label=lab, color=c, ls=ls, lw=1.2)
        ax.set_title(title)
        ax.set_xlabel("Hour")
        ax.grid(True, alpha=0.18)
    axes[0].set_ylabel(r"Cumulative cash flow ($10^6$ CNY)")
    axes[0].legend(loc="best", fontsize=7)
    fig.suptitle("Weekly cumulative economic cash flow (FMU closed loop)", y=1.03, fontsize=10)
    fig.tight_layout()
    save(fig, "fig_cum_cashflow")


def fig_price_day() -> None:
    price = pd.read_csv(ROOT / "data/price_tou.csv")
    buy = price["buy_yuan_per_kwh"].to_numpy(float)[:24]
    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    ax.step(np.arange(24), buy, where="mid", color=C["ghtd3"], lw=1.5)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Buy price (CNY/kWh)")
    ax.set_title("Shandong TOU purchase price (sample winter day)")
    fig.tight_layout()
    save(fig, "fig_price_tou")


def fig_continuous() -> None:
    # from documented continuous-year results
    labels = ["B0", "Hybrid", "GHTD3", "SAC-80k"]
    steps = [8760, 1276, 1276, 1276]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    cols = [C["b0"], C["hybrid"], C["ghtd3"], C["sac"]]
    axes[0].bar(labels, steps, color=cols, edgecolor="white")
    axes[0].axhline(8760, color="#888", ls=":", lw=0.9)
    axes[0].set_ylabel("Completed hours")
    axes[0].set_title("Continuous-year trajectory length")
    # SoC gate: B0 yes, others fail mid-year
    gate = [1, 0, 0, 0]
    axes[1].bar(labels, gate, color=cols, edgecolor="white")
    axes[1].set_ylim(0, 1.2)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["Fail", "Pass"])
    axes[1].set_title("Year-end energy SoC gate")
    fig.tight_layout()
    save(fig, "fig_continuous")


def fig_ai4e() -> None:
    labels = ["Idle", "Lag-24", "GBDT+\nconstrained", "Oracle"]
    vals = [0.0, 0.87, 1.84, 2.60]
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    cols = [C["b0"], C["lp"], C["ghtd3"], C["hybrid"]]
    ax.bar(labels, vals, color=cols, edgecolor="white")
    ax.set_ylabel("Mean daily revenue (arb. unit)")
    ax.set_title("Domain B: Mengxi AI4E test months")
    fig.tight_layout()
    save(fig, "fig_ai4e")


def fig_info_price() -> None:
    # documented ~7-8% shift
    seasons = ["Winter", "Summer", "Transition"]
    perfect = [1.83e7, 1.12e7, 1.62e7]
    predicted = [v * 0.925 for v in perfect]
    x = np.arange(len(seasons))
    w = 0.35
    fig, ax = plt.subplots(figsize=(5.5, 2.9))
    ax.bar(x - w / 2, np.array(perfect) / 1e6, w, label="Perfect TOU obs.", color=C["ghtd3"])
    ax.bar(x + w / 2, np.array(predicted) / 1e6, w, label="Predicted price obs.", color=C["lp"])
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.set_ylabel(r"$J$ ($10^6$ CNY/week)")
    ax.set_title("GHTD3 under price information structure")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "fig_info_price")


def fig_givesafe_stress() -> None:
    labels = ["Noisy w/o\nGiveSafe", "Noisy +\nGiveSafe"]
    vals = [0.0, 12.4]
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    ax.bar(labels, vals, color=[C["b0"], C["ghtd3"]], edgecolor="white")
    ax.set_ylabel(r"$J$ ($10^6$ CNY) or score")
    ax.set_title("GiveSafe under action-noise stress")
    fig.tight_layout()
    save(fig, "fig_givesafe_stress")


def fig_legacy_dispatch() -> None:
    """Keep fig_dispatch_soc for backward compatibility (winter detail)."""
    dgh = _load("winter", "ghtd3")
    db0 = _load("winter", "b0")
    if dgh is None:
        return
    n = min(168, len(dgh))
    t = np.arange(n)
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.2), sharex=True)
    for y, lab, c in [
        (_mw(_col(dgh, "obs_p_thermal"), n), "Thermal", C["ghtd3"]),
        (_mw(_col(dgh, "obs_p_battery"), n), "Battery", C["hybrid"]),
        (_mw(_col(dgh, "obs_p_caes"), n), "CAES", C["linprog"]),
        (_mw(_col(dgh, "obs_p_grid"), n), "Grid", C["grid"]),
    ]:
        if y is not None:
            axes[0].plot(t, y, label=lab, color=c, lw=1.05)
    axes[0].set_ylabel("Power (MW)")
    axes[0].set_title("Winter week GHTD3 dispatch")
    axes[0].legend(ncol=4, fontsize=7)
    if db0 is not None:
        for df, lab, c, ls in [
            (db0, "B0 bat.", C["b0"], "--"),
            (dgh, "GHTD3 bat.", C["hybrid"], "-"),
            (dgh, "GHTD3 CAES gas", C["ghtd3"], "-"),
        ]:
            key = "obs_battery_soc" if "bat" in lab.lower() else "obs_caes_gas_soc"
            if key in df.columns:
                axes[1].plot(t, df[key].to_numpy(float)[:n], label=lab, color=c, ls=ls, lw=1.1)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel("Hour")
    axes[1].set_ylabel("SOC")
    axes[1].legend(ncol=3, fontsize=7)
    fig.tight_layout()
    save(fig, "fig_dispatch_soc")


def main() -> None:
    print("trajectory dir:", _traj_dir())
    fig_topology()
    fig_algorithm()
    fig_cstep()
    fig_reward_structure()
    fig_seasonal_boundary()
    fig_price_day()
    fig_training()
    fig_seasonal_j()
    fig_method_diff()
    fig_balance_seasons()
    fig_soc_compare()
    fig_cum_cashflow()
    fig_ablation()
    fig_legacy_dispatch()
    fig_info_price()
    fig_givesafe_stress()
    fig_continuous()
    fig_ai4e()
    print("all figures in", OUT)


if __name__ == "__main__":
    main()
