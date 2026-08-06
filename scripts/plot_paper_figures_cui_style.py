#!/usr/bin/env python
"""Cui-style paper figures: power balance, SoC matrix, cost, sensitivity, ablation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAJ = ROOT / "runs" / "paper_dispatch_traj"
FIG = ROOT / "Paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SEASONS = ("winter", "transition", "summer")
METHODS_BALANCE = ("ghtd3", "td3", "pso", "linprog")
METHOD_LABEL = {
    "ghtd3": "Safe Market-GHTD3",
    "td3": "TD3-scratch",
    "pso": "PSO",
    "linprog": "linprog MPC",
    "b0": "B0 (rule)",
}
COLORS = {
    "thermal": "#d62728",
    "wind": "#2ca02c",
    "pv": "#ff7f0e",
    "battery": "#1f77b4",
    "caes": "#9467bd",
    "grid": "#7f7f7f",
    "load": "#000000",
}


def _mw(series: pd.Series) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    # obs powers in some exports are raw FMU scale; if |median| >> 1e4, treat as W→MW
    med = np.nanmedian(np.abs(x[np.isfinite(x)])) if np.any(np.isfinite(x)) else 0.0
    if med > 1e4:
        x = x / 1e6
    return x


def load_traj(season: str, method: str) -> pd.DataFrame | None:
    p = TRAJ / f"{season}_{method}.csv"
    if not p.is_file():
        return None
    return pd.read_csv(p)


def plot_balance_season(season: str) -> Path | None:
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 11.0), sharex=True)
    any_ok = False
    for ax, method in zip(axes, METHODS_BALANCE):
        df = load_traj(season, method)
        ax.set_title(METHOD_LABEL.get(method, method), loc="left", fontsize=11)
        if df is None or len(df) == 0:
            ax.text(0.5, 0.5, "no trajectory", ha="center", va="center", transform=ax.transAxes)
            continue
        any_ok = True
        t = np.arange(len(df))
        load = _mw(df["obs_p_load_actual"]) if "obs_p_load_actual" in df else np.zeros(len(df))
        thermal = _mw(df["obs_p_thermal"]) if "obs_p_thermal" in df else np.zeros(len(df))
        wind = _mw(df["obs_p_wind_actual"]) if "obs_p_wind_actual" in df else np.zeros(len(df))
        pv = _mw(df["obs_p_pv_actual"]) if "obs_p_pv_actual" in df else np.zeros(len(df))
        bat = _mw(df["obs_p_battery"]) if "obs_p_battery" in df else np.zeros(len(df))
        caes = _mw(df["obs_p_caes"]) if "obs_p_caes" in df else np.zeros(len(df))
        grid = _mw(df["obs_p_grid"]) if "obs_p_grid" in df else np.zeros(len(df))

        # stacked generation-like positive sources (approx view)
        pos_th = np.clip(thermal, 0, None)
        pos_w = np.clip(wind, 0, None)
        pos_pv = np.clip(pv, 0, None)
        pos_bat = np.clip(bat, 0, None)
        pos_caes = np.clip(caes, 0, None)
        # discharge positive for storage in plot convention: battery discharge often positive electric out
        ax.stackplot(
            t,
            pos_th,
            pos_w,
            pos_pv,
            pos_bat,
            pos_caes,
            labels=["Thermal", "Wind", "PV", "Battery+", "CAES+"],
            colors=[COLORS["thermal"], COLORS["wind"], COLORS["pv"], COLORS["battery"], COLORS["caes"]],
            alpha=0.85,
        )
        ax.plot(t, load, color=COLORS["load"], lw=1.2, label="Load")
        ax.plot(t, grid, color=COLORS["grid"], lw=1.0, ls="--", label="Grid")
        ax.set_ylabel("Power (MW)")
        ax.grid(True, alpha=0.25)
        if method == METHODS_BALANCE[0]:
            ax.legend(loc="upper right", ncol=4, fontsize=8, frameon=False)
    axes[-1].set_xlabel("Hour in weekly episode")
    fig.suptitle(f"Electric power balance — {season}", y=0.995, fontsize=13)
    fig.tight_layout()
    out = FIG / f"fig_balance_{season}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out if any_ok else None


def plot_soc_matrix() -> Path:
    methods = ("ghtd3", "td3", "pso", "b0")
    fig, axes = plt.subplots(len(SEASONS), len(methods), figsize=(12.5, 8.5), sharex=True, sharey=True)
    for i, season in enumerate(SEASONS):
        for j, method in enumerate(methods):
            ax = axes[i, j]
            df = load_traj(season, method)
            if df is None or len(df) == 0:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
            else:
                t = np.arange(len(df))
                if "obs_battery_soc" in df:
                    ax.plot(t, df["obs_battery_soc"], label="Battery", color=COLORS["battery"], lw=1.2)
                if "obs_caes_gas_soc" in df:
                    ax.plot(t, df["obs_caes_gas_soc"], label="CAES gas", color=COLORS["caes"], lw=1.2)
                ax.set_ylim(-0.05, 1.05)
                ax.grid(True, alpha=0.25)
            if i == 0:
                ax.set_title(METHOD_LABEL.get(method, method), fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{season}\nSoC")
            if i == len(SEASONS) - 1:
                ax.set_xlabel("Hour")
            if i == 0 and j == len(methods) - 1:
                ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("Battery and CAES gas SoC across seasons and methods", y=1.01)
    fig.tight_layout()
    out = FIG / "fig_soc_seasons.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cum_cashflow() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), sharey=True)
    for ax, season in zip(axes, SEASONS):
        for method, color in (
            ("ghtd3", "#d62728"),
            ("td3", "#1f77b4"),
            ("pso", "#ff7f0e"),
            ("b0", "#7f7f7f"),
        ):
            df = load_traj(season, method)
            if df is None or "rt_economic_cashflow_delta" not in df.columns:
                continue
            cum = pd.to_numeric(df["rt_economic_cashflow_delta"], errors="coerce").fillna(0).cumsum()
            # scale to 1e6 CNY if large
            scale = 1e6 if cum.abs().max() > 1e5 else 1.0
            ax.plot(cum.to_numpy() / scale, label=METHOD_LABEL.get(method, method), color=color, lw=1.4)
        ax.set_title(season.capitalize())
        ax.set_xlabel("Hour")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Cumulative cash-flow (×10⁶ CNY)" if True else "Cumulative ΔJ")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Weekly cumulative market cash-flow", y=1.02)
    fig.tight_layout()
    out = FIG / "fig_cum_cashflow.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cost_components() -> Path:
    """Bar chart of aggregated cost components from trajectories (seed-1 exports)."""
    comps = [
        ("rt_market_buy_cost", "Buy"),
        ("rt_market_sell_revenue", "Sell"),
        ("rt_raw_thermal_cost", "Thermal"),
        ("rt_raw_battery_cost", "Battery"),
        ("rt_raw_caes_cost", "CAES"),
    ]
    methods = ("ghtd3", "td3", "pso", "linprog", "b0")
    # one figure with 3 season panels
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
    x = np.arange(len(comps))
    width = 0.15
    for ax, season in zip(axes, SEASONS):
        for k, method in enumerate(methods):
            df = load_traj(season, method)
            vals = []
            for col, _ in comps:
                if df is None or col not in df.columns:
                    vals.append(0.0)
                else:
                    vals.append(float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum()) / 1e6)
            ax.bar(x + (k - 2) * width, vals, width=width, label=METHOD_LABEL.get(method, method))
        ax.set_xticks(x)
        ax.set_xticklabels([c[1] for c in comps], rotation=20)
        ax.set_title(season.capitalize())
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Weekly sum (×10⁶ CNY)")
    axes[2].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Cost / revenue components (weekly sums)", y=1.02)
    fig.tight_layout()
    out = FIG / "fig_cost_components.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_ablation_bars() -> Path | None:
    # Hard-coded from paper tab:ablation (seed0 short budget)
    variants = ["Full(35k)", "w/o MSGP", "w/o MS-HER", "w/o F-MLE"]
    # mean delta vs TD3
    mean_delta = [42.7, 39.9, 11.4, 41.9]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    colors = ["#2ca02c", "#1f77b4", "#d62728", "#ff7f0e"]
    ax.bar(variants, mean_delta, color=colors, edgecolor="k", lw=0.4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel(r"Mean seasonal $\Delta$ vs TD3-scratch")
    ax.set_title("Component ablations (seed 0)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG / "fig_ablation_bars.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_sensitivity(summary_path: Path | None = None) -> Path | None:
    """Plot c / alpha sensitivity if summary JSON exists."""
    path = summary_path or (ROOT / "runs" / "ghtd3_sens_summary.json")
    if not path.is_file():
        # placeholder with expected axes for layout tests — skip if no data
        print(f"[skip] sensitivity summary missing: {path}")
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    # expected format: {"c": [{"value":8,"reward_mean":..},...], "alpha":[...]}
    for ax, key, xlabel in (
        (axes[0], "c", r"Subgoal interval $c$"),
        (axes[1], "alpha", r"Intrinsic weight $\alpha_{\mathrm{end}}$"),
    ):
        pts = data.get(key) or []
        if not pts:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        xs = [p["value"] for p in pts]
        ys = [p["reward_mean"] for p in pts]
        yerr = [p.get("reward_std", 0) for p in pts]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", color="#d62728", capsize=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean weekly episode reward")
        ax.grid(True, alpha=0.3)
    axes[0].set_title("(a) Subgoal interval")
    axes[1].set_title(r"(b) Intrinsic $\alpha$")
    fig.suptitle("Sensitivity of Safe Market-GHTD3 (seed 0, matched budget)", y=1.02)
    fig.tight_layout()
    out = FIG / "fig_sensitivity_c_alpha.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    outs = []
    for season in SEASONS:
        p = plot_balance_season(season)
        if p:
            outs.append(p)
            print("wrote", p)
    outs.append(plot_soc_matrix())
    print("wrote", outs[-1])
    outs.append(plot_cum_cashflow())
    print("wrote", outs[-1])
    outs.append(plot_cost_components())
    print("wrote", outs[-1])
    outs.append(plot_ablation_bars())
    print("wrote", outs[-1])
    sens = plot_sensitivity()
    if sens:
        outs.append(sens)
        print("wrote", sens)
    print("done", len(outs), "figures")


if __name__ == "__main__":
    main()
