#!/usr/bin/env python
"""Cui-style paper figures: power balance, SoC matrix, cost, sensitivity, ablation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "runs" / "seasonal_v1"
FIG = ROOT / "Paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SEASONS = ("winter", "transition", "summer")
FULL_WEEK = 168
METHODS_BALANCE = ("hmsd", "td3", "pso", "linprog")
METHOD_LABEL = {
    "hmsd": "HMSD",
    "td3": "TD3",
    "pso": "PSO",
    "linprog": "linprog MPC",
    "sac": "SAC",
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
    peak = np.nanmax(np.abs(x[np.isfinite(x)])) if np.any(np.isfinite(x)) else 0.0
    if peak > 1e4:
        x = x / 1e6
    return x


def _inj_mw(series: pd.Series) -> np.ndarray:
    """Bus injection as positive MW (FMU generation / discharge is negative W)."""
    return np.clip(-_mw(series), 0.0, None)


def load_traj(season: str, method: str) -> pd.DataFrame | None:
    alias = {"ghtd3": "hmsd"}
    m = alias.get(method, method)
    p = V1 / season / f"{m}_s0" / "trajectories" / "eval.csv"
    if not p.is_file():
        return None
    return pd.read_csv(p)


def load_full_week_traj(season: str, method: str) -> pd.DataFrame | None:
    """Return a 168 h trajectory only; truncated evals are not plotted."""
    df = load_traj(season, method)
    if df is None or len(df) < FULL_WEEK:
        return None
    return df.iloc[:FULL_WEEK].copy()


def _draw_balance(ax, df: pd.DataFrame, *, legend: bool) -> None:
    t = np.arange(FULL_WEEK)
    load = _mw(df["obs_p_load_actual"]) if "obs_p_load_actual" in df else np.zeros(FULL_WEEK)
    thermal = _inj_mw(df["obs_p_thermal"]) if "obs_p_thermal" in df else np.zeros(FULL_WEEK)
    wind = _inj_mw(df["obs_p_wind_actual"]) if "obs_p_wind_actual" in df else np.zeros(FULL_WEEK)
    pv = _inj_mw(df["obs_p_pv_actual"]) if "obs_p_pv_actual" in df else np.zeros(FULL_WEEK)
    bat = _inj_mw(df["obs_p_battery"]) if "obs_p_battery" in df else np.zeros(FULL_WEEK)
    caes = _inj_mw(df["obs_p_caes"]) if "obs_p_caes" in df else np.zeros(FULL_WEEK)
    grid = _mw(df["obs_p_grid"]) if "obs_p_grid" in df else np.zeros(FULL_WEEK)
    ax.stackplot(
        t,
        thermal,
        wind,
        pv,
        bat,
        caes,
        labels=["Thermal", "Wind", "PV", "Battery disch.", "CAES disch."],
        colors=[COLORS["thermal"], COLORS["wind"], COLORS["pv"], COLORS["battery"], COLORS["caes"]],
        alpha=0.85,
    )
    ax.plot(t, load, color=COLORS["load"], lw=1.2, label="Load")
    ax.plot(t, grid, color=COLORS["grid"], lw=1.0, ls="--", label="Grid (+ import)")
    ax.set_xlim(0, FULL_WEEK - 1)
    ax.set_ylabel("Power (MW)")
    ax.grid(True, alpha=0.25)
    if legend:
        ax.legend(loc="upper right", ncol=4, fontsize=8, frameon=False)


def plot_balance_fullweek(season: str, methods: tuple[str, ...], outfile: str) -> Path | None:
    frames = [(m, load_full_week_traj(season, m)) for m in methods]
    frames = [(m, df) for m, df in frames if df is not None]
    if not frames:
        print(f"[skip] no full-week traj for {season} {methods}")
        return None
    fig, axes = plt.subplots(len(frames), 1, figsize=(10.5, 2.7 * len(frames) + 0.6), sharex=True)
    if len(frames) == 1:
        axes = [axes]
    for i, (method, df) in enumerate(frames):
        axes[i].set_title(METHOD_LABEL.get(method, method), loc="left", fontsize=11)
        _draw_balance(axes[i], df, legend=(i == 0))
    axes[-1].set_xlabel("Hour in weekly episode")
    fig.suptitle(f"Electric power balance — {season} (full {FULL_WEEK} h only)", y=0.995, fontsize=12)
    fig.tight_layout()
    out = FIG / outfile
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_balance_season(season: str) -> Path | None:
    """Kept for callers; still draws only methods that finished 168 h."""
    return plot_balance_fullweek(season, METHODS_BALANCE, f"fig_balance_{season}.png")


def plot_soc_matrix() -> Path:
    """Full-week SoC only: transition HMSD/linprog and winter linprog (PSO has no eval.csv)."""
    cells = (
        ("transition", "hmsd"),
        ("transition", "linprog"),
        ("winter", "linprog"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2), sharey=True)
    for ax, (season, method) in zip(axes, cells):
        df = load_full_week_traj(season, method)
        ax.set_title(f"{season.capitalize()} / {METHOD_LABEL.get(method, method)}", fontsize=10)
        if df is None:
            ax.text(0.5, 0.5, "no full-week traj", ha="center", va="center", transform=ax.transAxes)
        else:
            t = np.arange(FULL_WEEK)
            if "obs_battery_soc" in df:
                ax.plot(t, df["obs_battery_soc"], label="Battery", color=COLORS["battery"], lw=1.2)
            if "obs_caes_gas_soc" in df:
                ax.plot(t, df["obs_caes_gas_soc"], label="CAES gas", color=COLORS["caes"], lw=1.2)
            ax.set_xlim(0, FULL_WEEK - 1)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.25)
        ax.set_xlabel("Hour")
    axes[0].set_ylabel("SoC")
    axes[-1].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle(f"Battery and CAES gas SoC (full {FULL_WEEK} h weeks only)", y=1.03)
    fig.tight_layout()
    out = FIG / "fig_soc_seasons.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cum_cashflow() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), sharey=True)
    palette = {"hmsd": "#5ad8a6", "pso": "#e8684a", "linprog": "#f6bd16"}
    for ax, season in zip(axes, SEASONS):
        for method, color in palette.items():
            df = load_full_week_traj(season, method)
            if df is None or "rt_economic_cashflow_delta" not in df.columns:
                continue
            cum = pd.to_numeric(df["rt_economic_cashflow_delta"], errors="coerce").fillna(0).cumsum()
            scale = 1e6 if cum.abs().max() > 1e5 else 1.0
            ax.plot(cum.to_numpy() / scale, label=METHOD_LABEL.get(method, method), color=color, lw=1.4)
        ax.set_title(season.capitalize())
        ax.set_xlabel("Hour")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, FULL_WEEK - 1)
    axes[0].set_ylabel("Cumulative cash-flow (×10⁶ CNY)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Cumulative market cash-flow (full {FULL_WEEK} h only)", y=1.02)
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
    methods = ("hmsd", "pso", "linprog")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
    x = np.arange(len(comps))
    width = 0.22
    for ax, season in zip(axes, SEASONS):
        for k, method in enumerate(methods):
            df = load_full_week_traj(season, method)
            if df is None:
                continue
            vals = []
            for col, _ in comps:
                if col not in df.columns:
                    vals.append(np.nan)
                else:
                    vals.append(float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum()) / 1e6)
            ax.bar(x + (k - 1) * width, vals, width=width, label=METHOD_LABEL.get(method, method))
        ax.set_xticks(x)
        ax.set_xticklabels([c[1] for c in comps], rotation=20)
        ax.set_title(season.capitalize())
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Weekly sum (×10⁶ CNY, 168 h)")
    axes[1].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle(f"Cost / revenue components (full {FULL_WEEK} h only)", y=1.02)
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
    p = plot_balance_fullweek("transition", ("hmsd", "linprog"), "fig_balance_transition.png")
    if p:
        outs.append(p)
        print("wrote", p)
    if load_full_week_traj("winter", "pso") is not None:
        p = plot_balance_fullweek("winter", ("pso", "linprog"), "fig_balance_winter.png")
        if p:
            outs.append(p)
            print("wrote", p)
    else:
        print("[skip] winter PSO has no 168 h eval.csv")
    outs.append(plot_soc_matrix())
    print("wrote", outs[-1])
    print("done", len(outs), "figures")


if __name__ == "__main__":
    main()
