#!/usr/bin/env python
"""Plot soft-shell full-week dispatch figures (appendix / contrast; not hard-protocol main tables).

Reads RL trajectories from runs/seasonal_v1_soft_shell and linprog from seasonal_v1.
Only draws methods that finished 168 h under soft shell (or hard linprog).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SOFT = ROOT / "runs" / "seasonal_v1_soft_shell"
HARD = ROOT / "runs" / "seasonal_v1"
FIG = ROOT / "Paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

FULL_WEEK = 168
METHOD_LABEL = {
    "hmsd": "HMSD",
    "td3": "TD3",
    "sac": "SAC",
    "linprog": "linprog MPC",
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
    return np.clip(-_mw(series), 0.0, None)


def load_traj(season: str, method: str) -> pd.DataFrame | None:
    if method == "linprog":
        p = HARD / season / f"{method}_s0" / "trajectories" / "eval.csv"
    else:
        p = SOFT / season / f"{method}_s0" / "trajectories" / "eval.csv"
    if not p.is_file():
        return None
    return pd.read_csv(p)


def load_full(season: str, method: str) -> pd.DataFrame | None:
    df = load_traj(season, method)
    if df is None or len(df) < FULL_WEEK:
        return None
    # Prefer physically valid rows when soft-shell pads failures
    if "transition_valid" in df.columns:
        valid = df[df["transition_valid"].astype(str).str.lower().isin(("true", "1")) | (df["transition_valid"] == True)]
        if len(valid) >= FULL_WEEK:
            return valid.iloc[:FULL_WEEK].copy()
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


def plot_season(season: str, methods: tuple[str, ...]) -> Path | None:
    frames = [(m, load_full(season, m)) for m in methods]
    frames = [(m, df) for m, df in frames if df is not None]
    if not frames:
        print(f"[skip] no full-week soft-shell traj for {season}")
        return None
    fig, axes = plt.subplots(len(frames), 1, figsize=(10.5, 2.7 * len(frames) + 0.6), sharex=True)
    if len(frames) == 1:
        axes = [axes]
    for i, (method, df) in enumerate(frames):
        axes[i].set_title(METHOD_LABEL.get(method, method), loc="left", fontsize=11)
        _draw_balance(axes[i], df, legend=(i == 0))
    axes[-1].set_xlabel("Hour in weekly episode")
    fig.suptitle(
        f"Soft-shell electric balance — {season} (full {FULL_WEEK} h; appendix contrast)",
        y=0.995,
        fontsize=12,
    )
    fig.tight_layout()
    out = FIG / f"fig_balance_soft_shell_{season}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def plot_soc_winter() -> Path | None:
    cells = (("winter", "hmsd"), ("winter", "td3"), ("winter", "sac"), ("winter", "linprog"))
    frames = [(s, m, load_full(s, m)) for s, m in cells]
    frames = [(s, m, df) for s, m, df in frames if df is not None]
    if not frames:
        return None
    fig, axes = plt.subplots(1, len(frames), figsize=(3.0 * len(frames) + 0.5, 3.2), sharey=True)
    if len(frames) == 1:
        axes = [axes]
    for ax, (season, method, df) in zip(axes, frames):
        ax.set_title(METHOD_LABEL.get(method, method), fontsize=10)
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
    fig.suptitle("Soft-shell winter SoC (full 168 h)", y=1.03)
    fig.tight_layout()
    out = FIG / "fig_soc_soft_shell_winter.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def write_hours_table() -> Path:
    rows = []
    for season in ("winter", "transition", "summer"):
        for method in ("hmsd", "td3", "sac"):
            p = SOFT / season / f"{method}_s0" / "soft_shell_eval.json"
            if not p.is_file():
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            k = d.get("kpi") or {}
            rows.append(
                {
                    "season": season,
                    "method": method,
                    "status": d.get("status"),
                    "valid_steps": k.get("valid_steps"),
                    "soft_shell_count": k.get("soft_shell_count"),
                    "episode_reward": k.get("episode_reward"),
                }
            )
    out = ROOT / "docs" / "tab_soft_shell_seed0.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ROOT / "docs" / "tab_soft_shell_seed0.md"
    lines = [
        "# Soft-shell held-out eval (seed 0)",
        "",
        "Eval-only shell on frozen seasonal_v1 weights. Not the hard-protocol main table.",
        "",
        "| season | method | status | valid_steps | soft_shell_count | episode_reward |",
        "|--------|--------|--------|-------------|------------------|----------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['season']} | {r['method']} | {r['status']} | {r['valid_steps']} | "
            f"{r['soft_shell_count']} | {r['episode_reward']} |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out, md)
    return out


def main() -> None:
    write_hours_table()
    # Winter: all three RL + linprog full under soft shell
    plot_season("winter", ("hmsd", "td3", "sac", "linprog"))
    # Transition / summer: only methods that finished 168 h
    plot_season("transition", ("hmsd", "linprog"))
    plot_season("summer", ("hmsd", "linprog"))
    plot_soc_winter()


if __name__ == "__main__":
    main()
