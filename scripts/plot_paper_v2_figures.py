#!/usr/bin/env python
"""Schematic and (optional) result figures for the PC-HybridTD3 paper.

Default: figures that do not need the new seasonal_v1 traces.
Pass --with-results to fill horizon / mode-hour plots when eval files exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "Paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
TRAJ = ROOT / "runs" / "seasonal_v1"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

DIS_LO, DIS_HI = -1.0, -0.33
CHG_LO, CHG_HI = 0.86, 1.0
C_DIS = "#0072B2"
C_IDLE = "#999999"
C_CHG = "#D55E00"
C_SAC = "#CC79A7"
C_TD3 = "#009E73"


def _save(fig, name: str) -> Path:
    png = FIG / f"{name}.png"
    fig.savefig(png)
    try:
        fig.savefig(FIG / f"{name}.pdf")
    except Exception:
        pass
    plt.close(fig)
    print("wrote", png)
    return png


def plot_caes_legal() -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.6), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    for ax, (dis, idle_w, chg, subtitle) in zip(
        axes,
        [
            ((DIS_LO, DIS_HI), 0.06, (CHG_LO, CHG_HI), r"(a) Device envelope $\mathcal{A}_C$"),
            ((-0.72, -0.40), 0.06, (0.90, 1.00), r"(b) State-dependent $\mathcal{M}_k(s)\subseteq\mathcal{A}_C$"),
        ],
    ):
        ax.set_xlim(-1.22, 1.22)
        ax.set_ylim(-0.55, 0.72)
        ax.axhline(0, color="#222", lw=1.0)
        ax.add_patch(Rectangle((dis[0], -0.10), dis[1] - dis[0], 0.20, color=C_DIS, alpha=0.88, zorder=3))
        ax.add_patch(Rectangle((-idle_w / 2, -0.10), idle_w, 0.20, color=C_IDLE, alpha=0.95, zorder=3))
        ax.add_patch(Rectangle((chg[0], -0.10), chg[1] - chg[0], 0.20, color=C_CHG, alpha=0.88, zorder=3))
        ax.annotate(
            "",
            xy=(chg[0] - 0.04, 0.0),
            xytext=(dis[1] + 0.04, 0.0),
            arrowprops=dict(arrowstyle="<->", color="#888", lw=0.9),
        )
        ax.text(0.12, 0.28, "infeasible gap", ha="center", va="bottom", fontsize=8, color="#666")
        ax.text((dis[0] + dis[1]) / 2, -0.34, "discharge", ha="center", fontsize=8, color=C_DIS)
        ax.text(0.0, -0.34, "idle", ha="center", fontsize=8, color="#555")
        ax.text((chg[0] + chg[1]) / 2, -0.34, "charge", ha="center", fontsize=8, color=C_CHG)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(subtitle, loc="left", fontsize=9, pad=2)
    axes[1].set_xlabel(r"CAES command $u_{\mathrm{caes}}$ (charge $+$, discharge $-$)")
    fig.tight_layout(h_pad=0.35)
    return _save(fig, "fig_caes_legal")


def _decode_static(z: np.ndarray) -> np.ndarray:
    """Static-band hybrid: mode from sign of z, magnitude on the device envelope."""
    u = np.zeros_like(z)
    mag = np.clip(np.abs(z), 0.0, 1.0)
    dis = z < 0
    chg = z > 0
    u[dis] = DIS_LO + mag[dis] * (DIS_HI - DIS_LO)
    u[chg] = CHG_LO + mag[chg] * (CHG_HI - CHG_LO)
    return u


def _decode_dynamic(z: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mag = np.clip((z + 1.0) / 2.0, 0.0, 1.0)
    return lo + mag * (hi - lo)


def plot_action_rep() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharey=True)
    z = np.linspace(-1.0, 1.0, 800)

    ax = axes[0]
    u_proj = np.where((z >= DIS_LO) & (z <= DIS_HI), z, np.where((z >= CHG_LO) & (z <= CHG_HI), z, 0.0))
    ax.fill_betweenx([-1.2, 1.2], DIS_LO, DIS_HI, color=C_DIS, alpha=0.10, zorder=0)
    ax.fill_betweenx([-1.2, 1.2], CHG_LO, CHG_HI, color=C_CHG, alpha=0.10, zorder=0)
    ax.plot(z, u_proj, color="#222", lw=1.5)
    ax.axhline(0, color="#bbb", lw=0.5)
    ax.set_title("(a) Projection TD3", loc="left", fontsize=9)
    ax.set_xlabel(r"raw $z_{\mathrm{caes}}$")
    ax.set_ylabel(r"$u_{\mathrm{caes}}$")
    ax.text(0.0, 0.42, "gap mapped to idle", fontsize=7.5, color="#555", ha="center")

    ax = axes[1]
    u_static = _decode_static(z)
    ax.fill_betweenx([-1.2, 1.2], DIS_LO, DIS_HI, color=C_DIS, alpha=0.10, zorder=0)
    ax.fill_betweenx([-1.2, 1.2], CHG_LO, CHG_HI, color=C_CHG, alpha=0.10, zorder=0)
    ax.plot(z[z <= 0], u_static[z <= 0], color=C_DIS, lw=1.5, label="discharge")
    ax.plot(z[z >= 0], u_static[z >= 0], color=C_CHG, lw=1.5, label="charge")
    ax.axhline(0, color="#bbb", lw=0.5)
    ax.set_title("(b) Static-band hybrid", loc="left", fontsize=9)
    ax.set_xlabel(r"mode latent $z$")
    ax.legend(loc="lower right", fontsize=7)

    ax = axes[2]
    lo, hi = -0.72, -0.40
    mag = np.linspace(0.0, 1.0, 800)
    u_dyn = lo + mag * (hi - lo)
    ax.axhspan(lo, hi, color=C_DIS, alpha=0.16, zorder=0)
    ax.plot(mag, u_dyn, color="#222", lw=1.5)
    ax.axhline(0, color="#bbb", lw=0.5)
    ax.set_title(r"(c) PC-HybridTD3 $\mathcal{M}_k(s)$", loc="left", fontsize=9)
    ax.set_xlabel(r"magnitude $z\in[0,1]$")
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks([0, 0.5, 1])
    ax.text(0.50, 0.15, "current discharge interval", fontsize=7.5, color="#555", ha="center")

    for ax in axes[:2]:
        ax.set_xlim(-1.05, 1.05)
        ax.set_xticks([-1, 0, 1])
    for ax in axes:
        ax.set_ylim(-1.15, 1.15)
        ax.set_yticks([-1, 0, 1])
    fig.tight_layout(w_pad=0.8)
    return _save(fig, "fig_action_rep")


def plot_horizon(root: Path = TRAJ) -> Path | None:
    seasons = ("winter", "transition", "summer")
    methods = ("sac", "td3", "pso", "linprog")
    labels = {"sac": "Hybrid SAC", "td3": "Hybrid TD3", "pso": "PSO", "linprog": "LP"}
    hours = {m: [] for m in methods}
    found = False
    for season in seasons:
        for m in methods:
            p = root / season / f"{m}_s0" / "train_result.json"
            h = np.nan
            if p.is_file():
                found = True
                d = json.loads(p.read_text(encoding="utf-8"))
                ev = d.get("eval") or {}
                h = float(ev.get("n_steps") or ev.get("steps") or d.get("n_steps") or np.nan)
                if (d.get("status") or "").lower() in ("eval_failed", "failed"):
                    h = 0.0
            hours[m].append(h)
    if not found:
        print("skip fig_horizon (no train_result.json)")
        return None
    x = np.arange(len(seasons))
    width = 0.18
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    colors = {"sac": C_SAC, "td3": C_TD3, "pso": "#E69F00", "linprog": "#56B4E9"}
    for i, m in enumerate(methods):
        ax.bar(x + (i - 1.5) * width, hours[m], width=width, label=labels[m], color=colors[m])
    ax.axhline(168, ls="--", color="#444", lw=0.8, label="full week")
    ax.set_xticks(x)
    ax.set_xticklabels([s.title() for s in seasons])
    ax.set_ylabel("Held-out hours completed")
    ax.set_ylim(0, 190)
    ax.set_title("Executability: hours closed under GiveSafe, no fallback")
    ax.legend(ncol=3, loc="upper right")
    return _save(fig, "fig_horizon")


def plot_mode_hours(root: Path = TRAJ) -> Path | None:
    seasons = ("winter", "transition", "summer")
    rows = []
    for season in seasons:
        csv = root / season / "sac_s0" / "trajectories" / "eval.csv"
        if not csv.is_file():
            continue
        import pandas as pd

        df = pd.read_csv(csv)
        col = None
        for c in df.columns:
            if "u_caes" in c.lower() or c.lower() == "caes_u" or "caes" in c.lower() and "u" in c.lower():
                col = c
                break
        if col is None:
            continue
        u = pd.to_numeric(df[col], errors="coerce").to_numpy()
        n = max(len(u), 1)
        dis = float(np.mean(u < -0.05) * 168)
        chg = float(np.mean(u > 0.05) * 168)
        idle = 168.0 - dis - chg
        rows.append((season, dis, idle, chg))
    if not rows:
        print("skip fig_mode_hours (no eval.csv)")
        return None
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    x = np.arange(len(rows))
    dis = [r[1] for r in rows]
    idle = [r[2] for r in rows]
    chg = [r[3] for r in rows]
    ax.bar(x, dis, color=C_DIS, label="discharge")
    ax.bar(x, idle, bottom=dis, color=C_IDLE, label="idle")
    ax.bar(x, chg, bottom=np.array(dis) + np.array(idle), color=C_CHG, label="charge")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0].title() for r in rows])
    ax.set_ylabel("Hours in held-out week")
    ax.set_title("Hybrid SAC: CAES mode occupancy")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 190)
    return _save(fig, "fig_mode_hours")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-results", action="store_true")
    args = ap.parse_args()
    plot_caes_legal()
    plot_action_rep()
    # keep old filename so leftover tex still compiles
    src = FIG / "fig_caes_legal.png"
    dst = FIG / "fig_caes_feasible_set.png"
    if src.is_file():
        dst.write_bytes(src.read_bytes())
        pdf = FIG / "fig_caes_legal.pdf"
        if pdf.is_file():
            (FIG / "fig_caes_feasible_set.pdf").write_bytes(pdf.read_bytes())
    if args.with_results:
        plot_horizon()
        plot_mode_hours()


if __name__ == "__main__":
    main()
