#!/usr/bin/env python
"""Schematic and (optional) result figures for the FMU hybrid-SAC paper.

Default: figures that do not need the new seasonal_v1 traces.
Pass --with-results to fill horizon / mode-hour plots when eval files exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
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
C_BOX = "#F4F7FB"
C_LINE = "#333333"


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


def _box(ax, xy, w, h, text, *, fc=C_BOX, ec=C_LINE, fs=8, lw=1.0, weight="normal"):
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight=weight)


def _arrow(ax, p0, p1, **kw):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=10,
            lw=1.0,
            color=kw.get("color", C_LINE),
            connectionstyle=kw.get("connectionstyle", "arc3,rad=0"),
        )
    )


def plot_caes_legal() -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.55, 0.85)
    ax.axhline(0, color="#222", lw=1.1)
    ax.plot([-1.25, 1.25], [0, 0], color="#222", lw=1.1)
    ax.add_patch(Rectangle((DIS_LO, -0.08), DIS_HI - DIS_LO, 0.16, color=C_DIS, alpha=0.85, zorder=3))
    ax.add_patch(Rectangle((-0.03, -0.08), 0.06, 0.16, color=C_IDLE, alpha=0.95, zorder=3))
    ax.add_patch(Rectangle((CHG_LO, -0.08), CHG_HI - CHG_LO, 0.16, color=C_CHG, alpha=0.85, zorder=3))
    ax.annotate(
        "",
        xy=(0.42, 0.0),
        xytext=(-0.18, 0.0),
        arrowprops=dict(arrowstyle="<->", color="#888", lw=1.0),
    )
    ax.text(0.12, 0.28, "illegal gap\n(projection → idle)", ha="center", va="bottom", fontsize=8, color="#666")
    ax.text((DIS_LO + DIS_HI) / 2, -0.32, r"discharge $[-1,-0.33]$", ha="center", fontsize=8, color=C_DIS)
    ax.text(0.0, -0.32, r"idle $\{0\}$", ha="center", fontsize=8, color="#555")
    ax.text((CHG_LO + CHG_HI) / 2, -0.32, r"charge $[0.86,1]$", ha="center", fontsize=8, color=C_CHG)
    ax.plot([DIS_LO, DIS_HI, 0.0, CHG_LO, CHG_HI], [0, 0, 0, 0, 0], "k|", ms=10, zorder=4)
    ax.set_yticks([])
    ax.set_xlabel(r"CAES command $u_{\mathrm{caes}}$ (charge $+$, discharge $-$)")
    ax.set_title("Disconnected legal set (state-dependent mask may drop a band)")
    ax.spines["left"].set_visible(False)
    return _save(fig, "fig_caes_legal")


def plot_action_rep() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.55))
    z = np.linspace(-1.0, 1.0, 700)

    ax = axes[0]
    u_proj = np.where((z >= DIS_LO) & (z <= DIS_HI), z, np.where((z >= CHG_LO) & (z <= CHG_HI), z, 0.0))
    ax.fill_between([-1, DIS_HI], -1.15, 1.15, color=C_DIS, alpha=0.08)
    ax.fill_between([CHG_LO, 1], -1.15, 1.15, color=C_CHG, alpha=0.08)
    ax.plot(z, u_proj, color="#222", lw=1.6)
    ax.axhline(0, color="#bbb", lw=0.6)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel(r"raw $z_{\mathrm{caes}}$")
    ax.set_ylabel(r"$u_{\mathrm{caes}}$")
    ax.set_title("(i) Scalar projection")
    ax.text(0.0, 0.55, "gap → idle", fontsize=8, color="#666", ha="center")

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    ax.set_title("(ii) Fixed-band hybrid")
    _box(ax, (0.4, 4.15), 2.8, 1.45, r"mode logits" + "\n" + r"$k\in\{-,0,+\}$", fc="#E8F4FC", ec=C_DIS)
    _box(ax, (3.6, 4.15), 2.8, 1.45, r"mag on static" + "\n" + r"device envelope", fc="#FDEFE2", ec=C_CHG)
    _box(ax, (6.8, 4.15), 2.8, 1.45, "clamp after" + "\nsampling", fc="#F3F3F3", ec="#888")
    _arrow(ax, (3.2, 4.88), (3.6, 4.88))
    _arrow(ax, (6.4, 4.88), (6.8, 4.88))
    ax.text(5.0, 2.55, "mask $k$; magnitude not on $\\mathcal{M}_k(s)$", ha="center", fontsize=8, color="#555")
    ax.add_patch(Rectangle((1.1, 1.15), 2.2, 0.45, color=C_DIS, alpha=0.85))
    ax.add_patch(Rectangle((4.55, 1.15), 0.7, 0.45, color=C_IDLE, alpha=0.95))
    ax.add_patch(Rectangle((6.5, 1.15), 2.2, 0.45, color=C_CHG, alpha=0.85))
    ax.text(2.2, 1.75, "discharge", color=C_DIS, fontsize=8, ha="center")
    ax.text(4.9, 1.75, "idle", color="#555", fontsize=8, ha="center")
    ax.text(7.6, 1.75, "charge", color=C_CHG, fontsize=8, ha="center")

    ax = axes[2]
    lo, hi = -0.72, -0.40
    y = 1.0 / (1.0 + np.exp(-3.2 * z))
    u_dyn = lo + y * (hi - lo)
    ax.fill_between([lo, hi], -1.15, 1.15, color=C_DIS, alpha=0.16)
    ax.plot(z, u_dyn, color="#222", lw=1.6)
    ax.axhline(0, color="#bbb", lw=0.6)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel(r"latent $z_k$")
    ax.set_title(r"(iii) FS-HSAC $\mathcal{M}_k(s)$")
    ax.text(0.15, -0.55, r"affine $+$ Jacobian", fontsize=8, color="#666", ha="center")

    fig.suptitle("CAES action: projection vs fixed band vs feasible support", y=1.04, fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig_action_rep")


def plot_algorithm() -> Path:
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    ax.text(6.1, 7.05, "Hybrid-action closed loop on the Sysplorer FMU", ha="center", fontsize=12, fontweight="bold")

    _box(ax, (0.25, 4.85), 2.7, 1.65, "State $s_t$\nSoC, thermo, power,\nTOU features", fc="#EEF3F8", ec="#4C6A87", fs=8)
    _box(ax, (3.45, 4.75), 3.3, 1.85, "Hybrid SAC actor\n$u_{tp}$, $u_{bat}$\nCAES $(k,m)$", fc="#F8EEF4", ec=C_SAC, fs=8, weight="bold")
    _box(ax, (7.2, 4.85), 2.55, 1.65, r"Legal map $\Phi$" + "\n" + r"$u=u(k,m)$" + "\nmask $\\mathcal{F}(s)$", fc="#EEF7F2", ec=C_TD3, fs=8)
    _box(ax, (10.05, 4.85), 1.95, 1.65, "Twin Q\n$+\\,\\alpha$", fc="#F8EEF4", ec=C_SAC, fs=8)

    _box(ax, (3.45, 2.85), 6.3, 1.35, "GiveSafe (adopted): reject $a\\notin\\mathcal{F}(s)$; fallback off\nrejected transitions still enter replay", fc="#FFF8E8", ec="#C47B00", fs=8)

    _box(ax, (0.25, 0.45), 5.7, 1.75, "Sysplorer FMU (1 h step)\nthermal–battery–CAES DAE", fc="#F3F3F3", ec="#333", fs=8, weight="bold")
    _box(ax, (6.35, 0.45), 5.65, 1.75, r"Reward $r^{\mathrm{ext}}\propto\Delta J^{\mathrm{gen}}$" + "\nTOU cash, ETS CO$_2$, spill, wear", fc="#F3F3F3", ec="#333", fs=8)

    _arrow(ax, (2.95, 5.65), (3.45, 5.65))
    _arrow(ax, (6.75, 5.65), (7.2, 5.65))
    _arrow(ax, (8.45, 4.85), (7.2, 4.2))
    _arrow(ax, (6.6, 2.85), (4.0, 2.2))
    _arrow(ax, (3.1, 0.45), (3.1, 0.15))
    ax.annotate(
        "",
        xy=(1.55, 4.85),
        xytext=(1.55, 2.2),
        arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.05),
    )
    ax.text(0.35, 3.45, r"next $s$", fontsize=7.5, color="#555")
    ax.annotate(
        "",
        xy=(8.5, 2.2),
        xytext=(4.8, 2.2),
        arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.05),
    )
    ax.text(9.7, 3.55, "update", fontsize=7, color=C_SAC)
    _arrow(ax, (10.9, 4.85), (10.9, 2.2), connectionstyle="arc3,rad=0.0")
    return _save(fig, "fig_algorithm")


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
    plot_algorithm()
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
