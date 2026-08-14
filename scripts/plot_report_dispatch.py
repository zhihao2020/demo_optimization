#!/usr/bin/env python
"""PPT-oriented figures for the HMSD training-result briefing.

KPI numbers come from docs/_seed0_raw.json + the seed-0 memo tables.
Dispatch traces come from runs/paper_dispatch_traj/ (typical-week FMU
exports). Generation is FMU-negative; this script converts to MW sources.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
TRAJ = ROOT / "runs" / "paper_dispatch_traj"
SEED0 = ROOT / "docs" / "_seed0_raw.json"
OUT = ROOT / "picture" / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.dpi": 160,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linestyle": "-",
    "lines.linewidth": 1.8,
})

# Okabe-Ito + ocean dusk
COL = {
    "thermal": "#D55E00",
    "wind": "#009E73",
    "pv": "#E69F00",
    "battery": "#0072B2",
    "caes": "#CC79A7",
    "grid": "#7F7F7F",
    "load": "#000000",
    "hmsd": "#E76F51",
    "td3": "#264653",
    "sac": "#2A9D8F",
    "b0": "#B0BEC5",
    "b1": "#90A4AE",
    "lp": "#8D6E63",
    "pso": "#F4A261",
    "ink": "#1B1B1B",
}

METHOD_COLOR = {
    "hmsd": COL["hmsd"],
    "td3": COL["td3"],
    "sac": COL["sac"],
    "b0": COL["b0"],
    "b1": COL["b1"],
    "linprog": COL["lp"],
    "pso": COL["pso"],
}
METHOD_CN = {
    "hmsd": "分层调度",
    "td3": "单层强化学习",
    "sac": "对照强化学习",
    "b0": "厂站规则",
    "b1": "峰谷规则",
    "linprog": "滚动规划",
    "pso": "粒子群",
}
SEASON_CN = {"winter": "冬季", "transition": "过渡季", "summer": "夏季"}

# B0 / B1 are in the seed-0 memo but not in _seed0_raw.json
SEED0_B0B1 = {
    ("winter", "b0"): {"R": 54.28, "Jgen": 6.273e6, "CF": 8.063e6, "SOC": True, "SOC_l1": None, "uns": 0.0, "curt": 0.0, "bat": 0.0, "caes": 143.0, "th": 25200.0, "carbon": None, "deg": None, "buy_mwh": None, "sell_mwh": None},
    ("winter", "b1"): {"R": 35.29, "Jgen": 3.818e6, "CF": 5.964e6, "SOC": True, "SOC_l1": None, "uns": 0.0, "curt": 0.0, "bat": 4395.0, "caes": 11811.0, "th": 24807.0, "carbon": None, "deg": None, "buy_mwh": None, "sell_mwh": None},
    ("transition", "b0"): {"R": 8.40, "Jgen": -8.868e5, "CF": 1.396e6, "SOC": True, "SOC_l1": None, "uns": 0.0, "curt": 0.0, "bat": 0.0, "caes": 2530.0, "th": 25200.0, "carbon": None, "deg": None, "buy_mwh": None, "sell_mwh": None},
    ("transition", "b1"): {"R": 3.38, "Jgen": -1.127e6, "CF": 1.128e6, "SOC": True, "SOC_l1": None, "uns": 0.0, "curt": 0.0, "bat": 2337.0, "caes": 6011.0, "th": 24980.0, "carbon": None, "deg": None, "buy_mwh": None, "sell_mwh": None},
    ("summer", "b0"): {"R": 22.06, "Jgen": 1.208e6, "CF": 3.190e6, "SOC": True, "SOC_l1": None, "uns": 0.0, "curt": 0.0, "bat": 0.0, "caes": 723.0, "th": 25200.0, "carbon": None, "deg": None, "buy_mwh": None, "sell_mwh": None},
    ("summer", "b1"): {"R": 12.98, "Jgen": 3.515e5, "CF": 2.487e6, "SOC": True, "SOC_l1": None, "uns": 0.0, "curt": 0.0, "bat": 2337.0, "caes": 6893.0, "th": 24935.0, "carbon": None, "deg": None, "buy_mwh": None, "sell_mwh": None},
}


def load_seed0() -> dict[tuple[str, str], dict]:
    rows = json.loads(SEED0.read_text(encoding="utf-8"))
    out = dict(SEED0_B0B1)
    for r in rows:
        out[(r["season"], r["method"])] = r
    return out


def mw_signed(series: pd.Series) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    peak = np.nanmax(np.abs(x[np.isfinite(x)])) if np.any(np.isfinite(x)) else 0.0
    if peak > 1e4:
        x = x / 1e6
    return x


def load_traj(season: str, method: str) -> pd.DataFrame | None:
    p = TRAJ / f"{season}_{method}.csv"
    if not p.is_file():
        return None
    return pd.read_csv(p)


def sources(df: pd.DataFrame) -> dict[str, np.ndarray]:
    th = -mw_signed(df["obs_p_thermal"])
    wind = -mw_signed(df["obs_p_wind_actual"])
    pv = -mw_signed(df["obs_p_pv_actual"])
    bat = mw_signed(df["obs_p_battery"])
    caes = mw_signed(df["obs_p_caes"])
    grid = mw_signed(df["obs_p_grid"])
    load = mw_signed(df["obs_p_load_actual"])
    wind_av = -mw_signed(df["obs_p_wind_available"]) if "obs_p_wind_available" in df else wind
    pv_av = -mw_signed(df["obs_p_pv_available"]) if "obs_p_pv_available" in df else pv
    curt = np.abs(mw_signed(df["obs_p_curtailment"])) if "obs_p_curtailment" in df else np.zeros(len(df))
    return {
        "thermal": np.clip(th, 0, None),
        "wind": np.clip(wind, 0, None),
        "pv": np.clip(pv, 0, None),
        "bat_dis": np.clip(-bat, 0, None),
        "bat_ch": np.clip(bat, 0, None),
        "caes_dis": np.clip(-caes, 0, None),
        "caes_ch": np.clip(caes, 0, None),
        "sell": np.clip(-grid, 0, None),
        "buy": np.clip(grid, 0, None),
        "load": np.clip(load, 0, None),
        "wind_av": np.clip(wind_av, 0, None),
        "pv_av": np.clip(pv_av, 0, None),
        "curt": curt,
        "bat_soc": pd.to_numeric(df["obs_battery_soc"], errors="coerce").to_numpy(float),
        "gas_soc": pd.to_numeric(df["obs_caes_gas_soc"], errors="coerce").to_numpy(float),
        "price_buy": pd.to_numeric(df.get("rt_market_buy_yuan_per_kwh", 0), errors="coerce").to_numpy(float),
    }


def plot_kpi_jgen(data: dict) -> Path:
    seasons = ["winter", "transition", "summer"]
    methods = ["b0", "linprog", "pso", "sac", "td3", "hmsd"]
    x = np.arange(len(seasons))
    n = len(methods)
    width = 0.13
    fig, ax = plt.subplots(figsize=(11.2, 4.4))
    for i, m in enumerate(methods):
        vals = []
        for s in seasons:
            row = data.get((s, m), {})
            vals.append((row.get("Jgen") or 0.0) / 1e6)
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, vals, width * 0.92,
            label=METHOD_CN[m], color=METHOD_COLOR[m],
            edgecolor="white", linewidth=0.4, zorder=3,
        )
        for bar, v in zip(bars, vals):
            if abs(v) >= 0.4:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.25 if v >= 0 else -0.55),
                    f"{v:.1f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=7.2, color="#333",
                )
    ax.axhline(0, color="#888", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([SEASON_CN[s] for s in seasons])
    ax.set_ylabel("综合收益（百万元 / 周）")
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=8.5)
    fig.tight_layout()
    out = OUT / "fig_kpi_jgen_bars.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_kpi_reward(data: dict) -> Path:
    seasons = ["winter", "transition", "summer"]
    methods = ["b0", "sac", "td3", "hmsd"]
    x = np.arange(len(seasons))
    n = len(methods)
    width = 0.18
    fig, ax = plt.subplots(figsize=(10.4, 4.3))
    for i, m in enumerate(methods):
        vals, socs = [], []
        for s in seasons:
            row = data.get((s, m), {})
            vals.append(float(row.get("R") or 0.0))
            socs.append(bool(row.get("SOC")))
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, vals, width * 0.9,
            label=METHOD_CN[m], color=METHOD_COLOR[m],
            edgecolor="white", linewidth=0.4, zorder=3,
        )
        for bar, v, ok in zip(bars, vals, socs):
            mark = "过门" if ok else "未过"
            color = "#2E7D32" if ok else "#C62828"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (3.2 if v >= 0 else -7.5),
                f"{v:.1f}\n{mark}", ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=7.4, color=color, linespacing=1.15,
            )
    ax.axhline(0, color="#888", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([SEASON_CN[s] for s in seasons])
    ax.set_ylabel("周评分")
    ax.set_ylim(-20, 145)
    ax.legend(ncol=4, loc="upper left")
    ax.set_title("考试周 · 周评分与周末库存是否过门")
    out = OUT / "fig_kpi_reward_soc.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_dispatch(season: str, method: str, hours: int | None = 72) -> Path:
    df = load_traj(season, method)
    if df is None:
        raise FileNotFoundError(season + method)
    src = sources(df)
    n = len(df) if hours is None else min(hours, len(df))
    t = np.arange(n)
    fig, axes = plt.subplots(2, 1, figsize=(11.4, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.55, 1.0]})
    ax = axes[0]
    stack = [
        src["thermal"][:n], src["wind"][:n], src["pv"][:n],
        src["bat_dis"][:n], src["caes_dis"][:n], src["buy"][:n],
    ]
    labels = ["火电", "风电", "光伏", "电池放电", "压空放电", "购电"]
    colors = [COL["thermal"], COL["wind"], COL["pv"], COL["battery"], COL["caes"], "#B0BEC5"]
    ax.stackplot(t, *stack, labels=labels, colors=colors, alpha=0.88)
    ax.plot(t, src["load"][:n], color=COL["load"], lw=1.6, label="负荷")
    ax.plot(t, src["sell"][:n], color=COL["grid"], lw=1.2, ls="--", label="售电")
    ax.set_ylabel("功率 / MW")
    title_m = "分层调度" if method == "ghtd3" else METHOD_CN.get(method, method)
    ax.set_title(f"{SEASON_CN[season]} · {title_m}  功率平衡（前 {n} 小时）")
    ax.legend(ncol=4, loc="upper right", fontsize=8)
    ax.set_ylim(0, None)

    ax2 = axes[1]
    ax2.plot(t, src["bat_soc"][:n], color=COL["battery"], lw=1.8, label="电池库存")
    ax2.plot(t, src["gas_soc"][:n], color=COL["caes"], lw=1.8, label="气库库存")
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_ylabel("荷电状态")
    ax2.set_xlabel("小时")
    ax2.legend(loc="upper right")
    if np.nanmax(src["price_buy"][:n]) > 0:
        ax3 = ax2.twinx()
        ax3.plot(t, src["price_buy"][:n], color="#888", lw=1.0, ls=":", alpha=0.85)
        ax3.set_ylabel("购电价  元/kWh", color="#666")
        ax3.tick_params(axis="y", labelcolor="#666")
        ax3.spines["top"].set_visible(False)
    fig.tight_layout()
    out = OUT / f"fig_dispatch_{season}_{method}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_dispatch_compare(season: str, hours: int = 72) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(11.4, 6.4), sharex=True)
    for ax, method, title in (
        (axes[0], "ghtd3", "分层调度"),
        (axes[1], "td3", "单层强化学习"),
    ):
        df = load_traj(season, method)
        if df is None:
            ax.text(0.5, 0.5, "无轨迹", ha="center", va="center")
            continue
        src = sources(df)
        n = min(hours, len(df))
        t = np.arange(n)
        ax.stackplot(
            t,
            src["thermal"][:n], src["wind"][:n], src["pv"][:n],
            src["bat_dis"][:n], src["caes_dis"][:n],
            labels=["火电", "风电", "光伏", "电池放电", "压空放电"],
            colors=[COL["thermal"], COL["wind"], COL["pv"], COL["battery"], COL["caes"]],
            alpha=0.88,
        )
        ax.plot(t, src["load"][:n], color="k", lw=1.4, label="负荷")
        ax.set_ylabel("MW")
        ax.set_title(f"{SEASON_CN[season]} · {title}", loc="left")
        if method == "ghtd3":
            ax.legend(ncol=6, loc="upper right", fontsize=8)
    axes[-1].set_xlabel("小时")
    fig.suptitle(f"{SEASON_CN[season]}调度对照（前 {hours} 小时，典型周轨迹）", y=1.01)
    fig.tight_layout()
    out = OUT / f"fig_dispatch_compare_{season}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_renewable_util() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.6), sharey=True)
    for ax, season in zip(axes, ("winter", "transition", "summer")):
        df = load_traj(season, "ghtd3")
        if df is None:
            continue
        src = sources(df)
        t = np.arange(len(df))
        ax.fill_between(t, 0, src["wind_av"], color=COL["wind"], alpha=0.18, label="风电可发")
        ax.plot(t, src["wind"], color=COL["wind"], lw=1.3, label="风电实发")
        ax.fill_between(t, 0, src["pv_av"], color=COL["pv"], alpha=0.18, label="光伏可发")
        ax.plot(t, src["pv"], color=COL["pv"], lw=1.3, label="光伏实发")
        ax.set_title(SEASON_CN[season])
        ax.set_xlabel("小时")
        curt_mwh = float(np.nansum(src["curt"]))
        ax.text(
            0.98, 0.95, f"弃电 {curt_mwh:.1e} MWh",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#555",
        )
    axes[0].set_ylabel("功率 / MW")
    axes[2].legend(loc="upper left", fontsize=8)
    fig.suptitle("分层调度典型周：风光可发与实发几乎重合（弃电约等于零）", y=1.03)
    fig.tight_layout()
    out = OUT / "fig_renewable_util.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_cost_split(data: dict) -> Path:
    seasons = ["winter", "summer"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    labels = ["经营现金", "碳成本", "电池磨损"]
    for ax, season in zip(axes, seasons):
        h = data[season, "hmsd"]
        t = data[season, "td3"]
        h_vals = [h["CF"] / 1e6, (h.get("carbon") or 0) / 1e6, (h.get("deg") or 0) / 1e6]
        t_vals = [t["CF"] / 1e6, (t.get("carbon") or 0) / 1e6, (t.get("deg") or 0) / 1e6]
        x = np.arange(3)
        w = 0.36
        b1 = ax.bar(x - w / 2, h_vals, w, label="分层调度", color=COL["hmsd"], edgecolor="white")
        b2 = ax.bar(x + w / 2, t_vals, w, label="单层强化学习", color=COL["td3"], edgecolor="white")
        for bars in (b1, b2):
            for bar in bars:
                hgt = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, hgt + 0.15,
                        f"{hgt:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(SEASON_CN[season])
        ax.set_ylabel("百万元 / 周")
        ax.legend(loc="upper right")
    fig.suptitle("成本结构：现金收入对碳与磨损", y=1.02)
    fig.tight_layout()
    out = OUT / "fig_cost_split.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_throughput(data: dict) -> Path:
    seasons = ["winter", "transition", "summer"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    for ax, key, ylabel, title in (
        (axes[0], "th", "火电发电量 / MWh", "火电"),
        (axes[1], "bat", "电池吞吐 / MWh", "电池吞吐"),
    ):
        x = np.arange(len(seasons))
        w = 0.22
        for i, m in enumerate(("b0", "td3", "hmsd")):
            vals = [float(data[s, m].get(key) or 0) for s in seasons]
            bars = ax.bar(x + (i - 1) * w, vals, w * 0.92, label=METHOD_CN[m],
                          color=METHOD_COLOR[m], edgecolor="white")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.015,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([SEASON_CN[s] for s in seasons])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
    fig.suptitle("运行结构：分层更敢动储能；夏天因此多烧火电", y=1.02)
    fig.tight_layout()
    out = OUT / "fig_throughput.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_architecture() -> Path:
    fig, ax = plt.subplots(figsize=(11.6, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def box(x, y, w, h, fc, title, lines, tc="#1B1B1B"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                           facecolor=fc, edgecolor="#CCCCCC", linewidth=0.8)
        ax.add_patch(p)
        ax.text(x + 0.16, y + h - 0.28, title, fontsize=11, fontweight="bold",
                color=tc, va="top")
        ax.text(x + 0.16, y + h - 0.58, "\n".join(lines), fontsize=9.2,
                color="#333", va="top", linespacing=1.35)

    def arrow(x1, y1, x2, y2, text=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.3),
        )
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, text,
                    fontsize=8, color="#555", ha="center")

    box(0.25, 3.55, 3.35, 2.15, "#E8F2EE", "厂站与题目",
        ["风光火 + 电池 + 压空", "分时购售电", "一小时下一道指令，一周算清",
         "最大化综合收益"])
    box(4.05, 3.55, 3.7, 2.15, "#E8EDF2", "上层  每 8 小时",
        ["给出电池、气库接下来怎么走", "管跨日库存，不管这一小时",
         "和下层算同一本账"])
    box(8.15, 3.55, 3.55, 2.15, "#F5EDE4", "下层  每小时",
        ["火电、电池、压空具体出力", "不合法指令进不了仿真",
         "压空只能充、停、放"])
    box(0.25, 0.35, 3.35, 2.55, "#F7F7F5", "综合收益",
        ["现金 − 碳 − 弃电/缺供 − 磨损", "碳价 80 元/吨",
         "弃电 300 / 缺供 1000 元/兆瓦时", "周末库存回到周一初值附近",
         "偏差不超过约 6%"])
    box(4.05, 0.35, 3.7, 2.55, "#F7F7F5", "一起比的方法",
        ["厂站规则 / 峰谷规则", "滚动规划",
         "粒子群", "单层 / 对照强化学习", "同一厂站、同一本账、同一考试周"])
    box(8.15, 0.35, 3.55, 2.55, "#F7F7F5", "本汇报怎么报",
        ["预先定好的一次，不事后挑", "冬第 5 周",
         "过渡第 18 周", "夏第 31 周", "每季约五千回合"])

    arrow(3.6, 4.6, 4.05, 4.6, "观测")
    arrow(7.75, 4.6, 8.15, 4.6, "库存意图")
    arrow(9.9, 3.55, 9.9, 2.95, "")
    ax.text(10.05, 3.2, "执行", fontsize=8, color="#555")
    ax.set_title("分层调度：上层管库存，下层管机组", fontsize=14, pad=8)
    out = OUT / "fig_architecture.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def plot_market_buysell(data: dict) -> Path:
    seasons = ["winter", "transition", "summer"]
    methods = ["linprog", "sac", "td3", "hmsd"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2), sharex=True)
    x = np.arange(len(seasons))
    n = len(methods)
    width = 0.18
    for ax, key, title in (
        (axes[0], "buy_mwh", "购电"),
        (axes[1], "sell_mwh", "售电"),
    ):
        for i, m in enumerate(methods):
            vals = [float(data[s, m].get(key) or 0.0) for s in seasons]
            offset = (i - n / 2 + 0.5) * width
            bars = ax.bar(
                x + offset, vals, width * 0.92,
                label=METHOD_CN[m], color=METHOD_COLOR[m],
                edgecolor="white", linewidth=0.4, zorder=3,
            )
            for bar, v in zip(bars, vals):
                if m in ("hmsd", "td3") and v >= 80:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals) * 0.012,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=7.2, color="#333",
                    )
        ax.set_xticks(x)
        ax.set_xticklabels([SEASON_CN[s] for s in seasons])
        ax.set_ylabel("能量 / MWh")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper left" if key == "buy_mwh" else "upper right")
    fig.suptitle("考试周购售电量", y=1.02)
    fig.tight_layout()
    out = OUT / "fig_market_buysell.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_cash_split(data: dict) -> Path:
    seasons = ["winter", "summer"]
    labels = ["经营现金", "电网结算", "火电现金", "碳成本", "电池磨损"]
    keys = ["CF", "grid_cf", "th_cash", "carbon", "deg"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    for ax, season in zip(axes, seasons):
        x = np.arange(len(labels))
        w = 0.36
        for i, m in enumerate(("hmsd", "td3")):
            vals = []
            for k in keys:
                raw = data[season, m].get(k)
                v = (raw or 0.0) / 1e6
                if k in ("carbon", "deg"):
                    v = -abs(v)
                vals.append(v)
            bars = ax.bar(
                x + (i - 0.5) * w, vals, w * 0.92,
                label=METHOD_CN[m], color=METHOD_COLOR[m],
                edgecolor="white", linewidth=0.4, zorder=3,
            )
            for bar, v in zip(bars, vals):
                if abs(v) >= 0.15:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        v + (0.18 if v >= 0 else -0.45),
                        f"{v:.1f}", ha="center",
                        va="bottom" if v >= 0 else "top",
                        fontsize=7.2, color="#333",
                    )
        ax.axhline(0, color="#888", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("百万元 / 周")
        ax.set_title(SEASON_CN[season])
        ax.legend(loc="upper right")
    fig.suptitle("账本拆细（上为正收入，下为支出）", y=1.02)
    fig.tight_layout()
    out = OUT / "fig_cash_split.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_reward_parts(data: dict) -> Path:
    seasons = ["winter", "transition", "summer"]
    parts = [("econR", "经济项"), ("shape", "库存塑造"), ("termB", "期末奖罚")]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    x = np.arange(len(seasons))
    w = 0.24
    colors = {"econR": "#2A9D8F", "shape": "#E9C46A", "termB": "#E76F51"}
    for ax, method, title in ((axes[0], "hmsd", "分层调度"), (axes[1], "td3", "单层强化学习")):
        bottom_pos = np.zeros(3)
        bottom_neg = np.zeros(3)
        for key, lab in parts:
            vals = np.array([float(data[s, method].get(key) or 0.0) for s in seasons])
            pos = np.clip(vals, 0, None)
            neg = np.clip(vals, None, 0)
            ax.bar(x, pos, 0.55, bottom=bottom_pos, label=lab, color=colors[key],
                   edgecolor="white", linewidth=0.4)
            ax.bar(x, neg, 0.55, bottom=bottom_neg, color=colors[key],
                   edgecolor="white", linewidth=0.4)
            bottom_pos = bottom_pos + pos
            bottom_neg = bottom_neg + neg
            for i, v in enumerate(vals):
                if abs(v) >= 3:
                    y = (bottom_pos[i] - pos[i] / 2) if v >= 0 else (bottom_neg[i] - neg[i] / 2)
                    ax.text(x[i], y, f"{v:.1f}", ha="center", va="center",
                            fontsize=7.5, color="#222")
        ax.axhline(0, color="#888", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([SEASON_CN[s] for s in seasons])
        ax.set_title(title)
        ax.set_ylabel("周评分拆分")
        if method == "hmsd":
            ax.legend(loc="upper right")
    fig.suptitle("周评分拆开看：经营账 / 库存提醒 / 周末奖罚", y=1.02)
    fig.tight_layout()
    out = OUT / "fig_reward_parts.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_reject_rates(data: dict) -> Path:
    seasons = ["winter", "transition", "summer"]
    methods = ["td3", "sac"]
    fig, ax = plt.subplots(figsize=(10.4, 4.1))
    x = np.arange(len(seasons))
    w = 0.32
    for i, m in enumerate(methods):
        vals = []
        for s in seasons:
            r = data[s, m].get("reject_rate")
            vals.append(100.0 * float(r) if r is not None else 0.0)
        bars = ax.bar(
            x + (i - 0.5) * w, vals, w * 0.9,
            label=METHOD_CN[m], color=METHOD_COLOR[m],
            edgecolor="white", linewidth=0.4, zorder=3,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9, color="#333",
            )
    ax.set_xticks(x)
    ax.set_xticklabels([SEASON_CN[s] for s in seasons])
    ax.set_ylabel("被安全层拦住的指令 / %")
    ax.set_ylim(0, 80)
    ax.legend(loc="upper right")
    ax.set_title("考试周：单层与对照的指令拦截比例")
    ax.text(
        0.01, -0.18,
        "分层调度这张考试周没有单独记下拦截次数，图上不画。",
        transform=ax.transAxes, fontsize=8.5, color="#555",
    )
    fig.tight_layout()
    out = OUT / "fig_reject_td3_sac.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    data = load_seed0()
    paths = [
        plot_architecture(),
        plot_kpi_jgen(data),
        plot_kpi_reward(data),
        plot_throughput(data),
        plot_cost_split(data),
        plot_renewable_util(),
        plot_dispatch("winter", "ghtd3", 72),
        plot_dispatch("summer", "ghtd3", 72),
        plot_dispatch("summer", "td3", 72),
        plot_dispatch_compare("winter", 72),
        plot_dispatch_compare("summer", 72),
        plot_market_buysell(data),
        plot_cash_split(data),
        plot_reward_parts(data),
        plot_reject_rates(data),
    ]
    for p in paths:
        print(p.relative_to(ROOT), p.stat().st_size)


if __name__ == "__main__":
    main()
