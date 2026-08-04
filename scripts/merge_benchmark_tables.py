#!/usr/bin/env python
"""Merge non-PSO + PSO benchmark JSON and write paper table markdown."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optimization.metrics import relative_to_baseline  # noqa: E402


def main() -> None:
    main_path = Path("runs/benchmark_full_3season_20260804/benchmark_table.json")
    pso_path = Path("runs/benchmark_pso_3season_20260804/benchmark_table.json")
    main = json.loads(main_path.read_text(encoding="utf-8"))
    pso = json.loads(pso_path.read_text(encoding="utf-8"))

    rows = list(main["rows"])
    b0_by_season = {
        r["season"]: r for r in main["rows"] if r.get("method") == "b0"
    }
    for r in pso["rows"]:
        if r.get("method") != "pso":
            continue
        if r.get("net_cashflow_j") is None and r.get("raw_total_cost") is not None:
            r["net_cashflow_j"] = -float(r["raw_total_cost"])
        b0 = b0_by_season.get(r["season"])
        if b0 is not None and r.get("net_cashflow_j") is not None:
            r["vs_b0"] = relative_to_baseline(r, b0)
        rows.append(r)

    order = ["b0", "b1", "lp", "pso", "hybrid", "ghtd3"]
    season_order = ["winter", "summer", "transition"]
    rows_sorted: list[dict] = []
    for s in season_order:
        for m in order:
            for r in rows:
                if r.get("season") == s and r.get("method") == m:
                    rows_sorted.append(r)

    labels = {
        "b0": "B0 Rule (baseline)",
        "b1": "B1 Price-aware rule",
        "lp": "M1 Rolling LP (relaxed)",
        "pso": "M2 PSO (parametric)",
        "hybrid": "M3 Hybrid-GiveSafe-TD3",
        "ghtd3": "M4 Safe Market-GHTD3 (ours)",
    }
    season_lab = {"winter": "Winter", "summer": "Summer", "transition": "Transition"}

    out = Path("runs/benchmark_merged_3season_pso_20260804")
    out.mkdir(parents=True, exist_ok=True)
    merged = {
        "note": (
            "B0=conservative rule (original FMU operation without smart optimization); "
            "B1=price rule; LP=relaxed rolling heuristic; "
            "PSO=6-dim parametric TOU policy, 15 iters x 10 particles; "
            "Hybrid/GHTD3=trained RL checkpoints evaluated closed-loop on FMU."
        ),
        "pso_settings": {
            "iters": 15,
            "pop": 10,
            "approx_fmu_steps_per_season": 15 * 10 * 168,
        },
        "rows": rows_sorted,
    }
    (out / "benchmark_merged.json").write_text(
        json.dumps(merged, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    def fnum(x, nd=2):
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return "—"

    def sci(x):
        try:
            return f"{float(x):.3e}"
        except Exception:
            return "—"

    # English paper Table
    en = [
        "# Table draft: Seasonal closed-loop FMU comparison",
        "",
        "**Setting.** Price-taker TOU (Shandong proxy); weekly horizon $T=168$ h; same FMU and start times.",
        "B0 is the *original model operation* baseline (high thermal, storage idle).",
        "All metrics are measured on the high-fidelity FMU trajectory.",
        "",
        "## Table I. Weekly net cash flow, energy SOC, storage and thermal generation",
        "",
        "| Season | Method | Net cash flow $J$ (CNY) | $\\Delta J$ vs B0 | Episode reward | Energy SOC pass | $E^{\\mathrm{th}}$ (MWh) | Bat thr. (MWh) | CAES thr. (MWh) | Curt. (MWh) |",
        "|--------|--------|--------------------------:|------------------:|---------------:|:---------------:|------------------------:|---------------:|----------------:|------------:|",
    ]
    for r in rows_sorted:
        vs = r.get("vs_b0") or {}
        dj = vs.get("delta_j_vs_b0")
        en.append(
            "| {s} | {m} | {j} | {dj} | {rew} | {soc} | {th} | {bt} | {ct} | {cu} |".format(
                s=season_lab.get(r["season"], r["season"]),
                m=labels.get(r["method"], r["method"]),
                j=sci(r.get("net_cashflow_j")),
                dj=("—" if dj is None else sci(dj)),
                rew=fnum(r.get("episode_reward"), 1),
                soc="Y" if r.get("terminal_soc_satisfied") else "N",
                th=fnum(r.get("thermal_mwh"), 0),
                bt=fnum(r.get("battery_throughput_mwh"), 0),
                ct=fnum(r.get("caes_throughput_mwh"), 0),
                cu=fnum(r.get("curtailment_mwh"), 2),
            )
        )

    en += [
        "",
        "## Table II. Relative improvement vs B0 (selected)",
        "",
        "| Season | Method | $\\Delta J$ / $J_{B0}$ | Thermal ratio | Storage throughput ratio |",
        "|--------|--------|------------------------:|--------------:|-------------------------:|",
    ]
    for r in rows_sorted:
        if r.get("method") == "b0":
            continue
        vs = r.get("vs_b0") or {}
        b0 = b0_by_season.get(r["season"]) if False else None
        b0 = next(x for x in rows_sorted if x["season"] == r["season"] and x["method"] == "b0")
        j0 = float(b0.get("net_cashflow_j") or 0.0)
        dj = vs.get("delta_j_vs_b0")
        rel = (float(dj) / j0) if (dj is not None and abs(j0) > 1) else None
        en.append(
            "| {s} | {m} | {rel} | {tr} | {sr} |".format(
                s=season_lab.get(r["season"], r["season"]),
                m=labels.get(r["method"], r["method"]),
                rel=("—" if rel is None else f"{100*rel:.1f}%"),
                tr=fnum(vs.get("thermal_ratio_vs_b0"), 2),
                sr=fnum(vs.get("storage_throughput_ratio_vs_b0"), 2),
            )
        )

    en += [
        "",
        "## Table III. Computational effort (order of magnitude)",
        "",
        "| Method | FMU steps per season (eval) | Notes |",
        "|--------|----------------------------:|-------|",
        "| B0 / B1 / LP / Hybrid / GHTD3 (eval) | 168 | Single closed-loop week |",
        "| PSO search | $\\approx 15\\times 10\\times 168 = 2.52\\times 10^4$ | Plus final re-eval; three seasons $\\times 3$ |",
        "| Hybrid/GHTD3 training (offline) | $10^4$–$10^5$ valid steps | Not counted in weekly eval wall time |",
        "",
        "### Discussion bullets (for paper)",
        "",
        "1. **RL methods dominate** weekly $J$ in all seasons: Hybrid and GHTD3 improve net cash flow by about "
        "$+1.0\\times 10^7$ CNY/week in winter and transition, and turn summer from near-zero/negative B0 cash flow to $+1.1\\times 10^7$.",
        "2. **Thermal generation** falls from 25200 MWh (B0 full load) to ~8600–10300 MWh under RL, indicating reduced fuel burn with storage–market coordination.",
        "3. **PSO** (low-dimensional parametric policy, limited budget) beats B0 in summer/transition cash flow but **lags RL** substantially; winter PSO under-ran thermal and failed energy SOC — consistent with under-parameterized black-box search on hybrid non-convex actions.",
        "4. **Relaxed LP/heuristic** is safer than early broken LP but does not match RL economic performance; full non-convex CAES MILP is left as future work / upper-bound discussion.",
        "5. **Curtailment** is ~0 MWh for all methods in these three weeks under current FMU boundary (report physical metric; economic penalty coefficient may be zero).",
        "6. **GHTD3 ≈ Hybrid** economically; hierarchical method retains interpretability (SoC goals + market prior) with comparable closed-loop KPI.",
        "",
    ]
    (out / "paper_table_draft_en.md").write_text("\n".join(en) + "\n", encoding="utf-8")

    # Chinese version
    zh = [
        "# 论文表格草稿：三季 FMU 闭环对比（含 PSO）",
        "",
        "**设定**：price-taker 分时电价；周时域 $T=168$ h；同一 FMU 与起点。",
        "**B0** 为模型原始运行（保守规则：高火电+储能 idle）。全部指标在真实 FMU 轨迹上统计。",
        "",
        "## 表 1  各方法周净现金流、能量 SOC、储能与火电",
        "",
        "| 季节 | 方法 | 净现金流 $J$ (CNY) | $\\Delta J$ vs B0 | 周 reward | 能量 SOC | 火电 MWh | 电池吞吐 | CAES 吞吐 | 弃电 MWh |",
        "|------|------|-------------------:|-----------------:|----------:|:--------:|---------:|---------:|----------:|---------:|",
    ]
    for r in rows_sorted:
        vs = r.get("vs_b0") or {}
        dj = vs.get("delta_j_vs_b0")
        mlab = {
            "b0": "B0 保守规则（原始）",
            "b1": "B1 峰谷规则",
            "lp": "M1 滚动松弛 LP",
            "pso": "M2 PSO（参数化）",
            "hybrid": "M3 Hybrid-GiveSafe-TD3",
            "ghtd3": "M4 Safe Market-GHTD3（本文）",
        }.get(r["method"], r["method"])
        slab = {"winter": "冬", "summer": "夏", "transition": "过渡"}.get(r["season"], r["season"])
        zh.append(
            "| {s} | {m} | {j} | {dj} | {rew} | {soc} | {th} | {bt} | {ct} | {cu} |".format(
                s=slab,
                m=mlab,
                j=sci(r.get("net_cashflow_j")),
                dj=("—" if dj is None else sci(dj)),
                rew=fnum(r.get("episode_reward"), 1),
                soc="是" if r.get("terminal_soc_satisfied") else "否",
                th=fnum(r.get("thermal_mwh"), 0),
                bt=fnum(r.get("battery_throughput_mwh"), 0),
                ct=fnum(r.get("caes_throughput_mwh"), 0),
                cu=fnum(r.get("curtailment_mwh"), 2),
            )
        )

    zh += [
        "",
        "## 表 2  相对 B0 的变化（摘要）",
        "",
        "| 季节 | 方法 | $\\Delta J$ | 火电比 | 储能吞吐比 |",
        "|------|------|-----------:|-------:|-----------:|",
    ]
    for r in rows_sorted:
        if r.get("method") == "b0":
            continue
        vs = r.get("vs_b0") or {}
        mlab = {
            "b1": "B1 峰谷",
            "lp": "M1 LP",
            "pso": "M2 PSO",
            "hybrid": "M3 Hybrid",
            "ghtd3": "M4 GHTD3",
        }.get(r["method"], r["method"])
        slab = {"winter": "冬", "summer": "夏", "transition": "过渡"}.get(r["season"], r["season"])
        zh.append(
            "| {s} | {m} | {dj} | {tr} | {sr} |".format(
                s=slab,
                m=mlab,
                dj=sci(vs.get("delta_j_vs_b0")),
                tr=fnum(vs.get("thermal_ratio_vs_b0"), 2),
                sr=fnum(vs.get("storage_throughput_ratio_vs_b0"), 2),
            )
        )

    zh += [
        "",
        "## 表 3  计算量量级",
        "",
        "| 方法 | 每季 FMU 步数 | 说明 |",
        "|------|--------------:|------|",
        "| B0/B1/LP/Hybrid/GHTD3 评估 | 168 | 单周闭环 |",
        "| PSO 搜索 | $\\approx 2.52\\times 10^4$ | 15 代 × 10 粒子 × 168；三季再 ×3 |",
        "| RL 训练（离线） | $10^4$–$10^5$ | 不计入上表评估墙钟 |",
        "",
        "### 结果解读（可写入论文）",
        "",
        "1. **RL 全面领先**：冬/过渡季 Hybrid 与 GHTD3 相对 B0 净现金流约 **+$1.0\\times 10^7$ CNY/周**；夏季由 B0 接近零/负转为 **+$1.1\\times 10^7$**。",
        "2. **火电显著下降**：B0 满发 25200 MWh → RL 约 8600–10300 MWh，配合储能–分时套利。",
        "3. **PSO**（低维参数策略 + 有限代数）在夏/过渡季可优于 B0，但 **明显弱于 RL**；冬季 PSO 火电过低且 SOC 未过关，说明混合非凸动作上黑盒搜索预算不足。",
        "4. **松弛 LP** 可完整跑通 168 步，经济性介于规则与 RL 之间，**不能**代表完整非凸 CAES 数学规划最优。",
        "5. **弃电** 三周内均接近 0（物理量仍报告）；GHTD3 与 Hybrid 经济接近，分层方法保留 goal/市场 prior 可解释性。",
        "",
        "数据路径：`runs/benchmark_merged_3season_pso_20260804/`（实体在 `E:\\optimal_demo_cache\\runs\\...`）。",
        "",
    ]
    (out / "paper_table_draft_zh.md").write_text("\n".join(zh) + "\n", encoding="utf-8")
    # also docs
    docs = Path("docs")
    (docs / "论文表格草稿_三季PSO对比.md").write_text("\n".join(zh) + "\n", encoding="utf-8")
    (docs / "paper_table_draft_seasonal_pso.md").write_text("\n".join(en) + "\n", encoding="utf-8")
    print("wrote", out)
    for r in rows_sorted:
        print(
            r["season"],
            r["method"],
            sci(r.get("net_cashflow_j")),
            r.get("episode_reward"),
            r.get("terminal_soc_satisfied"),
        )


if __name__ == "__main__":
    main()
