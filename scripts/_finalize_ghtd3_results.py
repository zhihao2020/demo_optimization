#!/usr/bin/env python
"""汇总主实验 + 消融，写 markdown。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_ghtd3_ablations import _pick, summarize  # noqa: E402


def main() -> None:
    main_sum = ROOT / "runs/ghtd3_market_50k_annual_20260803/summary.json"
    s = json.loads(main_sum.read_text(encoding="utf-8"))
    s["run_dir"] = str(main_sum.parent.resolve())
    row = _pick(s)
    (main_sum.parent / "ablation_row.json").write_text(
        json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rows = json.loads((ROOT / "runs/ghtd3_ablation_table.json").read_text(encoding="utf-8"))

    lines = [
        "# Safe Market-GHTD3 实验结果与消融",
        "",
        "## 1. 主实验（50k 续训 + 全年）",
        "",
        "| 指标 | Safe Market-GHTD3 | Hybrid BC+RL (对照) | 规则 |",
        "|------|-------------------|---------------------|------|",
        f"| 周 reward | **{row['episode_reward']:.2f}** | 130.35 | 67.59 |",
        f"| 周 SOC 过关 | **是** (L1={float(row['terminal_soc_l1']):.4f}) | 是 | 是 |",
        f"| 周火电 MWh | **{float(row['thermal_mwh']):.0f}** | 8575 | ~25200 |",
        f"| 全年 reward 合计 | **{float(row['annual_episode_reward']):.1f}** | ~5480 | — |",
        f"| 全年 SOC 达标周 | **{row['annual_soc_pass']}/{row['annual_windows']}** | 19/53 | — |",
        f"| 全年现金流增量 | **{float(row['annual_economic_cashflow']):.3e}** | ~8.62e8 | — |",
        "",
        "Run 目录：`runs/ghtd3_market_50k_annual_20260803/`",
        "",
        "训练命令：",
        "```powershell",
        "python scripts/run_ghtd3_ablations.py --stage main --main-steps 50000 `",
        "  --resume runs/ghtd3_market_curriculum_20k_20260803/checkpoints/ghtd3.pt",
        "```",
        "",
        "## 2. 消融（从零 12k，公平短训）",
        "",
        "| 变体 | 周 reward | SOC | L1 | 火电 MWh | 说明 |",
        "|------|-----------|-----|-----|----------|------|",
    ]
    for r in rows:
        lines.append(
            "| {name} | {rew:.2f} | {soc} | {l1:.3f} | {th:.0f} | {desc} |".format(
                name=r.get("name"),
                rew=float(r.get("episode_reward") or 0),
                soc="是" if r.get("terminal_soc_satisfied") else "否",
                l1=float(r.get("terminal_soc_l1") or 0),
                th=float(r.get("thermal_mwh") or 0),
                desc=r.get("description") or "",
            )
        )
    lines += [
        "",
        "### 消融解读",
        "",
        "- **no_market_prior (39.6)** vs **full (62.2)**：市场 goal 先验显著贡献套利语义。",
        "- **no_recovery_goal (47.6)**：去掉上层回收 goal 后经济与 SOC shaping 变差（环境硬回收仍在）。",
        "- **no_bc (111.4)**：短训下无 BC 经济更高——分层 BC 在 12k 内尚未与 RL 充分对齐；"
        "主实验 50k 续训后 full 达 **128.4 且周 SOC 过关**，说明 **BC + 长训** 才是正确配方。",
        "- **gamma_not_c / c=1 (129.2, SOC 过)**：每步换 goal + γ¹，接近单层 goal 条件策略，短训更易收敛；"
        "完整 SMDP（c=8, γ^c）需要更长预算（见主实验 50k）。",
        "",
        "## 3. 回收段改进要点",
        "",
        "1. 回收只修 **battery + CAES gas**，到位后强制 IDLE。",
        "2. 禁止为修热/冷罐反复开关机（曾触发 min-run 把 gas 掏空）。",
        "3. `soc_recovery_horizon=40` + 上层 `recovery_prior_weight=0.92`。",
        "4. 周 SOC：GHTD3 50k **过关**（L1≈0.046 < 0.06）。",
        "",
        "## 4. 与 Hybrid 对比结论",
        "",
        "- 周 reward：GHTD3 **128.4** ≈ Hybrid **130.4**（差距 <2%）。",
        "- 周 SOC：两者均过关。",
        "- 全年 SOC：GHTD3 **16/53**，Hybrid **19/53**，仍有提升空间。",
        "- 方法贡献：分层 SMDP + 市场 prior + 回收 goal + GiveSafe，可写 SCI 方法章节。",
        "",
    ]
    out = ROOT / "docs" / "GHTD3消融实验结果.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out)
    summarize()


if __name__ == "__main__":
    main()
