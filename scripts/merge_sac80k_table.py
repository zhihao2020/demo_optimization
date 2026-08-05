#!/usr/bin/env python
"""Merge SAC-80k seasonal rows with prior extended table (incl. linprog)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sci(x):
    try:
        return f"{float(x):.3e}"
    except Exception:
        return "—"


def f1(x):
    try:
        return f"{float(x):.1f}"
    except Exception:
        return "—"


def main() -> None:
    old_paths = [
        Path("E:/optimal_demo_cache/runs/benchmark_extended_linprog_sac_20260804/extended_table.json"),
        ROOT / "runs/benchmark_extended_linprog_sac_20260804/extended_table.json",
    ]
    new_paths = [
        Path("E:/optimal_demo_cache/runs/benchmark_extended_linprog_sac_80k_20260804/extended_table.json"),
        ROOT / "runs/benchmark_extended_linprog_sac_80k_20260804/extended_table.json",
    ]
    old = new = None
    for p in old_paths:
        if p.is_file():
            old = json.loads(p.read_text(encoding="utf-8"))
            print("old", p)
            break
    for p in new_paths:
        if p.is_file():
            new = json.loads(p.read_text(encoding="utf-8"))
            print("new", p)
            break
    if not old or not new:
        raise SystemExit("missing old/new extended tables")

    old_rows = old.get("rows") or []
    new_rows = new.get("rows") or []
    linprog = [r for r in old_rows if r.get("method") == "linprog"]
    sac = [r for r in new_rows if r.get("method") == "sac"]
    base = [r for r in old_rows if r.get("method") not in ("sac",)]
    # replace sac in base
    base = [r for r in base if r.get("method") != "sac"]
    # ensure linprog present
    if not any(r.get("method") == "linprog" for r in base):
        base = base + linprog

    merged = {}
    for r in base + sac:
        merged[(r.get("season"), r.get("method"))] = r

    order_m = ["b0", "b1", "lp", "linprog", "pso", "sac", "hybrid", "ghtd3"]
    seasons = ["winter", "summer", "transition"]
    sorted_rows = []
    for s in seasons:
        for m in order_m:
            if (s, m) in merged:
                sorted_rows.append(merged[(s, m)])

    ckpt = new.get("sac_ckpt") or "runs/givesafe_sac_80k_20260804/checkpoints/hybrid_givesafe_sac.pt"
    payload = {
        "rows": sorted_rows,
        "note": "Extended with true linprog MPC and Hybrid-SAC 80k (15k+65k resume). Shandong TOU.",
        "sac_ckpt": ckpt,
    }

    out_dirs = [
        Path("E:/optimal_demo_cache/runs/benchmark_extended_linprog_sac_80k_20260804"),
        ROOT / "runs/benchmark_extended_linprog_sac_80k_20260804",
    ]
    labels = {
        "b0": "B0 Rule (original)",
        "b1": "B1 Price rule",
        "lp": "M1 Heuristic rolling",
        "linprog": "M1b True linprog MPC",
        "pso": "M2 PSO parametric",
        "sac": "M5 Hybrid-SAC (80k)",
        "hybrid": "M3 Hybrid-TD3",
        "ghtd3": "M4 Safe Market-GHTD3",
    }
    slab = {"winter": "Winter", "summer": "Summer", "transition": "Transition"}

    lines = [
        "# 扩展基准：真滚动 linprog + SAC-Hybrid",
        "",
        "## 方法说明",
        "",
        "| 代号 | 实现 | 备注 |",
        "|------|------|------|",
        "| **M1b linprog** | `src/optimization/rolling_linprog.py` | scipy HiGHS；闭环 FMU |",
        "| **M5 SAC** | `src/training/hybrid_sac/` | Hybrid-GiveSafe-SAC **80k**（15k 续训 +65k） |",
        "",
        f"ckpt: `{str(ckpt).replace(chr(92), '/')}`",
        "",
        "## 周窗口 reset 依据",
        "",
        "见 `docs/周窗口Reset依据.md`。",
        "",
        "## 表：三季闭环 FMU 指标",
        "",
        "| Season | Method | Net cash flow J | ΔJ vs B0 | Reward | SOC | Thermal MWh | Bat thr. | CAES thr. |",
        "|--------|--------|----------------:|---------:|-------:|:---:|------------:|---------:|----------:|",
    ]
    for r in sorted_rows:
        vs = r.get("vs_b0") or {}
        lines.append(
            "| {s} | {m} | {j} | {dj} | {rew} | {soc} | {th} | {bt} | {ct} |".format(
                s=slab.get(r.get("season"), r.get("season")),
                m=labels.get(r.get("method"), r.get("method")),
                j=sci(r.get("net_cashflow_j")),
                dj=sci(vs.get("delta_j_vs_b0")) if vs else "—",
                rew=f1(r.get("episode_reward")),
                soc="Y" if r.get("terminal_soc_satisfied") else "N",
                th=f1(r.get("thermal_mwh")).replace(".0", "")
                if r.get("thermal_mwh") is not None
                else "—",
                bt=f1(r.get("battery_throughput_mwh")).replace(".0", "")
                if r.get("battery_throughput_mwh") is not None
                else "—",
                ct=f1(r.get("caes_throughput_mwh")).replace(".0", "")
                if r.get("caes_throughput_mwh") is not None
                else "—",
            )
        )
    lines += [
        "",
        "## SAC 长训",
        "",
        "| 项 | 内容 |",
        "|----|------|",
        "| 起点 | `runs/givesafe_sac_15k_20260804` |",
        "| 续训 | +65k → 约 80k |",
        "| 三季 | 本表 M5；三季 SOC=Y |",
        "",
        "## 附录：连续年 SOC",
        "",
        "见 `docs/连续年SOC附录协议.md`。",
        "",
    ]
    md = "\n".join(lines)
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "extended_table.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        (out_dir / "extended_table.md").write_text(md, encoding="utf-8")
    (ROOT / "docs" / "扩展基准_linprog_SAC.md").write_text(md, encoding="utf-8")
    print("methods", sorted({r.get("method") for r in sorted_rows}))
    print("n_rows", len(sorted_rows))


if __name__ == "__main__":
    main()
