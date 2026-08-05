#!/usr/bin/env python
"""SCI 一区实验套件：训练列表 + 三季 eval + 汇总表。

用法示例
--------
# 仅汇总已有 runs
python scripts/run_sci_q1_suite.py --summary-only

# 跑完整套件（耗时长）
python scripts/run_sci_q1_suite.py --train --steps-main 40000 --steps-abl 20000
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def train_one(config: str, run_dir: str, steps: int, seed: int) -> None:
    _run(
        [
            sys.executable,
            "scripts/train_ghtd3.py",
            "--mode",
            "custom",
            "--steps",
            str(steps),
            "--seed",
            str(seed),
            "--run-dir",
            run_dir,
            "--config",
            config,
        ]
    )


def eval_one(ckpt: Path, out: Path, hybrid: str) -> list[dict]:
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "scripts/eval_ghtd3_vs_hybrid.py",
            "--ghtd3",
            str(ckpt),
            "--hybrid",
            hybrid,
            "--out",
            str(out),
        ]
    )
    if out.is_file():
        return json.loads(out.read_text(encoding="utf-8"))
    return []


def summarize(rows_by_name: dict[str, list[dict]], hybrid_name: str = "Hybrid") -> dict:
    """rows: season metrics from vs_hybrid.json"""
    summary = {"methods": {}, "table_md": ""}
    lines = [
        "| Method | Winter | Transition | Summer | mean Δ | #seasons > Hybrid |",
        "|--------|--------|------------|--------|--------|-------------------|",
    ]
    for name, rows in rows_by_name.items():
        if not rows:
            continue
        by_s = {r["season"]: r for r in rows}
        deltas = []
        rews = {}
        wins = 0
        for s in ("winter", "transition", "summer"):
            r = by_s.get(s) or {}
            d = float(r.get("delta_vs_hybrid") or 0.0)
            rews[s] = float(r.get("ghtd3_reward") or 0.0)
            hy = float(r.get("hybrid_reward") or 0.0)
            deltas.append(d)
            if d > 0:
                wins += 1
            rews[s + "_hy"] = hy
        mean_d = sum(deltas) / max(len(deltas), 1)
        summary["methods"][name] = {
            "reward": {s: rews[s] for s in ("winter", "transition", "summer")},
            "delta": {
                "winter": deltas[0],
                "transition": deltas[1],
                "summer": deltas[2],
                "mean": mean_d,
            },
            "wins": wins,
            "hybrid": {s: rews[s + "_hy"] for s in ("winter", "transition", "summer")},
        }
        lines.append(
            f"| {name} | {rews['winter']:.2f} ({deltas[0]:+.2f}) | "
            f"{rews['transition']:.2f} ({deltas[1]:+.2f}) | "
            f"{rews['summer']:.2f} ({deltas[2]:+.2f}) | "
            f"{mean_d:+.2f} | {wins}/3 |"
        )
    summary["table_md"] = "\n".join(lines)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--steps-main", type=int, default=40000)
    p.add_argument("--steps-abl", type=int, default=20000)
    p.add_argument(
        "--hybrid",
        default="runs/market_bc_rl_60k_20260803/checkpoints/hybrid_givesafe_td3.pt",
    )
    p.add_argument("--out", default="runs/sci_q1_suite/summary.json")
    args = p.parse_args()

    # (name, config, run_dir, steps, seed)
    jobs = [
        ("TEA-WS2 seed0", "src/config/ghtd3_config_tea_ws.yaml", "runs/ghtd3_tea_ws2_40k", args.steps_main, 0),
        ("TEA-WS2 seed1", "src/config/ghtd3_config_tea_ws.yaml", "runs/ghtd3_tea_ws2_s1_40k", args.steps_main, 1),
        ("TEA-WS2 seed2", "src/config/ghtd3_config_tea_ws.yaml", "runs/ghtd3_tea_ws2_s2_40k", args.steps_main, 2),
        ("ablate freeze-teacher", "src/config/ghtd3_config_tea_ws_freeze.yaml", "runs/ghtd3_abl_freeze_20k", args.steps_abl, 0),
        ("ablate no-prior", "src/config/ghtd3_config_tea_ws_noprior.yaml", "runs/ghtd3_abl_noprior_20k", args.steps_abl, 0),
        ("ablate no-HER", "src/config/ghtd3_config_tea_ws_noher.yaml", "runs/ghtd3_abl_noher_20k", args.steps_abl, 0),
    ]
    # 已有 ares 作对照
    extras = [
        ("ares-35k", Path("runs/ghtd3_ares_35k/checkpoints/ghtd3.pt"), Path("runs/ghtd3_ares_35k/vs_hybrid.json")),
        ("TEA-WS2 seed0", Path("runs/ghtd3_tea_ws2_40k/checkpoints/ghtd3.pt"), Path("runs/ghtd3_tea_ws2_40k/vs_hybrid.json")),
    ]

    if args.train and not args.summary_only:
        # seed0 已有则跳过训练
        for name, cfg, run_dir, steps, seed in jobs:
            ckpt = ROOT / run_dir / "checkpoints" / "ghtd3.pt"
            if name.endswith("seed0") and ckpt.is_file():
                print(f"[skip train] {name} exists {ckpt}")
                continue
            if "ablate" in name and ckpt.is_file():
                print(f"[skip train] {name} exists")
                continue
            if name.startswith("TEA-WS2 seed") and seed == 0 and ckpt.is_file():
                continue
            train_one(cfg, run_dir, steps, seed)

    rows_by = {}
    # eval all
    for name, cfg, run_dir, steps, seed in jobs:
        ckpt = ROOT / run_dir / "checkpoints" / "ghtd3.pt"
        out = ROOT / run_dir / "vs_hybrid.json"
        if not ckpt.is_file():
            print(f"[skip eval] missing {ckpt}")
            continue
        if out.is_file() and name.endswith("seed0"):
            # re-eval only if missing unless force
            rows_by[name] = json.loads(out.read_text(encoding="utf-8"))
            continue
        rows_by[name] = eval_one(ckpt, out, args.hybrid)

    for name, ckpt, out in extras:
        if out.is_file():
            rows_by[name] = json.loads(out.read_text(encoding="utf-8"))
        elif ckpt.is_file():
            rows_by[name] = eval_one(ckpt, out, args.hybrid)

    # multi-seed stats for TEA-WS2
    seed_rows = []
    for k, v in list(rows_by.items()):
        if k.startswith("TEA-WS2 seed"):
            seed_rows.append(v)
    multi = None
    if len(seed_rows) >= 2:
        import numpy as np

        multi = {}
        for s in ("winter", "transition", "summer"):
            rews = [float(next(r["ghtd3_reward"] for r in rows if r["season"] == s)) for rows in seed_rows]
            hy = float(next(r["hybrid_reward"] for r in seed_rows[0] if r["season"] == s))
            multi[s] = {
                "mean": float(np.mean(rews)),
                "std": float(np.std(rews, ddof=1) if len(rews) > 1 else 0.0),
                "hybrid": hy,
                "delta_mean": float(np.mean(rews) - hy),
            }
        multi["mean_delta"] = float(np.mean([multi[s]["delta_mean"] for s in multi]))

    summary = summarize(rows_by)
    summary["multi_seed_tea_ws2"] = multi
    summary["n_methods"] = len(rows_by)
    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md = outp.with_suffix(".md")
    md_body = ["# SCI Q1 suite summary", "", summary["table_md"], ""]
    if multi:
        md_body.append("## TEA-WS2 multi-seed")
        md_body.append("")
        for s, v in multi.items():
            if s == "mean_delta":
                md_body.append(f"- **mean Δ (all seasons)**: {v:+.3f}")
            else:
                md_body.append(
                    f"- **{s}**: {v['mean']:.2f} ± {v['std']:.2f} (Hybrid {v['hybrid']:.2f}, Δ {v['delta_mean']:+.2f})"
                )
    md.write_text("\n".join(md_body) + "\n", encoding="utf-8")
    print(md.read_text(encoding="utf-8"))
    print("wrote", outp)


if __name__ == "__main__":
    main()
