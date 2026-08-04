#!/usr/bin/env python
"""GHTD3 分层 goal 训练 CLI。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.ghtd3.train import run_ghtd3_training


def main() -> None:
    p = argparse.ArgumentParser(description="GHTD3 hierarchical goal TD3")
    p.add_argument("--mode", choices=["smoke", "short", "custom"], default="smoke")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--annual-eval", action="store_true")
    p.add_argument("--resume", type=str, default=None, help="从 ghtd3.pt 续训")
    p.add_argument("--skip-bc", action="store_true", help="跳过分层 BC 预热")
    args = p.parse_args()

    if args.mode == "smoke":
        steps = args.steps or 3000
        run_dir = args.run_dir or "runs/ghtd3_smoke"
    elif args.mode == "short":
        steps = args.steps or 20000
        run_dir = args.run_dir or "runs/ghtd3_short"
    else:
        steps = args.steps or 10000
        run_dir = args.run_dir or "runs/ghtd3_custom"

    result = run_ghtd3_training(
        total_valid_steps=steps,
        run_dir=run_dir,
        seed=args.seed,
        annual_evaluation=args.annual_eval,
        resume_from=args.resume,
        skip_bc=bool(args.skip_bc or args.resume),
    )
    # 精简打印
    summary = {
        "status": result.get("status"),
        "valid_steps": result.get("valid_steps"),
        "stats": result.get("stats"),
        "last_metrics": result.get("last_metrics"),
        "eval": {
            k: (result.get("eval") or {}).get(k)
            for k in (
                "episode_reward",
                "weekly_raw_total_cost",
                "economic_cashflow_total",
                "terminal_soc_satisfied",
                "metrics",
            )
        },
        "rule": {
            k: (result.get("rule") or {}).get(k)
            for k in (
                "episode_reward",
                "weekly_raw_total_cost",
                "economic_cashflow_total",
                "terminal_soc_satisfied",
            )
        },
        "price_rule": {
            k: (result.get("price_rule") or {}).get(k)
            for k in (
                "episode_reward",
                "weekly_raw_total_cost",
                "economic_cashflow_total",
                "terminal_soc_satisfied",
            )
        },
        "innovations": result.get("innovations"),
        "annual_eval": result.get("annual_eval"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
