#!/usr/bin/env python
"""Hybrid-GiveSafe-SAC 训练 CLI。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config.paths import apply_process_cache_env, resolve_run_dir  # noqa: E402

apply_process_cache_env()

from training.hybrid_sac.train import run_hybrid_sac_training  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Hybrid-GiveSafe-SAC")
    p.add_argument("--mode", choices=["smoke", "short", "custom"], default="short")
    p.add_argument("--steps", type=int, default=None, help="本 run 新采有效步数（续训为追加步数）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--no-shadow", action="store_true")
    p.add_argument("--annual-eval", action="store_true")
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从 hybrid_givesafe_sac.pt 续训（权重热启动；replay 仍从空缓冲重建）",
    )
    args = p.parse_args()

    if args.mode == "smoke":
        steps = args.steps or 3000
        run_dir = args.run_dir or "runs/givesafe_sac_smoke"
    elif args.mode == "short":
        steps = args.steps or 20000
        run_dir = args.run_dir or "runs/givesafe_sac_short"
    else:
        steps = args.steps or 10000
        run_dir = args.run_dir or "runs/givesafe_sac_custom"

    run_dir = str(resolve_run_dir(run_dir))
    result = run_hybrid_sac_training(
        total_valid_steps=steps,
        run_dir=run_dir,
        seed=args.seed,
        enable_shadow=False if args.no_shadow else None,
        annual_evaluation=args.annual_eval,
        resume_from=args.resume,
    )
    summary = {
        "status": result.get("status"),
        "valid_steps": result.get("valid_steps"),
        "algo": result.get("algo"),
        "eval": {
            k: (result.get("eval") or {}).get(k)
            for k in ("episode_reward", "weekly_raw_total_cost", "terminal_soc_satisfied")
        },
        "run_dir": run_dir,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
