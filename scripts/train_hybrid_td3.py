#!/usr/bin/env python
"""Hybrid-GiveSafe-TD3 训练 CLI。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.hybrid_td3.train import run_formal, run_short, run_smoke


def main():
    p = argparse.ArgumentParser(description="Hybrid-GiveSafe-TD3")
    p.add_argument("--mode", choices=["smoke", "short", "formal"], default="smoke")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--no-shadow", action="store_true", help="禁用 Shadow FMU（仅一级 Oracle）")
    p.add_argument("--no-forecast", action="store_true", help="不追加 CSV 前瞻观测；用于同种子基线对比")
    p.add_argument("--annual-eval", action="store_true", help="训练后按全年 8760 h 输出对比指标（运行时间较长）")
    args = p.parse_args()
    kwargs = {"seed": args.seed, "enable_shadow": not args.no_shadow}
    if args.no_forecast:
        kwargs["forecast_enabled"] = False
    if args.annual_eval:
        kwargs["annual_evaluation"] = True
    if args.run_dir:
        kwargs["run_dir"] = args.run_dir
    if args.mode == "smoke":
        result = run_smoke(total_valid_steps=args.steps or 5000, **kwargs)
    elif args.mode == "short":
        result = run_short(total_valid_steps=args.steps or 20000, **kwargs)
    else:
        result = run_formal(total_valid_steps=args.steps or 100000, **kwargs)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
