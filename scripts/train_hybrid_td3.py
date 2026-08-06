#!/usr/bin/env python
"""单层 GiveSafe-TD3 训练 CLI（含论文典型 from-scratch 模式）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config.paths import apply_process_cache_env, resolve_run_dir  # noqa: E402

apply_process_cache_env()

from training.hybrid_td3.train import run_formal, run_short, run_smoke, run_td3_scratch  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Single-layer GiveSafe-TD3 (scratch or BC-style)")
    p.add_argument(
        "--mode",
        choices=["smoke", "short", "formal", "scratch"],
        default="smoke",
        help="scratch=论文典型单层 TD3（无规则示范主导，从零训）",
    )
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--no-shadow", action="store_true", help="禁用 Shadow FMU（仅一级 Oracle）")
    p.add_argument("--no-forecast", action="store_true", help="不追加 CSV 前瞻观测；用于同种子基线对比")
    p.add_argument("--annual-eval", action="store_true", help="训练后按全年 8760 h 输出对比指标（运行时间较长）")
    p.add_argument("--resume", type=str, default=None, help="从 HybridTD3 checkpoint (.pt) 续训")
    p.add_argument(
        "--reset-critic",
        action="store_true",
        help="续训时重置 Critic（保留 Actor），用于 Q 发散后的恢复",
    )
    p.add_argument("--rule-demo-fraction", type=float, default=None, help="训练时规则示范比例")
    args = p.parse_args()
    kwargs = {"seed": args.seed, "enable_shadow": not args.no_shadow}
    if args.no_forecast:
        kwargs["forecast_enabled"] = False
    if args.annual_eval:
        kwargs["annual_evaluation"] = True
    if args.run_dir:
        kwargs["run_dir"] = str(resolve_run_dir(args.run_dir))
    if args.resume:
        kwargs["resume_from"] = args.resume
        kwargs["reset_critic_on_resume"] = args.reset_critic
    if args.rule_demo_fraction is not None:
        kwargs["rule_demo_fraction"] = float(args.rule_demo_fraction)
    if args.mode == "smoke":
        result = run_smoke(total_valid_steps=args.steps or 5000, **kwargs)
    elif args.mode == "short":
        result = run_short(total_valid_steps=args.steps or 20000, **kwargs)
    elif args.mode == "scratch":
        # 论文 baseline：无 BC 规则主导；checkpoint 文件名仍为 hybrid_givesafe_td3.pt（兼容加载器）
        if not args.run_dir:
            kwargs["run_dir"] = str(resolve_run_dir(f"runs/td3_scratch_s{args.seed}_{args.steps or 35000}"))
        result = run_td3_scratch(total_valid_steps=args.steps or 35000, **kwargs)
    else:
        result = run_formal(total_valid_steps=args.steps or 100000, **kwargs)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
