#!/usr/bin/env python
"""规则 BC 预热 → Hybrid-GiveSafe-TD3 微调 → 周/全年评估。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.hybrid_td3.bc_rule import run_rule_bc_pretrain
from training.hybrid_td3.train import run_hybrid_training


def main() -> None:
    p = argparse.ArgumentParser(description="Rule BC pretrain + TD3 finetune")
    p.add_argument("--run-dir", type=str, default="runs/bc_then_rl_20260731")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bc-epochs", type=int, default=120)
    p.add_argument("--bc-windows", type=int, default=None, help="规则演示周窗数，默认覆盖全年")
    p.add_argument("--rl-steps", type=int, default=50000)
    p.add_argument("--no-shadow", action="store_true", default=True)
    p.add_argument("--shadow", action="store_true", help="启用 Shadow FMU")
    p.add_argument("--annual-eval", action="store_true", default=True)
    p.add_argument("--no-annual-eval", action="store_true")
    p.add_argument("--rule-demo-fraction", type=float, default=0.55)
    p.add_argument("--skip-bc", action="store_true", help="跳过 BC，直接用已有 bc checkpoint")
    p.add_argument("--bc-checkpoint", type=str, default=None)
    p.add_argument("--no-price-aware", action="store_true", help="BC 用保守 idle 规则而非峰谷规则")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    use_shadow = bool(args.shadow)
    annual = bool(args.annual_eval) and not bool(args.no_annual_eval)
    price_aware = not bool(args.no_price_aware)

    bc_dir = run_dir / "bc"
    rl_dir = run_dir / "rl"

    if args.skip_bc:
        ckpt = Path(args.bc_checkpoint or (bc_dir / "checkpoints" / "hybrid_givesafe_td3.pt"))
        bc_result = {"status": "skipped", "checkpoint": str(ckpt)}
    else:
        bc_result = run_rule_bc_pretrain(
            run_dir=bc_dir,
            seed=args.seed,
            n_windows=args.bc_windows,
            epochs=args.bc_epochs,
            price_aware=price_aware,
        )
        ckpt = Path(bc_result["checkpoint"])

    print(json.dumps({"phase": "bc", **bc_result}, ensure_ascii=False, indent=2, default=str))

    rl_result = run_hybrid_training(
        total_valid_steps=args.rl_steps,
        run_dir=rl_dir,
        seed=args.seed + 100,
        formal=False,
        enable_shadow=use_shadow,
        annual_evaluation=annual,
        resume_from=ckpt,
        reset_critic_on_resume=True,
        rule_demo_fraction=args.rule_demo_fraction,
        random_explore_start=0.18,
        random_explore_end=0.04,
        gradient_steps=2,
        learning_starts=512,
    )
    print(json.dumps({"phase": "rl", **{k: rl_result.get(k) for k in (
        "status", "valid_steps", "eval", "rule", "annual_eval", "last_metrics",
        "proposal_rejection_rate", "main_fmu_execution_safety_rate", "training_recipe",
        "formal_gate_blockers",
    )}}, ensure_ascii=False, indent=2, default=str))

    summary = {
        "bc": bc_result,
        "rl_run_dir": str(rl_dir),
        "rl_status": rl_result.get("status"),
        "rl_eval": rl_result.get("eval"),
        "rule_eval": rl_result.get("rule"),
        "annual_eval": rl_result.get("annual_eval"),
        "last_metrics": rl_result.get("last_metrics"),
        "stats": rl_result.get("stats"),
    }
    # 与历史最佳对照
    hist = ROOT / "runs" / "givesafe_td3_80k_prior_20260730" / "summary.json"
    if hist.is_file():
        prev = json.loads(hist.read_text(encoding="utf-8"))
        summary["compare_to_80k_prior"] = {
            "prev_week_cf": (prev.get("eval") or {}).get("economic_cashflow_total"),
            "prev_annual_cf": (prev.get("annual_eval") or {}).get("annual_economic_cashflow"),
            "prev_week_reward": (prev.get("eval") or {}).get("episode_reward"),
        }
    (run_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({"pipeline": "done", "summary_path": str(run_dir / "pipeline_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
