#!/usr/bin/env python
"""市场环境下实验编排：残差电价 → BiLSTM → 价格规则 rollout → Hybrid smoke/short。

示例：
  python scripts/run_market_experiments.py --stage all --hybrid-mode smoke
  python scripts/run_market_experiments.py --stage data,bilstm,rule
  python scripts/run_market_experiments.py --stage hybrid --hybrid-mode short --steps 15000
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd or ROOT))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def stage_data() -> None:
    run([sys.executable, "scripts/build_price_residual_series.py"])


def stage_bilstm(*, epochs: int, device: str) -> None:
    run(
        [
            sys.executable,
            "scripts/train_price_bilstm.py",
            "--epochs",
            str(epochs),
            "--device",
            device,
        ]
    )


def stage_rule(*, use_predicted_obs: bool) -> None:
    # 用实现价结算；可选预测价进 obs
    env_extra = {}
    if use_predicted_obs:
        pred = ROOT / "data" / "price_predicted.csv"
        realized = ROOT / "data" / "price_realized.csv"
        if pred.is_file() and realized.is_file():
            # 临时写一份实验用 env 覆盖通过 CLI 环境变量不方便，直接改用子脚本参数
            pass
    run([sys.executable, "scripts/rollout_price_aware_rule.py"] + (
        ["--use-predicted-obs"] if use_predicted_obs else []
    ))


def stage_hybrid(*, mode: str, steps: int | None, seed: int, no_shadow: bool) -> None:
    cmd = [
        sys.executable,
        "scripts/train_hybrid_td3.py",
        "--mode",
        mode,
        "--seed",
        str(seed),
        "--run-dir",
        f"runs/market_hybrid_{mode}_{seed}",
    ]
    if steps is not None:
        cmd += ["--steps", str(steps)]
    if no_shadow:
        cmd.append("--no-shadow")
    run(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        default="data,bilstm,rule",
        help="逗号分隔: data,bilstm,rule,hybrid,all",
    )
    ap.add_argument("--hybrid-mode", choices=["smoke", "short", "formal"], default="smoke")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-shadow", action="store_true")
    ap.add_argument("--use-predicted-obs", action="store_true")
    args = ap.parse_args()

    stages = [s.strip() for s in args.stage.split(",") if s.strip()]
    if "all" in stages:
        stages = ["data", "bilstm", "rule", "hybrid"]

    summary: dict = {"stages": stages}
    for s in stages:
        if s == "data":
            stage_data()
        elif s == "bilstm":
            stage_bilstm(epochs=args.epochs, device=args.device)
        elif s == "rule":
            stage_rule(use_predicted_obs=args.use_predicted_obs)
        elif s == "hybrid":
            stage_hybrid(
                mode=args.hybrid_mode,
                steps=args.steps,
                seed=args.seed,
                no_shadow=args.no_shadow,
            )
        else:
            raise SystemExit(f"未知 stage: {s}")
    print(json.dumps({"ok": True, **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
