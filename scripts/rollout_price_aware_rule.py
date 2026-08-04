#!/usr/bin/env python
"""价格感知规则一周 rollout（市场结算环境）。"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from controllers.price_aware_rule import PriceAwareRuleController  # noqa: E402
from envs.power_system_env import PowerSystemEnv  # noqa: E402
from training.evaluate_td3 import evaluate_policy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs/price_aware_rule")
    ap.add_argument("--use-predicted-obs", action="store_true")
    ap.add_argument("--use-realized-settle", action="store_true", help="结算用 price_realized.csv")
    args = ap.parse_args()

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    cfg_src = ROOT / "src" / "config" / "env_config.yaml"
    cfg = yaml.safe_load(cfg_src.read_text(encoding="utf-8"))
    market = cfg.setdefault("market", {})
    market["available"] = True
    if args.use_realized_settle and (ROOT / "data" / "price_realized.csv").is_file():
        market["price_path"] = "data/price_realized.csv"
    if args.use_predicted_obs and (ROOT / "data" / "price_predicted.csv").is_file():
        market["obs_price_path"] = "data/price_predicted.csv"
    cfg_path = out / "env_config.override.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    traj = out / "trajectories"
    traj.mkdir(exist_ok=True)
    with PowerSystemEnv(config_path=cfg_path) as env:
        result = evaluate_policy(
            env,
            PriceAwareRuleController(env),
            traj / "rollout.csv",
        )
    (out / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
