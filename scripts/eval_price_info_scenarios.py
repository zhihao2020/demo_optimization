#!/usr/bin/env python
"""电价信息场景：Perfect obs vs Predicted obs（结算始终 realized）。"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from controllers.price_aware_rule import PriceAwareRuleController  # noqa: E402
from controllers.rule_based_controller import RuleBasedController  # noqa: E402
from envs.power_system_env import PowerSystemEnv  # noqa: E402
from safety import GiveSafeController, load_givesafe_config  # noqa: E402
from training.evaluate_td3 import evaluate_policy  # noqa: E402
from training.ghtd3.agent import GHTD3Agent  # noqa: E402
from training.ghtd3.train import GHTD3PolicyWrapper, load_ghtd3_config  # noqa: E402


def _patch_market(mode: str) -> Path:
    """写临时 env_config：perfect / predicted / no_price。"""
    src = ROOT / "src/config/env_config.yaml"
    cfg = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    market = dict(cfg.get("market") or {})
    market["available"] = True
    if mode == "perfect":
        market["price_path"] = "data/price_tou.csv"
        market["obs_price_path"] = None
    elif mode == "predicted":
        # 结算用实现价，观测用 BiLSTM 预测价
        market["price_path"] = "data/price_realized.csv"
        market["obs_price_path"] = "data/price_predicted.csv"
    elif mode == "no_price":
        market["available"] = False
        market["price_path"] = "data/price_tou.csv"
        market["obs_price_path"] = None
    else:
        raise ValueError(mode)
    cfg["market"] = market
    tmp = Path(tempfile.mkdtemp(prefix="env_price_"))
    # PowerSystemEnv loads from repo src/config — we monkey by copying into run override.
    # Instead: write full yaml and set via env var is not supported; use temporary replace carefully.
    out = tmp / "env_config.yaml"
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out


def _eval_ghtd3(mode: str, ckpt: str, start_time: float) -> dict[str, Any]:
    # Backup and temporarily swap env_config
    cfg_path = ROOT / "src/config/env_config.yaml"
    bak = cfg_path.read_text(encoding="utf-8")
    patched = _patch_market(mode)
    try:
        shutil.copy2(patched, cfg_path)
        env = PowerSystemEnv(run_id=f"price_{mode}", forecast_enabled=True)
        cfg = dict(load_ghtd3_config().get("ghtd3") or {})
        agent = GHTD3Agent(int(np.prod(env.observation_space.shape)), cfg)
        agent.load(ckpt)
        gs = GiveSafeController(
            oracle=env.oracle,
            shadow=None,
            config=load_givesafe_config(ROOT / "src/config/givesafe_config.yaml"),
        )
        pol = GHTD3PolicyWrapper(agent, env, gs, cfg)
        res = evaluate_policy(env, pol, reset_options={"start_time": float(start_time)})
        env.close()
        terms = res.get("cost_terms") or {}
        return {
            "mode": mode,
            "method": "ghtd3",
            "episode_reward": res.get("episode_reward"),
            "terminal_soc_satisfied": res.get("terminal_soc_satisfied"),
            "terminal_soc_l1": terms.get("terminal_soc_l1_error"),
            "economic_reward": terms.get("economic_reward"),
            "thermal_mwh": (res.get("metrics") or {}).get("thermal_generation_mwh"),
        }
    finally:
        cfg_path.write_text(bak, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--modes", type=str, default="perfect,predicted")
    p.add_argument(
        "--ghtd3-ckpt",
        type=str,
        default="runs/ghtd3_market_50k_annual_20260803/checkpoints/ghtd3.pt",
    )
    p.add_argument("--start-time", type=float, default=0.0, help="周起点秒")
    p.add_argument("--out-dir", type=str, default="runs/price_info_scenarios_20260803")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    rows = []
    for mode in modes:
        print(f"[price-info] mode={mode}")
        if mode == "no_price":
            print("  skip no_price for loaded GHTD3 (obs dim mismatch); use train-time ablation")
            continue
        row = _eval_ghtd3(mode, args.ghtd3_ckpt, args.start_time)
        rows.append(row)
        print(row)

    summary = {"rows": rows, "ckpt": args.ghtd3_ckpt, "start_time": args.start_time}
    (out / "price_info_table.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    lines = [
        "# 电价信息场景（Perfect vs Predicted）",
        "",
        "结算始终用实现价；观测可 perfect 或 BiLSTM 预测价。",
        "",
        "| 模式 | 周 reward | SOC | L1 | 经济项 | 火电 MWh |",
        "|------|-----------|-----|-----|--------|----------|",
    ]
    for r in rows:
        lines.append(
            "| {mode} | {rew:.2f} | {soc} | {l1:.4f} | {eco:.2f} | {th:.0f} |".format(
                mode=r["mode"],
                rew=float(r["episode_reward"] or 0),
                soc="是" if r["terminal_soc_satisfied"] else "否",
                l1=float(r["terminal_soc_l1"] or 0),
                eco=float(r["economic_reward"] or 0),
                th=float(r["thermal_mwh"] or 0),
            )
        )
    (out / "price_info_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "docs" / "电价信息场景实验结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
