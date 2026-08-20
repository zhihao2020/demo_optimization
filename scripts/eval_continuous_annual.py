#!/usr/bin/env python
"""附录：连续年 SOC 协议评估（单次 reset，跨周传递储能状态）。

默认主表协议见 evaluate_annual_policy(continuous_soc=False)。
文档：docs/连续年SOC附录协议.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config.paths import apply_process_cache_env, resolve_run_dir  # noqa: E402

apply_process_cache_env()

from actions import CaesMode  # noqa: E402
from controllers.rule_based_controller import RuleBasedController  # noqa: E402
from envs.power_system_env import PowerSystemEnv  # noqa: E402
from safety import GiveSafeController, NoSafeActionFoundError, load_givesafe_config  # noqa: E402
from training.evaluate_td3 import evaluate_continuous_annual_policy  # noqa: E402
from training.ghtd3.agent import GHTD3Agent  # noqa: E402
from training.ghtd3.train import GHTD3PolicyWrapper, load_ghtd3_config  # noqa: E402
from training.hybrid_sac.algorithm import HybridSAC  # noqa: E402
from training.hybrid_td3.algorithm import HybridTD3  # noqa: E402


class GiveSafeWrappedPolicy:
    """策略输出经 GiveSafe 过滤后再执行。"""

    def __init__(self, agent, env, controller):
        self.agent = agent
        self.env = env
        self.controller = controller

    def predict(self, obs, deterministic=True):
        try:
            feas = self.env.get_feasible_action_spec()
        except Exception:
            return {
                "u_tp": np.asarray([1.0], np.float32),
                "u_battery": np.asarray([0.0], np.float32),
                "caes_mode": int(CaesMode.IDLE),
                "caes_magnitude": np.asarray([0.0], np.float32),
            }

        def prop():
            return self.agent.select_action(obs, feas, deterministic=deterministic)

        try:
            return self.controller.select_safe_action(
                self.env.last_outputs,
                self.env.previous_thermal,
                prop,
                deterministic=deterministic,
                feasible_override=feas,
            ).safe_action
        except NoSafeActionFoundError:
            return {
                "u_tp": np.asarray([float(feas.u_tp_high)], np.float32),
                "u_battery": np.asarray([0.0], np.float32),
                "caes_mode": int(CaesMode.IDLE),
                "caes_magnitude": np.asarray([0.0], np.float32),
            }


def _default_ckpts() -> dict[str, Path]:
    return {
        "hybrid": Path("runs/market_soc_ok_15k_20260803/checkpoints/hybrid_givesafe_td3.pt"),
        "ghtd3": Path("runs/ghtd3_annual_soc_boost_40k_20260803/checkpoints/ghtd3.pt"),
        "sac": Path("runs/givesafe_sac_80k_20260804/checkpoints/hybrid_givesafe_sac.pt"),
        "sac_fallback": Path("runs/givesafe_sac_15k_20260804/checkpoints/hybrid_givesafe_sac.pt"),
    }


def build_policy(method: str, env: PowerSystemEnv, args: argparse.Namespace) -> Any:
    """按方法名构造评估策略。"""
    gs_cfg = load_givesafe_config(ROOT / "src/config/givesafe_config.yaml")
    ctrl = GiveSafeController(oracle=env.oracle, shadow=None, config=gs_cfg)
    ckpts = _default_ckpts()
    dim = int(np.prod(env.observation_space.shape))
    gamma = float(env.reward_calculator.config.get("gamma", 0.99))

    if method == "b0":
        return RuleBasedController(env)

    if method == "hybrid":
        ckpt = Path(args.hybrid_ckpt or ckpts["hybrid"])
        agent = HybridTD3(obs_dim=dim, gamma=gamma)
        agent.load(ckpt)
        return GiveSafeWrappedPolicy(agent, env, ctrl)

    if method == "ghtd3":
        ckpt = Path(args.ghtd3_ckpt or ckpts["ghtd3"])
        full_cfg = load_ghtd3_config(ROOT / "src/config/ghtd3_config_abs.yaml")
        cfg = dict(full_cfg.get("ghtd3") or full_cfg)
        agent = GHTD3Agent(dim, cfg)
        agent.load(ckpt)
        # 官方 wrapper 已含分层 goal + GiveSafe
        return GHTD3PolicyWrapper(agent, env, ctrl, cfg)

    if method == "sac":
        ckpt = Path(args.sac_ckpt) if args.sac_ckpt else ckpts["sac"]
        if not ckpt.is_file():
            ckpt = ckpts["sac_fallback"]
        if not ckpt.is_file():
            raise FileNotFoundError(f"SAC checkpoint not found: {ckpt}")
        agent = HybridSAC(obs_dim=dim, gamma=gamma)
        agent.load(ckpt)
        return GiveSafeWrappedPolicy(agent, env, ctrl)

    raise ValueError(f"unknown method: {method}")


def main() -> None:
    p = argparse.ArgumentParser(description="Continuous-year SoC appendix evaluation")
    p.add_argument("--methods", type=str, default="b0", help="comma list: b0,hybrid,ghtd3,sac")
    p.add_argument("--horizon-hours", type=int, default=8760)
    p.add_argument("--smoke-hours", type=int, default=None, help="override horizon for smoke (e.g. 336)")
    p.add_argument("--out-dir", type=str, default="runs/appendix_continuous_soc")
    p.add_argument("--hybrid-ckpt", type=str, default=None)
    p.add_argument("--ghtd3-ckpt", type=str, default=None)
    p.add_argument("--sac-ckpt", type=str, default=None)
    p.add_argument("--start-time", type=float, default=0.0)
    args = p.parse_args()

    horizon = int(args.smoke_hours or args.horizon_hours)
    out = resolve_run_dir(args.out_dir)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    rows: list[dict[str, Any]] = []

    for method in methods:
        print(f"=== continuous annual: {method} horizon={horizon}h ===", flush=True)
        env = PowerSystemEnv(
            run_id=f"cont_{method}_{horizon}",
            forecast_enabled=True,
            episode_steps=horizon,
        )
        try:
            pol = build_policy(method, env, args)
            sub = out / method
            sub.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            res = evaluate_continuous_annual_policy(
                env,
                pol,
                annual_horizon_hours=horizon,
                output_dir=sub,
                start_time=float(args.start_time),
            )
            wall = time.perf_counter() - t0
            row = {
                "method": method,
                "protocol": res.get("protocol"),
                "horizon_hours": horizon,
                "steps": res.get("steps"),
                "valid_steps": res.get("valid_steps"),
                "annual_raw_total_cost": res.get("annual_raw_total_cost"),
                "net_cashflow_j": -float(res.get("annual_raw_total_cost") or 0.0),
                "annual_episode_reward": res.get("annual_episode_reward"),
                "annual_economic_cashflow": res.get("annual_economic_cashflow"),
                "terminal_soc_satisfied_year_end": res.get("terminal_soc_satisfied_year_end"),
                "terminal_soc": res.get("terminal_soc"),
                "initial_soc": res.get("initial_soc"),
                "metrics": res.get("metrics"),
                "invalid_transition_count": res.get("invalid_transition_count"),
                "fmu_failure_count": res.get("fmu_failure_count"),
                "wall_s": wall,
            }
            rows.append(row)
            (sub / "summary.json").write_text(
                json.dumps(row, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
            )
            print(
                f"  J={row['net_cashflow_j']:.4e} rew={row['annual_episode_reward']} "
                f"soc_ok={row['terminal_soc_satisfied_year_end']} wall={wall:.1f}s",
                flush=True,
            )
        finally:
            env.close()

    payload = {
        "protocol": "continuous_soc",
        "horizon_hours": horizon,
        "rows": rows,
        "note": (
            "Appendix only. Do not merge with weekly_reset main table without labeling protocol. "
            "See docs/连续年SOC附录协议.md"
        ),
    }
    out_path = out / "continuous_summary.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
