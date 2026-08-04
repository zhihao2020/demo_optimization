#!/usr/bin/env python
"""信息结构消融：风光/负荷 perfect vs noisy 日前预报（仅污染观测）。

物理与结算仍由 FMU 真值 + 实现电价驱动。
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
from optimization.metrics import extract_kpi_from_eval  # noqa: E402
from safety import GiveSafeController, NoSafeActionFoundError, load_givesafe_config  # noqa: E402
from training.evaluate_td3 import evaluate_policy  # noqa: E402
from training.ghtd3.agent import GHTD3Agent  # noqa: E402
from training.ghtd3.train import GHTD3PolicyWrapper, load_ghtd3_config  # noqa: E402
from training.hybrid_td3.algorithm import HybridTD3  # noqa: E402

SEASONS = {
    "winter": 0.0,
    "summer": 180 * 24 * 3600.0,
    "transition": 90 * 24 * 3600.0,
}


class GiveSafeWrappedPolicy:
    def __init__(self, agent, env, controller):
        self.agent, self.env, self.controller = agent, env, controller

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


def build_policy(method: str, env: PowerSystemEnv, args: argparse.Namespace) -> Any:
    gs_cfg = load_givesafe_config(ROOT / "src/config/givesafe_config.yaml")
    ctrl = GiveSafeController(oracle=env.oracle, shadow=None, config=gs_cfg)
    dim = int(np.prod(env.observation_space.shape))
    gamma = float(env.reward_calculator.config.get("gamma", 0.99))

    if method == "b0":
        return RuleBasedController(env)
    if method == "hybrid":
        agent = HybridTD3(obs_dim=dim, gamma=gamma)
        agent.load(Path(args.hybrid_ckpt))
        return GiveSafeWrappedPolicy(agent, env, ctrl)
    if method == "ghtd3":
        full_cfg = load_ghtd3_config(ROOT / "src/config/ghtd3_config.yaml")
        cfg = dict(full_cfg.get("ghtd3") or full_cfg)
        agent = GHTD3Agent(dim, cfg)
        agent.load(Path(args.ghtd3_ckpt))
        return GHTD3PolicyWrapper(agent, env, ctrl, cfg)
    raise ValueError(method)


def main() -> None:
    p = argparse.ArgumentParser(description="Resource forecast perfect vs noisy ablation")
    p.add_argument("--methods", type=str, default="b0,hybrid,ghtd3")
    p.add_argument("--seasons", type=str, default="winter,summer,transition")
    p.add_argument("--modes", type=str, default="perfect,noisy")
    p.add_argument("--noise-sigma", type=float, default=0.10, help="乘性 σ（wind/irr/load）")
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument(
        "--hybrid-ckpt",
        default="runs/market_soc_ok_15k_20260803/checkpoints/hybrid_givesafe_td3.pt",
    )
    p.add_argument(
        "--ghtd3-ckpt",
        default="runs/ghtd3_annual_soc_boost_40k_20260803/checkpoints/ghtd3.pt",
    )
    p.add_argument("--out-dir", type=str, default="runs/forecast_info_ablation")
    args = p.parse_args()

    out = resolve_run_dir(args.out_dir)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    rows: list[dict[str, Any]] = []

    for season in seasons:
        start = SEASONS[season]
        for mode in modes:
            for method in methods:
                tag = f"{season}_{mode}_{method}"
                print(f"=== {tag} ===", flush=True)
                sigma = None
                if mode == "noisy":
                    sigma = {
                        "wind": float(args.noise_sigma),
                        "irradiance": float(args.noise_sigma),
                        "ambient_temperature": 0.0,
                        "planned_load": float(args.noise_sigma) * 0.8,
                    }
                env = PowerSystemEnv(
                    run_id=f"fc_{tag}",
                    forecast_enabled=True,
                    forecast_mode=mode,
                    forecast_noise_seed=int(args.noise_seed),
                    forecast_noise_sigma=sigma,
                )
                try:
                    pol = build_policy(method, env, args)
                    t0 = time.perf_counter()
                    res = evaluate_policy(
                        env, pol, reset_options={"start_time": float(start)}
                    )
                    wall = time.perf_counter() - t0
                    kpi = extract_kpi_from_eval(res, wall_s=wall, fmu_steps=res.get("valid_steps"))
                    raw = res.get("weekly_raw_total_cost")
                    if raw is not None:
                        kpi["net_cashflow_j"] = -float(raw)
                    kpi.update(
                        {
                            "season": season,
                            "forecast_mode": mode,
                            "method": method,
                            "noise_sigma": args.noise_sigma if mode == "noisy" else 0.0,
                            "noise_seed": args.noise_seed,
                            "provider_mode": getattr(env.forecast_provider, "mode", None),
                        }
                    )
                    rows.append(kpi)
                    print(
                        f"  J={kpi.get('net_cashflow_j')} rew={kpi.get('episode_reward')} "
                        f"soc={kpi.get('terminal_soc_satisfied')}",
                        flush=True,
                    )
                finally:
                    env.close()

    payload = {
        "rows": rows,
        "note": (
            "Observation-only resource forecast noise; FMU physics and price settlement unchanged. "
            "Not a full market-clearing model."
        ),
        "config": {
            "noise_sigma": args.noise_sigma,
            "noise_seed": args.noise_seed,
            "hybrid_ckpt": args.hybrid_ckpt,
            "ghtd3_ckpt": args.ghtd3_ckpt,
        },
    }
    path = out / "forecast_info_ablation.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
