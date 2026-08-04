#!/usr/bin/env python
"""三季典型周 × 多方法评估（SCI Case Studies 层 A）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from actions import CaesMode  # noqa: E402
from controllers.price_aware_rule import PriceAwareRuleController  # noqa: E402
from controllers.rule_based_controller import RuleBasedController  # noqa: E402
from envs.power_system_env import PowerSystemEnv  # noqa: E402
from safety import GiveSafeController, NoSafeActionFoundError, load_givesafe_config  # noqa: E402
from training.evaluate_td3 import evaluate_policy  # noqa: E402
from training.ghtd3.agent import GHTD3Agent  # noqa: E402
from training.ghtd3.train import GHTD3PolicyWrapper, load_ghtd3_config  # noqa: E402
from training.hybrid_td3.algorithm import HybridTD3  # noqa: E402

# 对标 Cui 三季 + 本仓库 seasonal starts
SEASONS = {
    "winter": 0,
    "summer": 180 * 24 * 3600,
    "transition": 90 * 24 * 3600,  # spring proxy
}


class HybridPolicy:
    def __init__(self, algo: HybridTD3, env: PowerSystemEnv, ctrl: GiveSafeController):
        self.algo = algo
        self.env = env
        self.ctrl = ctrl

    def predict(self, obs, deterministic: bool = True):
        try:
            feas = self.env.get_feasible_action_spec()
        except Exception:
            return {
                "u_tp": np.asarray([1.0], dtype=np.float32),
                "u_battery": np.asarray([0.0], dtype=np.float32),
                "caes_mode": int(CaesMode.IDLE),
                "caes_magnitude": np.asarray([0.0], dtype=np.float32),
            }

        def propose():
            return self.algo.select_action(obs, feas, deterministic=deterministic)

        try:
            return self.ctrl.select_safe_action(
                self.env.last_outputs,
                self.env.previous_thermal,
                propose,
                deterministic=deterministic,
                feasible_override=feas,
            ).safe_action
        except NoSafeActionFoundError:
            return {
                "u_tp": np.asarray([float(feas.u_tp_high)], dtype=np.float32),
                "u_battery": np.asarray([0.0], dtype=np.float32),
                "caes_mode": int(CaesMode.IDLE),
                "caes_magnitude": np.asarray([0.0], dtype=np.float32),
            }


def _pick(res: dict[str, Any]) -> dict[str, Any]:
    terms = res.get("cost_terms") or {}
    return {
        "episode_reward": res.get("episode_reward"),
        "terminal_soc_satisfied": res.get("terminal_soc_satisfied"),
        "terminal_soc_l1": terms.get("terminal_soc_l1_error"),
        "economic_reward": terms.get("economic_reward"),
        "market_sell_revenue": terms.get("market_sell_revenue"),
        "market_buy_cost": terms.get("market_buy_cost"),
        "thermal_mwh": (res.get("metrics") or {}).get("thermal_generation_mwh"),
        "curtailment_mwh": (res.get("metrics") or {}).get("curtailment_energy_mwh"),
        "unserved_mwh": (res.get("metrics") or {}).get("unserved_energy_mwh"),
        "battery_throughput_mwh": (res.get("metrics") or {}).get("battery_throughput_mwh"),
        "caes_throughput_mwh": (res.get("metrics") or {}).get("caes_throughput_mwh"),
        "terminal_soc": res.get("terminal_soc"),
    }


def eval_method(
    name: str,
    make_policy: Callable[[PowerSystemEnv], Any],
    season: str,
    start_time: float,
    out_csv: Path | None,
) -> dict[str, Any]:
    env = PowerSystemEnv(run_id=f"season_{season}_{name}", forecast_enabled=True)
    try:
        policy = make_policy(env)
        res = evaluate_policy(
            env,
            policy,
            output_csv=out_csv,
            reset_options={"start_time": float(start_time)},
        )
        row = _pick(res)
        row["method"] = name
        row["season"] = season
        row["start_time"] = start_time
        return row
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--methods", type=str, default="rule,price_rule,hybrid,ghtd3")
    p.add_argument(
        "--ghtd3-ckpt",
        type=str,
        default="runs/ghtd3_market_50k_annual_20260803/checkpoints/ghtd3.pt",
    )
    p.add_argument(
        "--hybrid-ckpt",
        type=str,
        default="runs/market_soc_ok_15k_20260803/checkpoints/hybrid_givesafe_td3.pt",
    )
    p.add_argument("--out-dir", type=str, default="runs/seasonal_scenarios_20260803")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj = out_dir / "trajectories"
    traj.mkdir(exist_ok=True)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    gs_cfg = load_givesafe_config(ROOT / "src/config/givesafe_config.yaml")
    ghtd3_cfg = dict(load_ghtd3_config().get("ghtd3") or {})

    # pre-load networks once
    hybrid_algo = None
    if "hybrid" in methods:
        # probe obs dim
        probe = PowerSystemEnv(run_id="probe", forecast_enabled=True)
        obs_dim = int(np.prod(probe.observation_space.shape))
        probe.close()
        hybrid_algo = HybridTD3(obs_dim=obs_dim, device="cpu")
        hybrid_algo.load(Path(args.hybrid_ckpt))

    ghtd3_agent = None
    if "ghtd3" in methods:
        probe = PowerSystemEnv(run_id="probe2", forecast_enabled=True)
        obs_dim = int(np.prod(probe.observation_space.shape))
        probe.close()
        ghtd3_agent = GHTD3Agent(obs_dim, ghtd3_cfg)
        ghtd3_agent.load(args.ghtd3_ckpt)

    factories: dict[str, Callable[[PowerSystemEnv], Any]] = {
        "rule": lambda env: RuleBasedController(env),
        "price_rule": lambda env: PriceAwareRuleController(env),
        "hybrid": lambda env: HybridPolicy(
            hybrid_algo, env, GiveSafeController(oracle=env.oracle, shadow=None, config=gs_cfg)
        ),
        "ghtd3": lambda env: GHTD3PolicyWrapper(
            ghtd3_agent,
            env,
            GiveSafeController(oracle=env.oracle, shadow=None, config=gs_cfg),
            ghtd3_cfg,
        ),
    }

    table: list[dict[str, Any]] = []
    for season, start in SEASONS.items():
        for name in methods:
            if name not in factories:
                raise ValueError(f"未知 method={name}")
            print(f"[eval] {season} × {name} ...")
            csv_path = traj / f"{season}_{name}.csv"
            row = eval_method(name, factories[name], season, start, csv_path)
            table.append(row)
            print(
                f"  rew={row['episode_reward']:.2f} soc={row['terminal_soc_satisfied']} "
                f"l1={row['terminal_soc_l1']}"
            )

    summary = {
        "seasons": SEASONS,
        "methods": methods,
        "ghtd3_ckpt": args.ghtd3_ckpt,
        "hybrid_ckpt": args.hybrid_ckpt,
        "rows": table,
    }
    (out_dir / "seasonal_table.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    # markdown
    lines = [
        "# 三季典型周场景结果（SCI Case Studies）",
        "",
        "| 季节 | 方法 | 周 reward | SOC | L1 | 火电 MWh | 弃电 MWh |",
        "|------|------|-----------|-----|-----|----------|----------|",
    ]
    for r in table:
        lines.append(
            "| {season} | {method} | {rew:.2f} | {soc} | {l1:.4f} | {th:.0f} | {cu:.2f} |".format(
                season=r["season"],
                method=r["method"],
                rew=float(r["episode_reward"] or 0),
                soc="是" if r["terminal_soc_satisfied"] else "否",
                l1=float(r["terminal_soc_l1"] or 0),
                th=float(r["thermal_mwh"] or 0),
                cu=float(r["curtailment_mwh"] or 0),
            )
        )
    lines.append("")
    (out_dir / "seasonal_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "docs" / "三季场景实验结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
