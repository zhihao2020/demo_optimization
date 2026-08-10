#!/usr/bin/env python
"""三季公平对比：Safe Market-GHTD3 (hybrid-anchor) vs Hybrid vs B0。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config.paths import apply_process_cache_env  # noqa: E402

apply_process_cache_env()

from controllers.rule_based_controller import RuleBasedController  # noqa: E402
from envs.power_system_env import PowerSystemEnv  # noqa: E402
from safety import GiveSafeController, load_givesafe_config  # noqa: E402
from training.evaluate_td3 import evaluate_policy  # noqa: E402
from training.ghtd3.agent import GHTD3Agent  # noqa: E402
from training.ghtd3.train import GHTD3PolicyWrapper, load_ghtd3_config  # noqa: E402
from training.hybrid_td3.algorithm import HybridTD3  # noqa: E402
from training.hybrid_td3.train import annual_episode_start_seconds  # noqa: E402


SEASONS = {
    "winter": 0,
    "summer": 4344 * 3600,  # approximate; override via starts
    "transition": 2160 * 3600,
}


def _start_for_season(env: PowerSystemEnv, name: str, idx: int) -> float:
    # 与 full benchmark 一致：用 annual_episode_start 的固定周起点
    starts = {"winter": 0, "transition": 13, "summer": 26}  # week indices approx
    week = starts.get(name, idx)
    return float(annual_episode_start_seconds(env.config["fmu"], env.episode_steps, week))


def eval_rule(start: float) -> dict:
    env = PowerSystemEnv(run_id=f"cmp_rule_{start}")
    pol = RuleBasedController(env)
    res = evaluate_policy(env, pol, None, reset_options={"start_time": start})
    env.close()
    return res


def eval_hybrid(ckpt: Path, start: float) -> dict:
    env = PowerSystemEnv(run_id=f"cmp_hyb_{start}")
    dim = int(np.prod(env.observation_space.shape))
    agent = HybridTD3(obs_dim=dim, explore_noise=0.0)
    agent.load(ckpt)
    gs = load_givesafe_config(ROOT / "src/config/givesafe_config.yaml")
    ctrl = GiveSafeController(oracle=env.oracle, shadow=None, config=gs)

    class Pol:
        def on_episode_reset(self, info=None):
            pass

        def predict(self, obs, deterministic=True):
            feas = env.get_feasible_action_spec()

            def propose():
                return agent.select_action(obs, feas, deterministic=True)

            return ctrl.select_safe_action(
                env.last_outputs, env.previous_thermal, propose, deterministic=True, feasible_override=feas
            ).safe_action

    res = evaluate_policy(env, Pol(), None, reset_options={"start_time": start})
    env.close()
    return res


def eval_ghtd3(ckpt: Path, start: float, hybrid_path: Path | None, config_path: Path | None = None) -> dict:
    env = PowerSystemEnv(run_id=f"cmp_gh_{start}")
    dim = int(np.prod(env.observation_space.shape))
    # 优先用 run 目录内配置（TEA/c 等）
    cfg_file = config_path
    if cfg_file is None:
        cand = Path(ckpt).resolve().parents[1] / "config" / "ghtd3_config.yaml"
        if cand.is_file():
            cfg_file = cand
    cfg = dict(load_ghtd3_config(cfg_file).get("ghtd3") or {})
    cfg["execution_mode"] = "goal_conditioned"
    _ = hybrid_path  # legacy arg kept for CLI compatibility; Hybrid teacher path removed
    agent = GHTD3Agent(dim, cfg)
    agent.load(ckpt, strict=False)
    agent.execution_mode = "goal_conditioned"
    agent.low_use_raw_obs = bool(cfg.get("low_use_raw_obs", False))
    gs = load_givesafe_config(ROOT / "src/config/givesafe_config.yaml")
    ctrl = GiveSafeController(oracle=env.oracle, shadow=None, config=gs)
    pol = GHTD3PolicyWrapper(agent, env, ctrl, cfg)
    res = evaluate_policy(env, pol, None, reset_options={"start_time": start})
    # 附带吞吐，便于根因分析
    metrics = res.get("metrics") or {}
    res = dict(res)
    res["battery_throughput_mwh"] = metrics.get("battery_throughput_mwh")
    res["caes_throughput_mwh"] = metrics.get("caes_throughput_mwh")
    res["thermal_generation_mwh"] = metrics.get("thermal_generation_mwh")
    env.close()
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ghtd3", type=str, required=True)
    p.add_argument(
        "--hybrid",
        type=str,
        default="runs/market_bc_rl_60k_20260803/checkpoints/hybrid_givesafe_td3.pt",
    )
    p.add_argument("--out", type=str, default="runs/ghtd3_vs_hybrid_table.json")
    args = p.parse_args()

    seasons = ["winter", "transition", "summer"]
    rows = []
    env_tmp = PowerSystemEnv(run_id="cmp_meta")
    for i, name in enumerate(seasons):
        start = _start_for_season(env_tmp, name, i)
        print("season", name, "start", start)
        b0 = eval_rule(start)
        hy = eval_hybrid(Path(args.hybrid), start)
        gh = eval_ghtd3(Path(args.ghtd3), start, Path(args.hybrid))
        row = {
            "season": name,
            "start_time": start,
            "b0_reward": b0.get("episode_reward"),
            "b0_soc": b0.get("terminal_soc_satisfied"),
            "hybrid_reward": hy.get("episode_reward"),
            "hybrid_soc": hy.get("terminal_soc_satisfied"),
            "hybrid_caes_mwh": (hy.get("metrics") or {}).get("caes_throughput_mwh"),
            "ghtd3_reward": gh.get("episode_reward"),
            "ghtd3_soc": gh.get("terminal_soc_satisfied"),
            "ghtd3_caes_mwh": gh.get("caes_throughput_mwh"),
            "ghtd3_bat_mwh": gh.get("battery_throughput_mwh"),
            "delta_vs_hybrid": float(gh.get("episode_reward") or 0) - float(hy.get("episode_reward") or 0),
        }
        rows.append(row)
        print(row)
    env_tmp.close()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
