#!/usr/bin/env python
"""IDD：Intent–Dispatch Decoupling 实证（对齐 Cui 高低层敏感性，适配本厂 FMU）。

检验：
  1) G→A 灵敏度（调用 diagnose 逻辑 / 简版扫 g）
  2) 冻高层：固定 market prior goal，只评估已训底层
  3) 冻底层：随机/中性动作，高层意图不进入执行
  4) 峰谷条件：同 g 在峰/谷观测上的动作差 vs 换 g 的差

输出 JSON，供论文图 C。
"""
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

from envs.power_system_env import PowerSystemEnv  # noqa: E402
from safety import GiveSafeController, load_givesafe_config  # noqa: E402
from training.evaluate_td3 import evaluate_policy  # noqa: E402
from training.ghtd3.agent import GHTD3Agent  # noqa: E402
from training.ghtd3.goals import (  # noqa: E402
    DEFAULT_SOC_KEYS,
    extract_soc,
    extract_soc_from_obs,
    market_conditioned_goal_prior,
    plant_intent_vector,
)
from training.ghtd3.train import GHTD3PolicyWrapper, load_ghtd3_config  # noqa: E402
from training.hybrid_td3.train import annual_episode_start_seconds  # noqa: E402


def _load_agent(ckpt: Path, config: Path | None) -> tuple[GHTD3Agent, dict]:
    env = PowerSystemEnv(run_id="idd_meta", forecast_enabled=True)
    dim = int(np.prod(env.observation_space.shape))
    cfg_path = config
    if cfg_path is None:
        cand = ckpt.resolve().parents[1] / "config" / "ghtd3_config.yaml"
        if cand.is_file():
            cfg_path = cand
    cfg = dict(load_ghtd3_config(cfg_path).get("ghtd3") or {})
    agent = GHTD3Agent(dim, cfg)
    agent.load(ckpt, strict=False)
    agent.execution_mode = str(cfg.get("execution_mode", "goal_conditioned")).lower()
    env.close()
    return agent, cfg


def goal_action_sensitivity(agent: GHTD3Agent, obs: np.ndarray, feasible, n: int = 9) -> dict:
    g0 = np.zeros(agent.goal_dim, dtype=np.float32)
    a0 = agent.select_composed_action(obs, g0, feasible, deterministic=True)
    rows = []
    alive = 0
    for i in range(agent.goal_dim):
        lo = float(agent.goal_low[i])
        hi = float(agent.goal_high[i])
        grid = np.linspace(lo, hi, n)
        l1s = []
        for v in grid:
            g = g0.copy()
            g[i] = float(v)
            a = agent.select_composed_action(obs, g, feasible, deterministic=True)
            d = abs(float(a["u_tp"][0]) - float(a0["u_tp"][0]))
            d += abs(float(a["u_battery"][0]) - float(a0["u_battery"][0]))
            d += abs(float(a["caes_magnitude"][0]) - float(a0["caes_magnitude"][0]))
            d += 0.25 * float(int(a["caes_mode"]) != int(a0["caes_mode"]))
            l1s.append(d)
        mx = float(np.max(l1s))
        if mx > 1e-3:
            alive += 1
        rows.append({"dim": int(i), "max_L1": mx, "mean_L1": float(np.mean(l1s))})
    return {"alive_dims": alive, "goal_dim": agent.goal_dim, "per_dim": rows}


def _prior_goal(env, agent, cfg, obs) -> np.ndarray:
    buy = None
    if getattr(env, "price_profile", None) is not None:
        try:
            buy, _ = env.price_profile.prices_at(float(env.adapter.time))
        except Exception:
            buy = None
    outs = env.last_outputs or {}
    intent = plant_intent_vector(outs) if outs else extract_soc_from_obs(obs, 2)
    soc = extract_soc_from_obs(obs, 2)
    soc_init = extract_soc(env.initial_soc, DEFAULT_SOC_KEYS) if env.initial_soc is not None else None
    rem = int(env.episode_steps - env.step_index)
    recovery = rem <= int(cfg.get("recovery_goal_horizon_steps", 36) or 0)
    th = float(intent[2]) if intent.size > 2 else None
    return market_conditioned_goal_prior(
        buy,
        soc,
        soc_init,
        goal_low=agent.goal_low,
        goal_high=agent.goal_high,
        charge_threshold=float(cfg.get("charge_threshold", 0.40)),
        discharge_threshold=float(cfg.get("discharge_threshold", 0.90)),
        recovery=recovery,
        strength=float(cfg.get("market_prior_strength", 0.14)),
        th_mean=th,
    )


def eval_freeze_high(ckpt: Path, cfg: dict, start: float) -> dict:
    """冻高层：每步 g = market prior（不调用 hi actor）。"""
    env = PowerSystemEnv(run_id="idd_freeze_hi", forecast_enabled=True)
    dim = int(np.prod(env.observation_space.shape))
    agent = GHTD3Agent(dim, cfg)
    agent.load(ckpt, strict=False)
    agent.execution_mode = str(cfg.get("execution_mode", "goal_conditioned")).lower()
    gs = load_givesafe_config(ROOT / "src/config/givesafe_config.yaml")
    ctrl = GiveSafeController(oracle=env.oracle, shadow=None, config=gs)

    class Pol:
        def __init__(self):
            self.c = int(cfg.get("subgoal_interval", 8))
            self.step = 0
            self.goal = np.zeros(agent.goal_dim, np.float32)

        def on_episode_reset(self, info=None):
            self.step = 0
            self.goal = np.zeros(agent.goal_dim, np.float32)

        def predict(self, obs, deterministic=True):
            if self.step % self.c == 0:
                self.goal = _prior_goal(env, agent, cfg, obs)
            self.step += 1
            feas = env.get_feasible_action_spec()

            def propose():
                return agent.select_composed_action(obs, self.goal, feas, deterministic=True)

            return ctrl.select_safe_action(
                env.last_outputs, env.previous_thermal, propose, deterministic=True, feasible_override=feas
            ).safe_action

    res = evaluate_policy(env, Pol(), None, reset_options={"start_time": start})
    env.close()
    return {
        "episode_reward": res.get("episode_reward"),
        "soc": res.get("terminal_soc_satisfied"),
        "mode": "freeze_high_prior_only",
    }


def eval_full(ckpt: Path, cfg: dict, start: float) -> dict:
    env = PowerSystemEnv(run_id="idd_full", forecast_enabled=True)
    dim = int(np.prod(env.observation_space.shape))
    agent = GHTD3Agent(dim, cfg)
    agent.load(ckpt, strict=False)
    agent.execution_mode = str(cfg.get("execution_mode", "goal_conditioned")).lower()
    gs = load_givesafe_config(ROOT / "src/config/givesafe_config.yaml")
    ctrl = GiveSafeController(oracle=env.oracle, shadow=None, config=gs)
    pol = GHTD3PolicyWrapper(agent, env, ctrl, cfg)
    res = evaluate_policy(env, pol, None, reset_options={"start_time": start})
    env.close()
    return {
        "episode_reward": res.get("episode_reward"),
        "soc": res.get("terminal_soc_satisfied"),
        "mode": "full",
    }


def eval_freeze_low_random(start: float) -> dict:
    """冻底层：忽略 goal，可行域随机（证明需要可学习底层）。"""
    env = PowerSystemEnv(run_id="idd_freeze_lo", forecast_enabled=True)
    rng = np.random.default_rng(0)

    class Pol:
        def on_episode_reset(self, info=None):
            pass

        def predict(self, obs, deterministic=True):
            feas = env.get_feasible_action_spec()
            u_tp = float(rng.uniform(feas.u_tp_low, feas.u_tp_high))
            u_bat = float(rng.uniform(feas.u_battery_low, feas.u_battery_high))
            modes = [i for i, ok in enumerate(
                [feas.mode_mask.discharge, feas.mode_mask.idle, feas.mode_mask.charge]
            ) if ok]
            mode = int(modes[int(rng.integers(0, len(modes)))]) if modes else 1
            mag = 0.0 if mode == 1 else float(rng.uniform(0.0, 1.0))
            return {
                "u_tp": np.asarray([u_tp], np.float32),
                "u_battery": np.asarray([u_bat], np.float32),
                "caes_mode": mode,
                "caes_magnitude": np.asarray([mag], np.float32),
            }

    res = evaluate_policy(env, Pol(), None, reset_options={"start_time": start})
    env.close()
    return {
        "episode_reward": res.get("episode_reward"),
        "soc": res.get("terminal_soc_satisfied"),
        "mode": "freeze_low_random",
    }


def main() -> None:
    p = argparse.ArgumentParser(description="IDD decoupling suite")
    p.add_argument("--ghtd3", type=str, required=True)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--out", type=str, default="runs/idd_decoupling.json")
    p.add_argument("--season", type=str, default="winter", choices=["winter", "transition", "summer"])
    args = p.parse_args()

    weeks = {"winter": 0, "transition": 13, "summer": 26}
    env_tmp = PowerSystemEnv(run_id="idd_start")
    start = float(annual_episode_start_seconds(env_tmp.config["fmu"], env_tmp.episode_steps, weeks[args.season]))
    env_tmp.close()

    ckpt = Path(args.ghtd3)
    cfg_path = Path(args.config) if args.config else None
    agent, cfg = _load_agent(ckpt, cfg_path)

    env = PowerSystemEnv(run_id="idd_sens", forecast_enabled=True)
    obs, _ = env.reset(seed=0, options={"start_time": start})
    feas = env.get_feasible_action_spec()
    sens = goal_action_sensitivity(agent, np.asarray(obs, np.float32), feas)
    env.close()

    full = eval_full(ckpt, cfg, start)
    fz_hi = eval_freeze_high(ckpt, cfg, start)
    fz_lo = eval_freeze_low_random(start)

    out = {
        "principle": "IDD",
        "season": args.season,
        "start_time": start,
        "goal_action_sensitivity": sens,
        "full": full,
        "freeze_high": fz_hi,
        "freeze_low_random": fz_lo,
        "decoupling_score": {
            "full_minus_freeze_high": float(full["episode_reward"] or 0)
            - float(fz_hi["episode_reward"] or 0),
            "full_minus_freeze_low": float(full["episode_reward"] or 0)
            - float(fz_lo["episode_reward"] or 0),
            "alive_goal_dims": sens["alive_dims"],
        },
    }
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("wrote", op)


if __name__ == "__main__":
    main()
