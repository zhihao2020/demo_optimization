"""GHTD3 训练：高层 goal + 底层 Hybrid-GiveSafe 执行。"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from actions import CaesMode
from controllers.price_aware_rule import PriceAwareRuleController
from controllers.rule_based_controller import RuleBasedController
from envs.failures import FeasibleSetEmpty
from envs.power_system_env import PowerSystemEnv
from safety import GiveSafeController, NoSafeActionFoundError, load_givesafe_config
from training.evaluate_td3 import evaluate_policy
from training.hybrid_td3.train import annual_episode_start_seconds

from .agent import GHTD3Agent
from .buffers import HighTransition, LowTransition
from .goals import (
    DEFAULT_SOC_KEYS,
    blend_goal_with_prior,
    extract_soc,
    extract_soc_from_obs,
    goal_transition,
    intrinsic_reward,
    market_conditioned_goal_prior,
)


def load_ghtd3_config(path: str | Path | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    p = Path(path) if path else root / "src" / "config" / "ghtd3_config.yaml"
    with Path(p).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class GHTD3PolicyWrapper:
    """评估：分层 goal + 市场/回收先验 + GiveSafe 底层。"""

    def __init__(self, agent: GHTD3Agent, env: PowerSystemEnv, controller: GiveSafeController, cfg: dict):
        self.agent = agent
        self.env = env
        self.controller = controller
        self.cfg = dict(cfg)
        self.c = int(cfg.get("subgoal_interval", 8))
        self.step_in_cycle = 0
        self.goal = np.zeros(agent.goal_dim, dtype=np.float32)

    def on_episode_reset(self, info: dict[str, Any]) -> None:
        self.step_in_cycle = 0
        self.goal = np.zeros(self.agent.goal_dim, dtype=np.float32)

    def _select_goal_with_prior(self, obs: np.ndarray) -> np.ndarray:
        goal = self.agent.select_goal(obs, deterministic=True, random=False)
        if not bool(self.cfg.get("market_goal_prior", True)):
            return goal
        buy = None
        if getattr(self.env, "price_profile", None) is not None:
            try:
                buy, _ = self.env.price_profile.prices_at(float(self.env.adapter.time))
            except Exception:
                buy = None
        soc_now = extract_soc_from_obs(obs, self.agent.goal_dim)
        soc_init = None
        if self.env.initial_soc is not None:
            soc_init = extract_soc(self.env.initial_soc, DEFAULT_SOC_KEYS[: self.agent.goal_dim])
        rem = int(self.env.episode_steps - self.env.step_index)
        recovery = rem <= int(self.cfg.get("recovery_goal_horizon_steps", 36) or 0)
        prior = market_conditioned_goal_prior(
            buy,
            soc_now,
            soc_init,
            goal_low=self.agent.goal_low,
            goal_high=self.agent.goal_high,
            charge_threshold=float(self.cfg.get("charge_threshold", 0.40)),
            discharge_threshold=float(self.cfg.get("discharge_threshold", 0.90)),
            recovery=recovery,
            strength=float(self.cfg.get("market_prior_strength", 0.14)),
        )
        w = float(self.cfg.get("market_prior_weight", 0.45))
        if recovery:
            w = max(w, float(self.cfg.get("recovery_prior_weight", 0.92)))
        return blend_goal_with_prior(
            goal, prior, prior_weight=w, goal_low=self.agent.goal_low, goal_high=self.agent.goal_high
        )

    def predict(self, obs, deterministic: bool = True):
        if self.step_in_cycle % self.c == 0:
            self.goal = self._select_goal_with_prior(obs)
        try:
            feasible = self.env.get_feasible_action_spec()
        except FeasibleSetEmpty:
            return {
                "u_tp": np.asarray([1.0], dtype=np.float32),
                "u_battery": np.asarray([0.0], dtype=np.float32),
                "caes_mode": int(CaesMode.IDLE),
                "caes_magnitude": np.asarray([0.0], dtype=np.float32),
            }

        def propose():
            return self.agent.select_low_action(obs, self.goal, feasible, deterministic=deterministic)

        try:
            gs = self.controller.select_safe_action(
                self.env.last_outputs,
                self.env.previous_thermal,
                propose,
                deterministic=deterministic,
                feasible_override=feasible,
            )
            action = gs.safe_action
        except NoSafeActionFoundError:
            action = {
                "u_tp": np.asarray([float(feasible.u_tp_high)], dtype=np.float32),
                "u_battery": np.asarray([0.0], dtype=np.float32),
                "caes_mode": int(CaesMode.IDLE),
                "caes_magnitude": np.asarray([0.0], dtype=np.float32),
            }
        self.step_in_cycle += 1
        # goal 在 env.step 后应转移；评估时在 on_transition 更新
        self._pending_obs = obs
        return action

    def on_transition(self, info: dict[str, Any]) -> None:
        if not info.get("transition_valid"):
            return
        outs = info.get("observations") or self.env.last_outputs or {}
        if not outs or not hasattr(self, "_pending_obs"):
            return
        soc_t = extract_soc_from_obs(self._pending_obs, self.agent.goal_dim)
        soc_tp1 = extract_soc(outs, DEFAULT_SOC_KEYS[: self.agent.goal_dim])
        self.goal = goal_transition(
            soc_t, self.goal, soc_tp1, self.agent.goal_low, self.agent.goal_high
        )


def run_ghtd3_training(
    total_valid_steps: int = 10000,
    run_dir: str | Path = "runs/ghtd3_smoke",
    seed: int = 0,
    config_path: str | Path | None = None,
    annual_evaluation: bool = False,
    resume_from: str | Path | None = None,
    skip_bc: bool = False,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    for name in ("config", "train", "checkpoints", "trajectories"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    full_cfg = load_ghtd3_config(config_path)
    cfg = dict(full_cfg.get("ghtd3") or full_cfg)
    shutil.copy2(root / "src/config/ghtd3_config.yaml", run_dir / "config" / "ghtd3_config.yaml")
    for name in ("env_config.yaml", "reward_config.yaml", "givesafe_config.yaml", "device_params.yaml"):
        src = root / "src/config" / name
        if src.exists():
            shutil.copy2(src, run_dir / "config" / name)

    np.random.seed(seed)
    env = PowerSystemEnv(run_id=run_dir.name, forecast_enabled=True)
    obs_dim = int(np.prod(env.observation_space.shape))
    agent = GHTD3Agent(obs_dim, cfg)
    gs_cfg = load_givesafe_config(root / "src/config/givesafe_config.yaml")
    controller = GiveSafeController(oracle=env.oracle, shadow=None, config=gs_cfg)

    bc_summary: dict[str, Any] | None = None
    if resume_from:
        agent.load(resume_from)
        skip_bc = True
    if bool(cfg.get("bc_pretrain", True)) and not skip_bc:
        from .bc_pretrain import (
            behavior_clone_low_actor,
            bc_pretrain_high_goals,
            collect_hierarchical_demos,
        )

        demos = collect_hierarchical_demos(
            env,
            agent,
            n_windows=int(cfg.get("bc_windows", 4)),
            seed=seed,
            price_aware=True,
            cfg=cfg,
        )
        low_bc = behavior_clone_low_actor(
            agent,
            demos,
            epochs=int(cfg.get("bc_epochs_low", 30)),
        )
        high_bc = bc_pretrain_high_goals(
            agent,
            demos,
            epochs=int(cfg.get("bc_epochs_high", 20)),
        )
        bc_summary = {"low": low_bc, "high": high_bc, "n_demos": int(demos["obs"].shape[0])}
        (run_dir / "train" / "bc_summary.json").write_text(
            json.dumps(bc_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    c = int(cfg.get("subgoal_interval", 8))
    alpha = float(cfg.get("intrinsic_alpha", 0.3))
    eps0 = float(cfg.get("epsilon_start", 0.3))
    eps1 = float(cfg.get("epsilon_end", 0.05))
    learn_lo = int(cfg.get("learning_starts_low", 512))
    learn_hi = int(cfg.get("learning_starts_high", 64))
    bs_lo = int(cfg.get("low_batch_size", 128))
    bs_hi = int(cfg.get("high_batch_size", 64))
    n_grad_lo = int(cfg.get("gradient_steps_low", 2))
    n_grad_hi = int(cfg.get("gradient_steps_high", 1))

    valid_steps = 0
    episode = 0
    step_log: list[dict] = []
    stats = {
        "high_goal_count": 0,
        "low_step_count": 0,
        "givesafe_reject": 0,
        "physical_ok": 0,
    }

    def reset_ep(idx: int):
        start = annual_episode_start_seconds(env.config["fmu"], env.episode_steps, idx)
        obs, info = env.reset(seed=seed + idx, options={"start_time": start})
        return obs, info

    obs, _ = reset_ep(episode)
    goal = np.zeros(agent.goal_dim, dtype=np.float32)
    cycle_ext = 0.0
    cycle_start_obs = obs.copy()
    cycle_soc_seq: list[np.ndarray] = [extract_soc_from_obs(obs, agent.goal_dim)]
    cycle_act_seq: list[dict] = []
    steps_in_cycle = 0
    cycle_goal = goal.copy()

    result: dict[str, Any] = {
        "status": "running",
        "algorithm": "SafeMarketGHTD3",
        "requested_valid_steps": total_valid_steps,
        "ghtd3": cfg,
        "observation_dim": obs_dim,
        "bc_pretrain": bc_summary,
    }

    try:
        while valid_steps < total_valid_steps:
            progress = valid_steps / max(total_valid_steps, 1)
            eps = eps0 + (eps1 - eps0) * progress

            try:
                feasible = env.get_feasible_action_spec()
            except FeasibleSetEmpty:
                episode += 1
                obs, _ = reset_ep(episode)
                steps_in_cycle = 0
                cycle_ext = 0.0
                cycle_start_obs = obs.copy()
                cycle_soc_seq = [extract_soc_from_obs(obs, agent.goal_dim)]
                cycle_act_seq = []
                continue

            # 高层：周期起点采样 goal（市场先验 + 回收目标，对齐论文分层并创新）
            if steps_in_cycle == 0:
                random_goal = (valid_steps < learn_lo) or (np.random.rand() < eps)
                goal = agent.select_goal(obs, deterministic=False, random=random_goal)
                if bool(cfg.get("market_goal_prior", True)):
                    buy = None
                    if getattr(env, "price_profile", None) is not None:
                        try:
                            buy, _ = env.price_profile.prices_at(float(env.adapter.time))
                        except Exception:
                            buy = None
                    soc_now = extract_soc_from_obs(obs, agent.goal_dim)
                    soc_init = None
                    if env.initial_soc is not None:
                        soc_init = extract_soc(env.initial_soc, DEFAULT_SOC_KEYS[: agent.goal_dim])
                    rem = int(env.episode_steps - env.step_index)
                    recovery = rem <= int(cfg.get("recovery_goal_horizon_steps", 36) or 0)
                    prior = market_conditioned_goal_prior(
                        buy,
                        soc_now,
                        soc_init,
                        goal_low=agent.goal_low,
                        goal_high=agent.goal_high,
                        charge_threshold=float(cfg.get("charge_threshold", 0.40)),
                        discharge_threshold=float(cfg.get("discharge_threshold", 0.90)),
                        recovery=recovery,
                        strength=float(cfg.get("market_prior_strength", 0.14)),
                    )
                    w = float(cfg.get("market_prior_weight", 0.45))
                    if recovery:
                        w = max(w, float(cfg.get("recovery_prior_weight", 0.92)))
                    goal = blend_goal_with_prior(
                        goal, prior, prior_weight=w, goal_low=agent.goal_low, goal_high=agent.goal_high
                    )
                cycle_goal = goal.copy()
                cycle_start_obs = obs.copy()
                cycle_ext = 0.0
                cycle_soc_seq = [extract_soc_from_obs(obs, agent.goal_dim)]
                cycle_act_seq = []
                stats["high_goal_count"] += 1

            obs_before = obs.copy()
            soc_before = extract_soc_from_obs(obs_before, agent.goal_dim)
            g_before = goal.copy()

            def propose():
                if valid_steps < learn_lo and np.random.rand() < 0.5:
                    # 冷启动：市场峰谷规则（非 idle），对齐 price-taker 套利
                    if bool(cfg.get("price_aware_bootstrap", True)) and getattr(env, "market_enabled", False):
                        return PriceAwareRuleController(env).predict(obs_before)
                    return RuleBasedController(env).predict(obs_before)
                return agent.select_low_action(obs_before, g_before, feasible, deterministic=False)

            def _on_rej(*_args):
                stats["givesafe_reject"] += 1

            try:
                gs = controller.select_safe_action(
                    env.last_outputs,
                    env.previous_thermal,
                    propose,
                    deterministic=False,
                    feasible_override=feasible,
                    on_rejection=_on_rej,
                )
                action = gs.safe_action
            except NoSafeActionFoundError:
                episode += 1
                obs, _ = reset_ep(episode)
                steps_in_cycle = 0
                continue

            next_obs, r_ext, terminated, truncated, info = env.step(action)
            stats["low_step_count"] += 1

            if not info.get("transition_valid"):
                episode += 1
                obs, _ = reset_ep(episode)
                steps_in_cycle = 0
                continue

            outs = info.get("observations") or env.last_outputs or {}
            soc_after = extract_soc(outs, DEFAULT_SOC_KEYS[: agent.goal_dim])
            r_int, shape_terms = intrinsic_reward(soc_before, g_before, soc_after, float(r_ext), alpha)
            goal_next = goal_transition(soc_before, g_before, soc_after, agent.goal_low, agent.goal_high)

            bounds = {
                "u_tp_low": float(info.get("u_tp_dynamic_low", feasible.u_tp_low)),
                "u_tp_high": float(info.get("u_tp_dynamic_high", feasible.u_tp_high)),
                "u_battery_low": float(info.get("u_battery_dynamic_low", feasible.u_battery_low)),
                "u_battery_high": float(info.get("u_battery_dynamic_high", feasible.u_battery_high)),
            }
            try:
                next_feas = env.get_feasible_action_spec()
                next_bounds = {
                    "u_tp_low": next_feas.u_tp_low,
                    "u_tp_high": next_feas.u_tp_high,
                    "u_battery_low": next_feas.u_battery_low,
                    "u_battery_high": next_feas.u_battery_high,
                }
                next_mask = next_feas.mode_mask.as_bool_array()
            except Exception:
                next_bounds = dict(bounds)
                next_mask = np.ones(3, dtype=bool)

            hybrid = info.get("hybrid_action") or {
                "u_tp": float(np.asarray(action["u_tp"]).ravel()[0]),
                "u_battery": float(np.asarray(action["u_battery"]).ravel()[0]),
                "caes_mode": int(action["caes_mode"]),
                "caes_magnitude": float(np.asarray(action["caes_magnitude"]).ravel()[0]),
            }
            done_flag = bool(terminated or truncated)
            agent.lo_buffer.add(
                LowTransition(
                    observation=obs_before.astype(np.float32),
                    goal=g_before.astype(np.float32),
                    hybrid_action={
                        "u_tp": float(hybrid["u_tp"]),
                        "u_battery": float(hybrid["u_battery"]),
                        "caes_mode": int(hybrid["caes_mode"]),
                        "caes_magnitude": float(hybrid.get("caes_magnitude", 0.0)),
                    },
                    reward_int=float(r_int),
                    next_observation=np.asarray(next_obs, dtype=np.float32),
                    next_goal=goal_next.astype(np.float32),
                    terminated=done_flag,
                    valid_mode_mask=feasible.mode_mask.as_bool_array(),
                    next_valid_mode_mask=next_mask,
                    dynamic_action_bounds=bounds,
                    next_dynamic_action_bounds=next_bounds,
                    reward_terms={**dict(info.get("reward_terms") or {}), **shape_terms},
                )
            )
            stats["physical_ok"] += 1
            valid_steps += 1
            cycle_ext += float(r_ext)
            cycle_soc_seq.append(soc_after.copy())
            cycle_act_seq.append(dict(hybrid))
            steps_in_cycle += 1
            goal = goal_next
            obs = next_obs

            # 周期结束：写高层转移（SMDP 外在奖励；默认 mean 归一化稳定 γ^c critic）
            cycle_done = steps_in_cycle >= c or done_flag
            if cycle_done:
                if bool(cfg.get("high_reward_normalize", True)) and steps_in_cycle > 0:
                    hi_r = float(cycle_ext) / float(steps_in_cycle)
                else:
                    hi_r = float(cycle_ext)
                agent.hi_buffer.add(
                    HighTransition(
                        observation=np.asarray(cycle_start_obs, dtype=np.float32),
                        goal=cycle_goal.astype(np.float32),
                        reward_ext_sum=hi_r,
                        next_observation=np.asarray(obs, dtype=np.float32),
                        terminated=done_flag,
                        soc_seq=list(cycle_soc_seq),
                        action_seq=list(cycle_act_seq),
                    )
                )
                steps_in_cycle = 0

            # 更新
            metrics: dict[str, float] = {}
            if len(agent.lo_buffer) >= learn_lo:
                for _ in range(n_grad_lo):
                    metrics.update(agent.update_low(min(bs_lo, len(agent.lo_buffer))))
            if len(agent.hi_buffer) >= learn_hi and (valid_steps % c == 0):
                for _ in range(n_grad_hi):
                    metrics.update(agent.update_high(min(bs_hi, len(agent.hi_buffer))))

            if valid_steps % 500 == 0 or valid_steps == total_valid_steps:
                step_log.append(
                    {
                        "valid_step": valid_steps,
                        "r_ext": float(r_ext),
                        "r_int": float(r_int),
                        "goal": goal.tolist(),
                        "eps": eps,
                        **metrics,
                    }
                )

            if done_flag:
                episode += 1
                obs, _ = reset_ep(episode)
                steps_in_cycle = 0
                goal = np.zeros(agent.goal_dim, dtype=np.float32)

        result["status"] = "completed"
        result["valid_steps"] = valid_steps
        result["episodes"] = episode
        result["stats"] = stats
        result["last_metrics"] = agent.last_metrics
        agent.save(run_dir / "checkpoints" / "ghtd3.pt")

        # 评估：基线规则 + 峰谷规则 + Safe Market-GHTD3
        rule_env = PowerSystemEnv(run_id=f"{run_dir.name}_rule")
        rule_res = evaluate_policy(rule_env, RuleBasedController(rule_env), run_dir / "trajectories" / "rule.csv")
        rule_env.close()

        price_rule_env = PowerSystemEnv(run_id=f"{run_dir.name}_price_rule")
        price_rule_res = evaluate_policy(
            price_rule_env,
            PriceAwareRuleController(price_rule_env),
            run_dir / "trajectories" / "price_rule.csv",
        )
        price_rule_env.close()

        eval_env = PowerSystemEnv(run_id=f"{run_dir.name}_eval")
        eval_ctrl = GiveSafeController(oracle=eval_env.oracle, shadow=None, config=gs_cfg)
        policy = GHTD3PolicyWrapper(agent, eval_env, eval_ctrl, cfg)
        eval_res = evaluate_policy(eval_env, policy, run_dir / "trajectories" / "eval.csv")
        eval_env.close()

        result["eval"] = eval_res
        result["rule"] = rule_res
        result["price_rule"] = price_rule_res
        result["innovations"] = {
            "smdp_gamma_c": True,
            "market_goal_prior": bool(cfg.get("market_goal_prior", True)),
            "recovery_goal_horizon_steps": int(cfg.get("recovery_goal_horizon_steps", 36) or 0),
            "price_aware_bootstrap": bool(cfg.get("price_aware_bootstrap", True)),
            "high_reward_normalize": bool(cfg.get("high_reward_normalize", True)),
            "hierarchical_bc_pretrain": bool(cfg.get("bc_pretrain", True)),
            "givesafe_low_level": True,
            "huber_q_clip_critics": True,
        }
        if annual_evaluation:
            from training.evaluate_td3 import evaluate_annual_policy

            ann_env = PowerSystemEnv(run_id=f"{run_dir.name}_annual")
            ann_ctrl = GiveSafeController(oracle=ann_env.oracle, shadow=None, config=gs_cfg)
            ann_pol = GHTD3PolicyWrapper(agent, ann_env, ann_ctrl, cfg)
            result["annual_eval"] = evaluate_annual_policy(
                ann_env,
                ann_pol,
                annual_horizon_hours=int(ann_env.config["fmu"]["annual_horizon_hours"]),
                output_dir=run_dir / "trajectories" / "annual_eval",
            )
            ann_env.close()
    finally:
        env.close()

    (run_dir / "train" / "step_log.json").write_text(
        json.dumps(step_log[-200:], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return result


def run_smoke(**kwargs) -> dict[str, Any]:
    return run_ghtd3_training(
        total_valid_steps=kwargs.pop("total_valid_steps", 3000),
        run_dir=kwargs.pop("run_dir", "runs/ghtd3_smoke"),
        **kwargs,
    )


def run_short(**kwargs) -> dict[str, Any]:
    return run_ghtd3_training(
        total_valid_steps=kwargs.pop("total_valid_steps", 20000),
        run_dir=kwargs.pop("run_dir", "runs/ghtd3_short"),
        **kwargs,
    )
