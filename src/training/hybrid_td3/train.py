"""Hybrid-GiveSafe-TD3 训练入口。

``run_smoke`` / ``run_short`` / ``run_formal`` 由 ``scripts/train_hybrid_td3.py`` 调用。
禁止规则 fallback；GiveSafe 重采样失败则抛错/截断，不伪造合法动作。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import yaml

from actions import CaesMode, FeasibilityOracle
from envs.failures import FeasibleSetEmpty
from envs.power_system_env import PowerSystemEnv
from envs.reward_calculator import IncompleteRewardConfigError
from fmu import FmuAdapter
from replay import HybridGiveSafeReplayBuffer
from safety import GiveSafeController, ShadowFmuValidator, load_givesafe_config
from training.evaluate_td3 import evaluate_annual_policy, evaluate_policy
from controllers.rule_based_controller import RuleBasedController

from .algorithm import HybridTD3
from .buffer import SafetyDataset
from .givesafe_collector import GiveSafeTransitionCollector


class RandomFeasiblePolicy:
    """在当前动态可行域内均匀采样，用作探索热身与基线对比。"""

    def __init__(self, env: PowerSystemEnv):
        self.env = env

    def predict(self, _obs, deterministic: bool = False) -> dict:
        feasible = self.env.get_feasible_action_spec()
        modes = [
            m
            for m, ok in zip(
                (CaesMode.DISCHARGE, CaesMode.IDLE, CaesMode.CHARGE),
                (feasible.mode_mask.discharge, feasible.mode_mask.idle, feasible.mode_mask.charge),
            )
            if ok
        ]
        if not modes:
            raise FeasibleSetEmpty("无可选 CAES 模式")
        mode = modes[int(np.random.randint(len(modes)))]
        u_tp = float(np.random.uniform(feasible.u_tp_low, feasible.u_tp_high))
        u_bat = float(np.random.uniform(feasible.u_battery_low, feasible.u_battery_high))
        mag = 0.0 if mode == CaesMode.IDLE else float(np.random.uniform(0.0, 1.0))
        return {
            "u_tp": np.asarray([u_tp], dtype=np.float32),
            "u_battery": np.asarray([u_bat], dtype=np.float32),
            "caes_mode": int(mode),
            "caes_magnitude": np.asarray([mag], dtype=np.float32),
        }


class HybridPolicyWrapper:
    """评估用：经 GiveSafeController 采样；绝不调用规则 fallback。"""

    def __init__(self, agent: HybridTD3, env: PowerSystemEnv, controller: GiveSafeController, deterministic: bool = True):
        self.agent = agent
        self.env = env
        self.controller = controller
        self.deterministic = deterministic

    def predict(self, obs, deterministic: bool | None = None):
        det = self.deterministic if deterministic is None else deterministic

        def propose():
            feasible = self.env.get_feasible_action_spec()
            return self.agent.select_action(obs, feasible, deterministic=det)

        gs = self.controller.select_safe_action(
            self.env.last_outputs,
            self.env.previous_thermal,
            propose,
            deterministic=det,
        )
        return gs.safe_action

    def on_episode_reset(self, info: dict[str, Any]) -> None:
        if self.controller.shadow is not None:
            self.controller.shadow.on_episode_reset(float(info.get("time", 0.0) or 0.0))

    def on_transition(self, info: dict[str, Any]) -> None:
        if self.controller.shadow is None or not info.get("transition_valid"):
            return
        self.controller.shadow.on_physical_success(
            {
                "u_tp": float(info["decoded_u_tp"]),
                "u_battery": float(info["decoded_u_battery"]),
                "u_caes": float(info["decoded_u_caes"]),
            }
        )


def _reward_ready(cfg: dict) -> bool:
    cref = (cfg.get("cost_reference") or {}).get("value")
    term = cfg.get("terminal_soc") or {}
    return cref is not None and float(cref) > 0 and term.get("bonus") is not None and term.get("tolerance") is not None


def load_givesafe_gates(root: Path) -> dict[str, Any]:
    cfg = load_givesafe_config(root / "src/config/givesafe_config.yaml")
    return dict(cfg.get("phase_e_gates") or {})


def load_phase_e_gates(root: Path) -> dict[str, Any]:
    """兼容 Phase D.5 测试：优先 givesafe_config，回退 feasibility_margins。"""
    gates = load_givesafe_gates(root)
    if gates:
        return gates
    path = root / "src/config/feasibility_margins.yaml"
    with path.open(encoding="utf-8") as f:
        marg = yaml.safe_load(f) or {}
    return dict(marg.get("phase_e_gates") or {})


def annual_episode_start_seconds(
    fmu_config: dict[str, Any], episode_steps: int, episode_index: int
) -> float:
    """返回周 episode 在 FMU 全年时序中的起点，尾窗覆盖全年最后不足一周的部分。"""
    if episode_index < 0:
        raise ValueError("episode_index 必须非负")
    base = float(fmu_config.get("start_time_seconds", 0.0))
    annual_hours = fmu_config.get("annual_horizon_hours")
    if annual_hours is None:
        return base
    step_seconds = float(fmu_config["decision_interval_seconds"])
    horizon_seconds = float(annual_hours) * 3600.0
    episode_seconds = int(episode_steps) * step_seconds
    if horizon_seconds <= 0 or episode_seconds <= 0 or episode_seconds > horizon_seconds:
        raise ValueError("annual_horizon_hours 必须不小于一个 episode 的长度")
    windows = math.ceil(horizon_seconds / episode_seconds)
    window = episode_index % windows
    # 最后一窗允许与倒数窗口重叠，但绝不越过 FMU 年度 stop time。
    return base + min(window * episode_seconds, horizon_seconds - episode_seconds)


def check_formal_gates(
    env: PowerSystemEnv,
    buffer: HybridGiveSafeReplayBuffer,
    collector: GiveSafeTransitionCollector,
    *,
    gates_cfg: dict[str, Any] | None = None,
    eval_result: dict[str, Any] | None = None,
) -> list[str]:
    blockers: list[str] = []
    root = env.root
    with open(root / "src/config/reward_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not _reward_ready(cfg):
        blockers.append("C_ref / terminal_soc 未标定冻结")
    gates = gates_cfg or load_givesafe_gates(root)
    if gates.get("formal_default_blocked", True):
        blockers.append("formal_default_blocked=true")
    post = collector.stats.get("post_step_hard_constraint_violation_count", 0)
    unsafe = collector.stats.get("main_fmu_unsafe_execution_count", 0)
    if post > int(gates.get("max_post_step_hard_constraint_violations", 0)):
        blockers.append(f"post_step_hard_constraint_violation_count={post}")
    if unsafe > int(gates.get("max_main_fmu_unsafe_executions", 0)):
        blockers.append(f"main_fmu_unsafe_execution_count={unsafe}")
    if buffer.physical.rejected_count > 0:
        # physical 拒绝无效并不等于 invalid in buffer；只要 physical buffer 里全是 valid
        pass
    attempts = max(collector.stats.get("policy_attempt_count", 1), 1)
    no_safe = collector.stats.get("no_safe_action_found_count", 0)
    if no_safe / max(collector.stats.get("physical_transition_count", 1), 1) > float(
        gates.get("max_no_safe_action_found_rate", 0.05)
    ):
        blockers.append(f"NoSafeActionFound 率过高: {no_safe}")
    if eval_result is not None:
        # 确定性拒绝率由 eval 扩展字段提供
        det_rej = float(eval_result.get("proposal_rejection_rate", 0.0) or 0.0)
        if det_rej > float(gates.get("max_deterministic_rejection_rate", 0.02)):
            blockers.append(f"确定性 GiveSafe 拒绝率={det_rej:.4f}")
    return blockers


def run_hybrid_training(
    total_valid_steps: int = 5000,
    run_dir: str | Path = "runs/givesafe_td3_smoke",
    seed: int = 0,
    learning_starts: int = 256,
    batch_size: int = 128,
    formal: bool = False,
    enable_shadow: bool | None = None,
    forecast_enabled: bool | None = None,
    annual_evaluation: bool = False,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("config", "train", "checkpoints", "trajectories"):
        (run_dir / name).mkdir(exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    for cfg_name in (
        "env_config.yaml",
        "reward_config.yaml",
        "device_params.yaml",
        "feasibility_margins.yaml",
        "givesafe_config.yaml",
    ):
        src = root / "src/config" / cfg_name
        if src.exists():
            shutil.copy2(src, run_dir / "config" / cfg_name)

    gs_cfg = load_givesafe_config(root / "src/config/givesafe_config.yaml")
    if gs_cfg.get("givesafe", {}).get("use_fallback", False):
        raise RuntimeError("禁止 use_fallback")
    replay_cfg = gs_cfg.get("replay_sampling") or {}
    shadow_cfg = (gs_cfg.get("givesafe") or {}).get("shadow_validation") or {}
    use_shadow = bool(shadow_cfg.get("enabled", True)) if enable_shadow is None else bool(enable_shadow)

    np.random.seed(seed)
    try:
        env = PowerSystemEnv(require_complete_reward=formal, run_id=run_dir.name, forecast_enabled=forecast_enabled)
    except IncompleteRewardConfigError as exc:
        return {"status": "blocked_incomplete_reward", "error": str(exc)}

    gates_cfg = load_givesafe_gates(root)
    if formal and gates_cfg.get("formal_default_blocked", True):
        env.close()
        return {"status": "blocked_formal_gates", "blockers": ["formal_default_blocked=true"]}

    shadow = None
    if use_shadow:
        fmu_path = env.root / env.config["fmu"]["path"]
        step = float(env.config["fmu"]["communication_step_seconds"])
        registry = env.registry

        def factory():
            return FmuAdapter(fmu_path, step, registry)

        shadow = ShadowFmuValidator(
            factory=factory,
            oracle=env.oracle,
            enabled=True,
            mode=str(shadow_cfg.get("mode", "always")),
            near_boundary_fraction=float(shadow_cfg.get("near_boundary_fraction", 0.15)),
        )

    controller = GiveSafeController(oracle=env.oracle, shadow=shadow, config=gs_cfg)
    buffer = HybridGiveSafeReplayBuffer(
        capacity=100_000,
        physical_fraction=float(replay_cfg.get("physical_fraction", 0.7)),
        givesafe_fraction=float(replay_cfg.get("givesafe_fraction", 0.3)),
    )
    safety_dataset = SafetyDataset()
    collector = GiveSafeTransitionCollector(buffer, controller, shadow=shadow, safety_dataset=safety_dataset)
    agent = HybridTD3(
        obs_dim=int(np.prod(env.observation_space.shape)),
        gamma=float(env.reward_calculator.config.get("gamma", 0.99)),
    )
    random_policy = RandomFeasiblePolicy(env)

    valid_steps = 0
    episode = 0
    step_log: list[dict] = []
    episode_start_times: list[float] = []

    def reset_training_episode(index: int):
        start_time = annual_episode_start_seconds(env.config["fmu"], env.episode_steps, index)
        next_obs, reset_info = env.reset(seed=seed + index, options={"start_time": start_time})
        actual_start = float(reset_info.get("time", start_time) or start_time)
        episode_start_times.append(actual_start)
        collector.on_episode_reset(actual_start)
        return next_obs, reset_info

    obs, info0 = reset_training_episode(episode)
    result: dict[str, Any] = {
        "requested_valid_steps": total_valid_steps,
        "status": "running",
        "formal": formal,
        "givesafe": True,
        "use_fallback": False,
        "shadow_validation": shadow.capabilities() if shadow else {"enabled": False},
        "oracle_version": env.oracle.oracle_version,
        "annual_horizon_hours": env.config["fmu"].get("annual_horizon_hours"),
        "episode_start_schedule": "annual_cycling_windows",
        "forecast_enabled": env.forecast_enabled,
        "forecast_horizon_hours": env.forecast_provider.horizon_hours if env.forecast_provider else 0,
        "observation_dim": int(np.prod(env.observation_space.shape)),
    }

    try:
        while valid_steps < total_valid_steps:
            try:
                feasible = env.get_feasible_action_spec()
            except FeasibleSetEmpty:
                episode += 1
                obs, info0 = reset_training_episode(episode)
                continue

            def propose():
                if valid_steps < learning_starts or np.random.rand() < 0.1:
                    return random_policy.predict(obs)
                return agent.select_action(obs, env.get_feasible_action_spec(), deterministic=False)

            obs, reward, terminated, truncated, info = collector.step_with_givesafe(
                env, propose, deterministic=False
            )
            if info.get("transition_type") == "physical" and info.get("transition_valid"):
                valid_steps += 1
                if buffer.physical_size >= learning_starts:
                    metrics = agent.update(buffer, batch_size=min(batch_size, len(buffer)))
                else:
                    metrics = {}
                step_log.append(
                    {
                        "valid_step": valid_steps,
                        "reward": reward,
                        "attempts": info.get("givesafe_attempt_count"),
                        "rejected": info.get("givesafe_rejected_attempts"),
                        "mode": info.get("requested_caes_mode"),
                        **metrics,
                    }
                )
            if terminated or truncated:
                episode += 1
                obs, info0 = reset_training_episode(episode)
        else:
            result.update(status="completed", valid_steps=valid_steps)

        agent.save(run_dir / "checkpoints" / "hybrid_givesafe_td3.pt")
        safety_dataset.save(run_dir / "train" / "safety_dataset.json")

        # 规则仅作独立基准，不参与 GiveSafe
        rule_env = PowerSystemEnv(run_id=f"{run_dir.name}_rule", forecast_enabled=forecast_enabled)
        rule_result = evaluate_policy(rule_env, RuleBasedController(rule_env), run_dir / "trajectories" / "rule.csv")
        rule_env.close()

        eval_env = PowerSystemEnv(run_id=f"{run_dir.name}_eval", forecast_enabled=forecast_enabled)
        eval_shadow = None
        if use_shadow:
            fmu_path = eval_env.root / eval_env.config["fmu"]["path"]
            step = float(eval_env.config["fmu"]["communication_step_seconds"])

            def efactory():
                return FmuAdapter(fmu_path, step, eval_env.registry)

            eval_shadow = ShadowFmuValidator(
                factory=efactory,
                oracle=eval_env.oracle,
                enabled=True,
                mode=str(shadow_cfg.get("mode", "always")),
            )
        eval_ctrl = GiveSafeController(oracle=eval_env.oracle, shadow=eval_shadow, config=gs_cfg)
        eval_policy = HybridPolicyWrapper(agent, eval_env, eval_ctrl, deterministic=True)
        # 简单确定性 rollout：手动统计拒绝
        # evaluate_policy 直接 step；对 GiveSafe 评估用 wrapper.predict 已含安全环
        try:
            eval_result = evaluate_policy(eval_env, eval_policy, run_dir / "trajectories" / "eval.csv")
        finally:
            eval_shadow.close() if eval_shadow is not None else None
            eval_env.close()

        annual_eval_result = None
        if annual_evaluation:
            annual_env = PowerSystemEnv(run_id=f"{run_dir.name}_annual_eval", forecast_enabled=forecast_enabled)
            annual_shadow = None
            if use_shadow:
                fmu_path = annual_env.root / annual_env.config["fmu"]["path"]
                step = float(annual_env.config["fmu"]["communication_step_seconds"])

                def afactory():
                    return FmuAdapter(fmu_path, step, annual_env.registry)

                annual_shadow = ShadowFmuValidator(
                    factory=afactory,
                    oracle=annual_env.oracle,
                    enabled=True,
                    mode=str(shadow_cfg.get("mode", "always")),
                )
            annual_ctrl = GiveSafeController(oracle=annual_env.oracle, shadow=annual_shadow, config=gs_cfg)
            annual_policy = HybridPolicyWrapper(agent, annual_env, annual_ctrl, deterministic=True)
            try:
                annual_eval_result = evaluate_annual_policy(
                    annual_env,
                    annual_policy,
                    annual_horizon_hours=int(annual_env.config["fmu"]["annual_horizon_hours"]),
                    output_dir=run_dir / "trajectories" / "annual_eval",
                )
            finally:
                annual_shadow.close() if annual_shadow is not None else None
                annual_env.close()

        attempts = max(collector.stats["policy_attempt_count"], 1)
        rej = collector.stats["givesafe_rejection_count"]
        main_exec = max(collector.stats["main_fmu_execution_count"], 1)
        post = collector.stats["post_step_hard_constraint_violation_count"]
        result.update(
            {
                "stats": collector.stats,
                "physical_replay_size": buffer.physical_size,
                "givesafe_replay_size": buffer.givesafe_size,
                "proposal_rejection_rate": rej / attempts,
                "false_safe_rate": collector.stats["givesafe_false_safe_count"]
                / max(collector.stats["main_fmu_execution_count"], 1),
                "main_fmu_execution_safety_rate": 1.0 - post / main_exec,
                "eval": eval_result,
                "annual_eval": annual_eval_result,
                "rule": rule_result,
                "last_metrics": agent.last_metrics,
                "episodes": episode,
                "episode_start_times_seconds": episode_start_times,
            }
        )
        blockers = check_formal_gates(env, buffer, collector, gates_cfg=gates_cfg, eval_result=eval_result)
        result["formal_gate_blockers"] = blockers
        result["phase_e_allowed"] = len(blockers) == 0
        if formal and blockers:
            result.update(status="blocked_formal_gates_post", blockers=blockers)
    finally:
        if shadow is not None:
            shadow.close()
        env.close()

    (run_dir / "train" / "step_log.json").write_text(
        json.dumps(step_log[-500:], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return result


def run_smoke(total_valid_steps: int = 5000, **kwargs) -> dict[str, Any]:
    return run_hybrid_training(
        total_valid_steps=total_valid_steps,
        run_dir=kwargs.pop("run_dir", "runs/givesafe_td3_smoke"),
        formal=False,
        **kwargs,
    )


def run_short(total_valid_steps: int = 20000, **kwargs) -> dict[str, Any]:
    return run_hybrid_training(
        total_valid_steps=total_valid_steps,
        run_dir=kwargs.pop("run_dir", "runs/givesafe_td3_short"),
        formal=False,
        **kwargs,
    )


def run_formal(total_valid_steps: int = 100000, **kwargs) -> dict[str, Any]:
    return run_hybrid_training(
        total_valid_steps=total_valid_steps,
        run_dir=kwargs.pop("run_dir", "runs/givesafe_td3_formal"),
        formal=True,
        **kwargs,
    )
