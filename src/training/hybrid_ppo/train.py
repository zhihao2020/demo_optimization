"""Hybrid-GiveSafe-PPO 训练入口。GiveSafe 拒绝不进 PPO batch。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from envs.failures import FeasibleSetEmpty
from envs.power_system_env import PowerSystemEnv
from envs.reward_calculator import IncompleteRewardConfigError
from fmu import FmuAdapter
from replay import HybridGiveSafeReplayBuffer
from safety import GiveSafeController, ShadowFmuValidator, load_givesafe_config
from training.hybrid_common.eval_and_save import (
    finalize_training_run,
    prepare_run_dir,
    write_summary_and_report,
)
from training.hybrid_td3.buffer import SafetyDataset
from training.hybrid_td3.givesafe_collector import GiveSafeTransitionCollector
from training.hybrid_td3.train import (
    annual_episode_start_seconds,
    check_formal_gates,
    load_givesafe_gates,
)

from .algorithm import HybridPPO
from .rollout import RolloutBuffer


def _reeval_log_prob(agent: HybridPPO, obs, action: dict, feasible) -> float:
    """用当前 Actor 重算已执行动作的 log_prob（PPO old policy 对齐）。

    Args:
        agent: HybridPPO 智能体。
        obs: 步前观测。
        action: 已执行混合动作。
        feasible: 步前可行域规格。

    Returns:
        标量 log_prob。
    """
    with torch.no_grad():
        o = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).view(1, -1)
        mask = torch.as_tensor(
            feasible.mode_mask.as_bool_array(), dtype=torch.bool, device=agent.device
        ).view(1, 3)
        out = agent.actor.evaluate_actions(
            o,
            torch.tensor([float(np.asarray(action["u_tp"]).reshape(-1)[0])], device=agent.device),
            torch.tensor(
                [float(np.asarray(action["u_battery"]).reshape(-1)[0])], device=agent.device
            ),
            torch.tensor([int(action["caes_mode"])], device=agent.device),
            torch.tensor(
                [float(np.asarray(action["caes_magnitude"]).reshape(-1)[0])], device=agent.device
            ),
            torch.tensor([feasible.u_tp_low], device=agent.device),
            torch.tensor([feasible.u_tp_high], device=agent.device),
            torch.tensor([feasible.u_battery_low], device=agent.device),
            torch.tensor([feasible.u_battery_high], device=agent.device),
            mask,
        )
    return float(out["log_prob"][0].cpu())


def run_hybrid_ppo_training(
    total_valid_steps: int = 5000,
    run_dir: str | Path = "runs/givesafe_ppo_smoke",
    seed: int = 0,
    rollout_steps: int = 2048,
    formal: bool = False,
    enable_shadow: bool | None = None,
    forecast_enabled: bool | None = None,
    annual_evaluation: bool = False,
) -> dict[str, Any]:
    """Hybrid-GiveSafe-PPO 主训练：物理步进 rollout，GiveSafe 拒绝跳过 PPO batch。

    Args:
        total_valid_steps: 目标物理有效步数。
        run_dir: 运行目录。
        seed: 随机种子。
        rollout_steps: 每次 PPO 更新的 rollout 长度。
        formal: 是否 formal 模式。
        enable_shadow: 影子 FMU 开关。
        forecast_enabled: 环境预测开关。
        annual_evaluation: 是否全年评估。

    Returns:
        训练 summary 字典。
    """
    run_dir = Path(run_dir)
    root = Path(__file__).resolve().parents[3]
    prepare_run_dir(run_dir, root)

    gs_cfg = load_givesafe_config(root / "src/config/givesafe_config.yaml")
    if gs_cfg.get("givesafe", {}).get("use_fallback", False):
        raise RuntimeError("禁止 use_fallback")
    replay_cfg = gs_cfg.get("replay_sampling") or {}
    shadow_cfg = (gs_cfg.get("givesafe") or {}).get("shadow_validation") or {}
    use_shadow = bool(shadow_cfg.get("enabled", True)) if enable_shadow is None else bool(enable_shadow)

    np.random.seed(seed)
    try:
        env = PowerSystemEnv(
            require_complete_reward=formal, run_id=run_dir.name, forecast_enabled=forecast_enabled
        )
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
            """构造影子仿真用功能模型单元适配器(FmuAdapter)。"""
            return FmuAdapter(fmu_path, step, registry)

        shadow = ShadowFmuValidator(
            factory=factory,
            oracle=env.oracle,
            enabled=True,
            mode=str(shadow_cfg.get("mode", "always")),
            near_boundary_fraction=float(shadow_cfg.get("near_boundary_fraction", 0.15)),
        )

    controller = GiveSafeController(oracle=env.oracle, shadow=shadow, config=gs_cfg)
    # GiveSafe 拒绝仍写入分区 buffer（供门禁统计）；PPO 不从中学习
    buffer = HybridGiveSafeReplayBuffer(
        capacity=100_000,
        physical_fraction=float(replay_cfg.get("physical_fraction", 0.7)),
        givesafe_fraction=float(replay_cfg.get("givesafe_fraction", 0.3)),
    )
    safety_dataset = SafetyDataset()
    collector = GiveSafeTransitionCollector(
        buffer, controller, shadow=shadow, safety_dataset=safety_dataset
    )
    obs_dim = int(np.prod(env.observation_space.shape))
    agent = HybridPPO(
        obs_dim=obs_dim,
        gamma=float(env.reward_calculator.config.get("gamma", 0.99)),
    )
    rollout = RolloutBuffer(capacity=rollout_steps, obs_dim=obs_dim)

    valid_steps = 0
    episode = 0
    step_log: list[dict] = []
    episode_start_times: list[float] = []
    givesafe_reject_count = 0

    def reset_training_episode(index: int):
        """按年度周窗口重置环境并记录起点。

        Args:
            index: episode 序号。

        Returns:
            (obs, reset_info) 元组。
        """
        start_time = annual_episode_start_seconds(env.config["fmu"], env.episode_steps, index)
        next_obs, reset_info = env.reset(seed=seed + index, options={"start_time": start_time})
        actual_start = float(reset_info.get("time", start_time) or start_time)
        episode_start_times.append(actual_start)
        collector.on_episode_reset(actual_start)
        return next_obs, reset_info

    obs, _info0 = reset_training_episode(episode)
    result: dict[str, Any] = {
        "algo": "hybrid_givesafe_ppo",
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
        "observation_dim": obs_dim,
        "rollout_steps": rollout_steps,
    }

    try:
        pbar = tqdm(total=total_valid_steps, desc="Hybrid-PPO", unit="step", dynamic_ncols=True)
        while valid_steps < total_valid_steps:
            rollout.reset()
            while len(rollout) < rollout_steps and valid_steps < total_valid_steps:
                try:
                    feasible = env.get_feasible_action_spec()
                except FeasibleSetEmpty:
                    episode += 1
                    obs, _info0 = reset_training_episode(episode)
                    continue

                pre_obs = np.asarray(obs, dtype=np.float32)
                pre_feasible = feasible

                def propose():
                    """向智能体请求探索动作。"""
                    return agent.select_action(obs, env.get_feasible_action_spec(), deterministic=False)

                obs, reward, terminated, truncated, info = collector.step_with_givesafe(
                    env, propose, deterministic=False
                )
                if info.get("transition_type") != "physical" or not info.get("transition_valid"):
                    givesafe_reject_count += 1
                    if terminated or truncated:
                        episode += 1
                        obs, _info0 = reset_training_episode(episode)
                    continue

                action = {
                    "u_tp": info.get("requested_u_tp", info.get("decoded_u_tp")),
                    "u_battery": info.get("requested_u_battery", info.get("decoded_u_battery")),
                    "caes_mode": info.get("requested_caes_mode", 1),
                    "caes_magnitude": info.get("requested_caes_magnitude", 0.0),
                }
                # hybrid_action 更可靠
                ha = info.get("hybrid_action") or {}
                if ha:
                    action = {
                        "u_tp": ha.get("u_tp", action["u_tp"]),
                        "u_battery": ha.get("u_battery", action["u_battery"]),
                        "caes_mode": ha.get("caes_mode", action["caes_mode"]),
                        "caes_magnitude": ha.get("caes_magnitude", action["caes_magnitude"]),
                    }
                log_prob = _reeval_log_prob(agent, pre_obs, action, pre_feasible)
                value = agent.value_numpy(pre_obs)
                done = bool(terminated or truncated)
                rollout.add(
                    obs=pre_obs,
                    next_obs=np.asarray(obs, dtype=np.float32),
                    action=action,
                    reward=float(reward),
                    done=done,
                    log_prob=log_prob,
                    value=value,
                    mode_mask=pre_feasible.mode_mask.as_bool_array(),
                    bounds={
                        "u_tp_low": pre_feasible.u_tp_low,
                        "u_tp_high": pre_feasible.u_tp_high,
                        "u_battery_low": pre_feasible.u_battery_low,
                        "u_battery_high": pre_feasible.u_battery_high,
                    },
                )
                valid_steps += 1
                pbar.update(1)
                pbar.set_postfix(ep=episode, r=f"{float(reward):.3f}", refresh=False)
                if done:
                    episode += 1
                    obs, _info0 = reset_training_episode(episode)

            if len(rollout) == 0:
                continue
            last_v = 0.0 if (terminated or truncated) else agent.value_numpy(obs)
            rollout.compute_gae(last_v, gamma=agent.gamma, gae_lambda=agent.gae_lambda)
            metrics = agent.update(rollout)
            step_log.append(
                {
                    "valid_step": valid_steps,
                    "rollout_len": len(rollout),
                    "givesafe_rejects_seen": givesafe_reject_count,
                    **metrics,
                }
            )
            pbar.set_postfix(
                ep=episode,
                r=f"{float(reward):.3f}",
                pi=f"{float(metrics.get('policy_loss', 0.0)):.3f}",
                refresh=False,
            )
        else:
            result.update(status="completed", valid_steps=valid_steps)
        pbar.close()

        safety_dataset.save(run_dir / "train" / "safety_dataset.json")
        result = finalize_training_run(
            run_dir=run_dir,
            agent=agent,
            checkpoint_name="hybrid_givesafe_ppo.pt",
            gs_cfg=gs_cfg,
            use_shadow=use_shadow,
            forecast_enabled=forecast_enabled,
            annual_evaluation=annual_evaluation,
            result=result,
            step_log=step_log,
            collector_stats=collector.stats,
            extra_result={
                "physical_replay_size": buffer.physical_size,
                "givesafe_replay_size": buffer.givesafe_size,
                "episodes": episode,
                "episode_start_times_seconds": episode_start_times,
                "ppo_givesafe_reject_skips": givesafe_reject_count,
            },
        )
        blockers = check_formal_gates(
            env, buffer, collector, gates_cfg=gates_cfg, eval_result=result.get("eval")
        )
        result["formal_gate_blockers"] = blockers
        result["phase_e_allowed"] = len(blockers) == 0
        if formal and blockers:
            result.update(status="blocked_formal_gates_post", blockers=blockers)
        result = write_summary_and_report(run_dir, result, step_log)
    finally:
        if shadow is not None:
            shadow.close()
        env.close()

    return result


def run_smoke(total_valid_steps: int = 5000, **kwargs) -> dict[str, Any]:
    """PPO 冒烟训练入口。

    Args:
        total_valid_steps: 有效步数目标。
        **kwargs: 传给 ``run_hybrid_ppo_training`` 的参数。

    Returns:
        训练 summary 字典。
    """
    return run_hybrid_ppo_training(
        total_valid_steps=total_valid_steps,
        run_dir=kwargs.pop("run_dir", "runs/givesafe_ppo_smoke"),
        formal=False,
        **kwargs,
    )


def run_short(total_valid_steps: int = 20000, **kwargs) -> dict[str, Any]:
    """PPO 短程训练入口。

    Args:
        total_valid_steps: 有效步数目标。
        **kwargs: 传给 ``run_hybrid_ppo_training`` 的参数。

    Returns:
        训练 summary 字典。
    """
    return run_hybrid_ppo_training(
        total_valid_steps=total_valid_steps,
        run_dir=kwargs.pop("run_dir", "runs/givesafe_ppo_short"),
        formal=False,
        **kwargs,
    )


def run_formal(total_valid_steps: int = 100000, **kwargs) -> dict[str, Any]:
    """PPO 正式训练入口。

    Args:
        total_valid_steps: 有效步数目标。
        **kwargs: 传给 ``run_hybrid_ppo_training`` 的参数。

    Returns:
        训练 summary 字典。
    """
    return run_hybrid_ppo_training(
        total_valid_steps=total_valid_steps,
        run_dir=kwargs.pop("run_dir", "runs/givesafe_ppo_formal"),
        formal=True,
        **kwargs,
    )
