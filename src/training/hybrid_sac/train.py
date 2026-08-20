"""Hybrid-GiveSafe-SAC 训练入口。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
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
from training.hybrid_common.policy_wrapper import RandomFeasiblePolicy
from training.hybrid_td3.buffer import SafetyDataset
from training.hybrid_td3.givesafe_collector import GiveSafeTransitionCollector
from training.episode_starts import training_start_seconds
from training.hybrid_td3.train import (
    _soft_shell_enabled,
    annual_episode_start_seconds,
    check_formal_gates,
    load_givesafe_gates,
)

from .algorithm import HybridSAC


def run_hybrid_sac_training(
    total_valid_steps: int = 5000,
    run_dir: str | Path = "runs/givesafe_sac_smoke",
    seed: int = 0,
    learning_starts: int = 256,
    batch_size: int = 128,
    formal: bool = False,
    enable_shadow: bool | None = None,
    forecast_enabled: bool | None = None,
    annual_evaluation: bool = False,
    resume_from: str | Path | None = None,
    soft_shell: bool | None = None,
) -> dict[str, Any]:
    """Hybrid-GiveSafe-SAC 主训练循环。

    Args:
        total_valid_steps: 目标物理有效步数（本 run 内新采步数；续训时为追加步数）。
        run_dir: 运行目录。
        seed: 随机种子。
        learning_starts: 开始学习前的 replay 样本数；续训时若已加载权重可置 0。
        batch_size: SAC 更新批大小。
        formal: 是否 formal 模式。
        enable_shadow: 影子 FMU 开关。
        forecast_enabled: 环境预测开关。
        annual_evaluation: 是否全年评估。
        resume_from: 可选 checkpoint 路径（``hybrid_givesafe_sac.pt``）。
        soft_shell: 软约束外壳；None 时读 ``SOFT_SHELL``。

    Returns:
        训练 summary 字典。
    """
    use_soft_shell = _soft_shell_enabled(soft_shell)
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
    buffer = HybridGiveSafeReplayBuffer(
        capacity=100_000,
        physical_fraction=float(replay_cfg.get("physical_fraction", 0.7)),
        givesafe_fraction=float(replay_cfg.get("givesafe_fraction", 0.3)),
    )
    safety_dataset = SafetyDataset()
    collector = GiveSafeTransitionCollector(
        buffer, controller, shadow=shadow, safety_dataset=safety_dataset, soft_shell=use_soft_shell
    )
    agent = HybridSAC(
        obs_dim=int(np.prod(env.observation_space.shape)),
        gamma=float(env.reward_calculator.config.get("gamma", 0.99)),
        parameterized_caes=True,
    )
    resumed: str | None = None
    if resume_from is not None:
        ckpt = Path(resume_from)
        if not ckpt.is_file():
            env.close()
            if shadow is not None:
                shadow.close()
            raise FileNotFoundError(f"SAC resume checkpoint not found: {ckpt}")
        agent.load(ckpt)
        resumed = str(ckpt.resolve())
        # 续训：权重已热，尽早开始梯度更新（仍需先填少量 replay）
        learning_starts = min(int(learning_starts), 64)
    random_policy = RandomFeasiblePolicy(env)

    valid_steps = 0
    episode = 0
    step_log: list[dict] = []
    episode_start_times: list[float] = []

    def reset_training_episode(index: int):
        start_time = training_start_seconds(
            env.config["fmu"],
            env.episode_steps,
            index,
            annual_episode_start_seconds=annual_episode_start_seconds,
        )
        next_obs, reset_info = env.reset(seed=seed + index, options={"start_time": start_time})
        actual_start = float(reset_info.get("time", start_time) or start_time)
        episode_start_times.append(actual_start)
        collector.on_episode_reset(actual_start)
        return next_obs, reset_info

    obs, _info0 = reset_training_episode(episode)
    result: dict[str, Any] = {
        "algo": "hybrid_givesafe_sac",
        "parameterized_caes": bool(agent.parameterized_caes),
        "requested_valid_steps": total_valid_steps,
        "status": "running",
        "formal": formal,
        "givesafe": True,
        "use_fallback": False,
        "soft_shell": use_soft_shell,
        "shadow_validation": shadow.capabilities() if shadow else {"enabled": False},
        "oracle_version": env.oracle.oracle_version,
        "annual_horizon_hours": env.config["fmu"].get("annual_horizon_hours"),
        "episode_start_schedule": "annual_cycling_windows",
        "forecast_enabled": env.forecast_enabled,
        "forecast_horizon_hours": env.forecast_provider.horizon_hours if env.forecast_provider else 0,
        "observation_dim": int(np.prod(env.observation_space.shape)),
        "resume_from": resumed,
        "agent_total_it_at_start": int(agent.total_it),
    }

    try:
        pbar = tqdm(total=total_valid_steps, desc="Hybrid-SAC", unit="step", dynamic_ncols=True)
        while valid_steps < total_valid_steps:
            try:
                env.get_feasible_action_spec()
            except FeasibleSetEmpty:
                episode += 1
                obs, _info0 = reset_training_episode(episode)
                continue

            def propose():
                """随机或策略采样下一步动作提案。"""
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
                # 与 TD3/GHTD3 一致：每 500 valid step 记一条，完整落盘（不截断尾窗）
                if valid_steps % 500 == 0 or valid_steps == total_valid_steps:
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
                pbar.update(1)
                pbar.set_postfix(
                    ep=episode,
                    r=f"{float(reward):.3f}",
                    alpha=f"{float(agent.last_metrics.get('alpha', 0.0)):.3f}",
                    refresh=False,
                )
            if terminated or truncated:
                episode += 1
                obs, _info0 = reset_training_episode(episode)
        else:
            result.update(status="completed", valid_steps=valid_steps)
        pbar.close()

        safety_dataset.save(run_dir / "train" / "safety_dataset.json")
        result = finalize_training_run(
            run_dir=run_dir,
            agent=agent,
            checkpoint_name="hybrid_givesafe_sac.pt",
            gs_cfg=gs_cfg,
            use_shadow=use_shadow,
            forecast_enabled=forecast_enabled,
            annual_evaluation=annual_evaluation,
            result=result,
            step_log=step_log,
            collector_stats=collector.stats,
            soft_shell=use_soft_shell,
            extra_result={
                "physical_replay_size": buffer.physical_size,
                "givesafe_replay_size": buffer.givesafe_size,
                "episodes": episode,
                "episode_start_times_seconds": episode_start_times,
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
    """SAC 冒烟训练入口。

    Args:
        total_valid_steps: 有效步数目标。
        **kwargs: 传给 ``run_hybrid_sac_training`` 的参数。

    Returns:
        训练 summary 字典。
    """
    return run_hybrid_sac_training(
        total_valid_steps=total_valid_steps,
        run_dir=kwargs.pop("run_dir", "runs/givesafe_sac_smoke"),
        formal=False,
        **kwargs,
    )


def run_short(total_valid_steps: int = 20000, **kwargs) -> dict[str, Any]:
    """SAC 短程训练入口。

    Args:
        total_valid_steps: 有效步数目标。
        **kwargs: 传给 ``run_hybrid_sac_training`` 的参数。

    Returns:
        训练 summary 字典。
    """
    return run_hybrid_sac_training(
        total_valid_steps=total_valid_steps,
        run_dir=kwargs.pop("run_dir", "runs/givesafe_sac_short"),
        formal=False,
        **kwargs,
    )


def run_formal(total_valid_steps: int = 100000, **kwargs) -> dict[str, Any]:
    """SAC 正式训练入口。

    Args:
        total_valid_steps: 有效步数目标。
        **kwargs: 传给 ``run_hybrid_sac_training`` 的参数。

    Returns:
        训练 summary 字典。
    """
    return run_hybrid_sac_training(
        total_valid_steps=total_valid_steps,
        run_dir=kwargs.pop("run_dir", "runs/givesafe_sac_formal"),
        formal=True,
        **kwargs,
    )
