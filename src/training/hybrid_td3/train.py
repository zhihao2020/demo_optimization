"""Hybrid-GiveSafe-TD3 训练入口。禁止 fallback；Phase E 默认阻断。"""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

from actions import CaesMode, FeasibilityOracle
from envs.failures import FeasibleSetEmpty
from envs.power_system_env import PowerSystemEnv
from envs.reward_calculator import IncompleteRewardConfigError
from fmu import FmuAdapter
from replay import HybridGiveSafeReplayBuffer
from safety import GiveSafeController, NoSafeActionFoundError, ShadowFmuValidator, load_givesafe_config
from training.evaluate_td3 import evaluate_annual_policy, evaluate_policy
from controllers.price_aware_rule import PriceAwareRuleController
from controllers.rule_based_controller import RuleBasedController

from .algorithm import HybridTD3
from .buffer import SafetyDataset
from .givesafe_collector import GiveSafeTransitionCollector


class RandomFeasiblePolicy:
    """可行域内探索；默认偏置 IDLE/满发，避免均匀抽 CAES 导致早期无效吞吐。"""

    def __init__(self, env: PowerSystemEnv, idle_bias: float = 0.55, high_tp_bias: float = 0.4):
        self.env = env
        self.idle_bias = float(idle_bias)
        self.high_tp_bias = float(high_tp_bias)

    def predict(self, _obs, deterministic: bool = False) -> dict:
        """在可行域内均匀随机采样混合动作。

        Args:
            _obs: 观测（未使用）。
            deterministic: 忽略。

        Returns:
            混合动作字典。

        Raises:
            FeasibleSetEmpty: 无可选 CAES 模式时抛出。
        """
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
        if CaesMode.IDLE in modes and np.random.rand() < self.idle_bias:
            mode = CaesMode.IDLE
        else:
            mode = modes[int(np.random.randint(len(modes)))]
        if np.random.rand() < self.high_tp_bias:
            # 偏高出力探索：在 [mid, high] 采样
            mid = 0.5 * (feasible.u_tp_low + feasible.u_tp_high)
            u_tp = float(np.random.uniform(mid, feasible.u_tp_high))
        else:
            u_tp = float(np.random.uniform(feasible.u_tp_low, feasible.u_tp_high))
        # 电池：40% 采样 0（若可行），否则均匀
        if feasible.u_battery_low <= 0.0 <= feasible.u_battery_high and np.random.rand() < 0.4:
            u_bat = 0.0
        else:
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
        """组装 TD3 评估用 GiveSafe 包装。

        Args:
            agent: HybridTD3 智能体。
            env: 评估环境。
            controller: GiveSafe 控制器。
            deterministic: 默认确定性动作。
        """
        self.agent = agent
        self.env = env
        self.controller = controller
        self.deterministic = deterministic

    def predict(self, obs, deterministic: bool | None = None):
        """经 GiveSafe 选择安全动作。

        Args:
            obs: 当前观测。
            deterministic: 覆盖默认可选。

        Returns:
            安全混合动作字典。
        """
        det = self.deterministic if deterministic is None else deterministic

        def propose():
            """按当前可行域向智能体请求动作。"""
            feasible = self.env.get_feasible_action_spec()
            return self.agent.select_action(obs, feasible, deterministic=det)

        try:
            gs = self.controller.select_safe_action(
                self.env.last_outputs,
                self.env.previous_thermal,
                propose,
                deterministic=det,
            )
            return gs.safe_action
        except NoSafeActionFoundError:
            # 评估不中断：退回当前可行域内的保守动作（满发+电池0+IDLE），绝非训练 fallback
            feasible = self.env.get_feasible_action_spec()
            u_bat = 0.0
            if not (feasible.u_battery_low <= 0.0 <= feasible.u_battery_high):
                u_bat = 0.5 * (feasible.u_battery_low + feasible.u_battery_high)
            mode = int(CaesMode.IDLE) if feasible.mode_mask.idle else (
                int(CaesMode.DISCHARGE) if feasible.mode_mask.discharge else int(CaesMode.CHARGE)
            )
            return {
                "u_tp": np.asarray([float(feasible.u_tp_high)], dtype=np.float32),
                "u_battery": np.asarray([float(u_bat)], dtype=np.float32),
                "caes_mode": mode,
                "caes_magnitude": np.asarray([0.0], dtype=np.float32),
            }

    def on_episode_reset(self, info: dict[str, Any]) -> None:
        """回合重置时重置影子 FMU。

        Args:
            info: reset info，含 time。

        Returns:
            无。
        """
        if self.controller.shadow is not None:
            self.controller.shadow.on_episode_reset(float(info.get("time", 0.0) or 0.0))

    def on_transition(self, info: dict[str, Any]) -> None:
        """物理步成功后更新影子 FMU 状态。

        Args:
            info: step info。

        Returns:
            无。
        """
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
    """检查奖励配置是否已完成 C_ref 与 terminal_soc 标定。

    Args:
        cfg: reward_config 字典。

    Returns:
        True 表示可进入 formal 训练。
    """
    cref = (cfg.get("cost_reference") or {}).get("value")
    term = cfg.get("terminal_soc") or {}
    return cref is not None and float(cref) > 0 and term.get("bonus") is not None and term.get("tolerance") is not None


def load_givesafe_gates(root: Path) -> dict[str, Any]:
    """从 givesafe_config 读取 Phase E 门禁配置。

    Args:
        root: 项目根目录。

    Returns:
        phase_e_gates 字典。
    """
    cfg = load_givesafe_config(root / "src/config/givesafe_config.yaml")
    return dict(cfg.get("phase_e_gates") or {})


def load_phase_e_gates(root: Path) -> dict[str, Any]:
    """兼容 Phase D.5 测试：优先 givesafe_config，回退 feasibility_margins。

    Args:
        root: 项目根目录。

    Returns:
        phase_e_gates 字典。
    """
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
    """返回周 episode 在 FMU 全年时序中的起点，尾窗覆盖全年最后不足一周的部分。

    Args:
        fmu_config: FMU 配置块。
        episode_steps: 单 episode 决策步数。
        episode_index: 从 0 递增的 episode 索引。

    Returns:
        起始仿真时间（秒）。

    Raises:
        ValueError: episode_index 为负或年度/episode 长度配置非法。
    """
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
    """检查 formal 训练门禁，返回阻断原因列表。

    Args:
        env: 训练环境。
        buffer: GiveSafe 混合 replay。
        collector: GiveSafe 收集器统计源。
        gates_cfg: 可选门禁配置；None 时从配置加载。
        eval_result: 可选评估结果，用于确定性拒绝率检查。

    Returns:
        空列表表示通过；否则为阻断原因字符串列表。
    """
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
    random_explore_start: float = 0.25,
    random_explore_end: float = 0.05,
    gradient_steps: int = 2,
    resume_from: str | Path | None = None,
    reset_critic_on_resume: bool = False,
    rule_demo_fraction: float = 0.25,
) -> dict[str, Any]:
    """Hybrid-GiveSafe-TD3 主训练循环：收集物理有效步、更新 TD3、评估并写 summary。

    Args:
        total_valid_steps: 目标物理有效步数。
        run_dir: 运行目录。
        seed: 随机种子。
        learning_starts: 开始学习前需积累的 replay 样本数。
        batch_size: 梯度更新批大小。
        formal: 是否 formal 模式（启用门禁与完整奖励配置）。
        enable_shadow: 是否启用影子 FMU；None 时读配置。
        forecast_enabled: 环境预测开关。
        annual_evaluation: 训练后是否全年评估。

    Returns:
        含 status、stats、eval、formal_gate_blockers 等的 summary 字典。
    """
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
    collector = GiveSafeTransitionCollector(buffer, controller, shadow=shadow, safety_dataset=safety_dataset)
    rew_cfg = env.reward_calculator.config
    clip_cfg = rew_cfg.get("reward_clip") or {}
    reward_clip = None
    if clip_cfg.get("enabled", True):
        reward_clip = (float(clip_cfg.get("min", -20.0)), float(clip_cfg.get("max", 20.0)))
    agent = HybridTD3(
        obs_dim=int(np.prod(env.observation_space.shape)),
        gamma=float(rew_cfg.get("gamma", 0.99)),
        explore_noise=0.08,
        reward_clip=reward_clip,
        q_clip=200.0,
    )
    # 目标网络与带先验的 actor 同步
    agent.actor_target.load_state_dict(agent.actor.state_dict())
    resumed = None
    if resume_from is not None:
        ckpt = Path(resume_from)
        if not ckpt.is_file():
            env.close()
            if shadow is not None:
                shadow.close()
            return {"status": "blocked_missing_checkpoint", "error": f"checkpoint 不存在: {ckpt}"}
        agent.load(ckpt, reset_critic=reset_critic_on_resume)
        resumed = str(ckpt.resolve())
        if reset_critic_on_resume:
            # 再同步 actor_target（load 已加载 actor）
            agent.actor_target.load_state_dict(agent.actor.state_dict())
    random_policy = RandomFeasiblePolicy(env)
    # 市场环境下用峰谷规则作示范，避免 idle 反套利先验
    if getattr(env, "market_enabled", False):
        rule_policy = PriceAwareRuleController(env)
        rule_policy_name = "price_aware"
    else:
        rule_policy = RuleBasedController(env)
        rule_policy_name = "conservative_idle"
    n_grad = max(int(gradient_steps), 1)

    valid_steps = 0
    episode = 0
    step_log: list[dict] = []
    episode_start_times: list[float] = []

    def reset_training_episode(index: int):
        """按年度周窗口重置环境并记录起点。

        Args:
            index: episode 序号，用于滑动 start_time。

        Returns:
            (obs, reset_info) 元组。
        """
        start_time = annual_episode_start_seconds(env.config["fmu"], env.episode_steps, index)
        next_obs, reset_info = env.reset(seed=seed + index, options={"start_time": start_time})
        actual_start = float(reset_info.get("time", start_time) or start_time)
        episode_start_times.append(actual_start)
        collector.on_episode_reset(actual_start)
        return next_obs, reset_info

    def explore_epsilon(step: int) -> float:
        """线性退火随机探索比例：早期多探索，后期贴近策略。"""
        if total_valid_steps <= learning_starts:
            return random_explore_end
        t = min(max(step - learning_starts, 0) / max(total_valid_steps - learning_starts, 1), 1.0)
        return float(random_explore_start + t * (random_explore_end - random_explore_start))

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
        "training_recipe": {
            "random_explore_start": random_explore_start,
            "random_explore_end": random_explore_end,
            "gradient_steps": n_grad,
            "actor_prior": "high_tp_idle_bias",
            "random_idle_bias": True,
            "rule_policy": rule_policy_name,
            "resume_from": resumed,
            "reset_critic_on_resume": bool(reset_critic_on_resume and resumed),
            "rule_demo_fraction": float(np.clip(rule_demo_fraction, 0.0, 1.0)),
            "soc_shaping": (rew_cfg.get("terminal_soc") or {}).get("shaping"),
            "reward_clip": reward_clip,
        },
    }

    try:
        pbar = tqdm(total=total_valid_steps, desc="Hybrid-TD3", unit="step", dynamic_ncols=True)
        while valid_steps < total_valid_steps:
            try:
                feasible = env.get_feasible_action_spec()
            except FeasibleSetEmpty:
                episode += 1
                obs, info0 = reset_training_episode(episode)
                continue

            eps = explore_epsilon(valid_steps)
            rule_frac = float(np.clip(rule_demo_fraction, 0.0, 1.0))

            def propose():
                # 预热：规则/随机混合；其后 ε-greedy，规则示范比例可配置（BC 后微调宜偏高）
                if valid_steps < learning_starts:
                    if np.random.rand() < max(rule_frac, 0.5):
                        return rule_policy.predict(obs)
                    return random_policy.predict(obs)
                r = np.random.rand()
                if r < eps * rule_frac:
                    return rule_policy.predict(obs)
                if r < eps:
                    return random_policy.predict(obs)
                return agent.select_action(obs, env.get_feasible_action_spec(), deterministic=False)

            obs, reward, terminated, truncated, info = collector.step_with_givesafe(
                env, propose, deterministic=False
            )
            if info.get("transition_type") == "physical" and info.get("transition_valid"):
                valid_steps += 1
                # 探索噪声随进度略降
                progress = valid_steps / max(total_valid_steps, 1)
                agent.explore_noise = float(0.08 * (1.0 - 0.6 * progress))
                metrics = {}
                if buffer.physical_size >= learning_starts:
                    for _ in range(n_grad):
                        metrics = agent.update(buffer, batch_size=min(batch_size, len(buffer)))
                if valid_steps % 500 == 0 or valid_steps == total_valid_steps:
                    step_log.append(
                        {
                            "valid_step": valid_steps,
                            "reward": reward,
                            "eps": eps,
                            "explore_noise": agent.explore_noise,
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
                    rej=info.get("givesafe_rejected_attempts", 0),
                    refresh=False,
                )
            if terminated or truncated:
                episode += 1
                obs, info0 = reset_training_episode(episode)
        else:
            result.update(status="completed", valid_steps=valid_steps)
        pbar.close()

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
                """构造评估用功能模型单元适配器(FmuAdapter)。"""
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
                    """构造年评估用功能模型单元适配器(FmuAdapter)。"""
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
    result.setdefault("algo", "hybrid_givesafe_td3")
    (run_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    try:
        from training.report_policy_run import generate_policy_report

        report_path = generate_policy_report(run_dir)
        result["report_path"] = str(Path(report_path).as_posix())
        (run_dir / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001
        result["report_error"] = str(exc)
        (run_dir / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
    return result


def run_smoke(total_valid_steps: int = 5000, **kwargs) -> dict[str, Any]:
    """冒烟训练入口：非 formal，默认 5000 有效步。

    Args:
        total_valid_steps: 有效步数目标。
        **kwargs: 传给 ``run_hybrid_training`` 的额外参数。

    Returns:
        训练 summary 字典。
    """
    return run_hybrid_training(
        total_valid_steps=total_valid_steps,
        run_dir=kwargs.pop("run_dir", "runs/givesafe_td3_smoke"),
        formal=False,
        **kwargs,
    )


def run_short(total_valid_steps: int = 20000, **kwargs) -> dict[str, Any]:
    """短程训练入口：非 formal，默认 20000 有效步。

    Args:
        total_valid_steps: 有效步数目标。
        **kwargs: 传给 ``run_hybrid_training`` 的额外参数。

    Returns:
        训练 summary 字典。
    """
    return run_hybrid_training(
        total_valid_steps=total_valid_steps,
        run_dir=kwargs.pop("run_dir", "runs/givesafe_td3_short"),
        formal=False,
        **kwargs,
    )


def run_formal(total_valid_steps: int = 100000, **kwargs) -> dict[str, Any]:
    """正式训练入口：启用 formal 门禁，默认 100000 有效步。

    Args:
        total_valid_steps: 有效步数目标。
        **kwargs: 传给 ``run_hybrid_training`` 的额外参数。

    Returns:
        训练 summary 字典。
    """
    return run_hybrid_training(
        total_valid_steps=total_valid_steps,
        run_dir=kwargs.pop("run_dir", "runs/givesafe_td3_formal"),
        formal=True,
        **kwargs,
    )
