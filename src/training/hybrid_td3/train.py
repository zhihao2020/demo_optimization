"""Hybrid-GiveSafe-TD3 训练入口。禁止 fallback；Phase E 默认阻断。"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

from envs.failures import FeasibleSetEmpty
from envs.power_system_env import PowerSystemEnv
from envs.reward_calculator import IncompleteRewardConfigError
from fmu import FmuAdapter, describe_fmu
from replay import HybridGiveSafeReplayBuffer
from safety import GiveSafeController, ShadowFmuValidator, load_givesafe_config
from training.episode_starts import eval_start_seconds, training_start_seconds
from training.evaluate_td3 import evaluate_annual_policy, evaluate_policy
from controllers.price_aware_rule import PriceAwareRuleController
from controllers.rule_based_controller import RuleBasedController
from training.hybrid_common.policy_wrapper import (
    HybridGiveSafePolicyWrapper as HybridPolicyWrapper,
    RandomFeasiblePolicy,
)

from .algorithm import HybridTD3
from .buffer import SafetyDataset
from .givesafe_collector import GiveSafeTransitionCollector


def _soft_shell_enabled(explicit: bool | None = None) -> bool:
    """CLI/显式参数优先；否则读环境变量 ``SOFT_SHELL=1``。"""
    if explicit is not None:
        return bool(explicit)
    return os.environ.get("SOFT_SHELL", "").strip().lower() in ("1", "true", "yes", "on")


def _git_commit(root: Path) -> str:
    head = root / ".git" / "HEAD"
    try:
        raw = head.read_text(encoding="utf-8").strip()
        if raw.startswith("ref:"):
            ref = root / ".git" / raw.split(":", 1)[1].strip()
            return ref.read_text(encoding="utf-8").strip()[:40]
        return raw[:40]
    except OSError:
        return ""


def _paper_cfg(root: Path) -> dict[str, Any]:
    path = root / "src" / "config" / "paper_pc_hybrid_td3.yaml"
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _paper_algo_cfg(root: Path) -> dict[str, Any]:
    return dict(_paper_cfg(root).get("algorithm") or {})


def _stamp_summary_status(result: dict[str, Any]) -> None:
    """检查.txt: summary always distinguishes training vs evaluation."""
    status = str(result.get("status") or "unknown")
    result.setdefault("training_status", status)
    result.setdefault("training_valid_steps", result.get("valid_steps"))
    ev = result.get("eval") or {}
    if isinstance(ev, dict) and ev:
        result.setdefault("evaluation_status", ev.get("eval_status") or ev.get("status") or "ok")
        result.setdefault("eval_completed_steps", ev.get("valid_steps", ev.get("steps")))
        if "stage_c_passed" in result:
            result["formal_gate_passed"] = bool(result.get("stage_c_passed"))
        else:
            result["formal_gate_passed"] = bool(result.get("phase_e_allowed")) and not ev.get(
                "eval_failed"
            )
    else:
        result.setdefault("evaluation_status", "not_run")
        if "stage_c_passed" in result:
            result["formal_gate_passed"] = bool(result.get("stage_c_passed"))
        else:
            result.setdefault("formal_gate_passed", False)


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


def _is_finite_number(value: Any) -> bool:
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return False
    return fv == fv and abs(fv) != float("inf")


def compute_stage_c_gates(
    *,
    last_metrics: dict[str, Any] | None,
    step_log: list[dict[str, Any]] | None,
    collector_stats: dict[str, Any] | None,
    eval_result: dict[str, Any] | None,
    random_eval: dict[str, Any] | None,
    unserved_eps_mwh: float = 1e-3,
) -> dict[str, Any]:
    """检查.txt §25/§40.18 Stage C formal gates. CAES usage is not a gate.

    Returns:
        Dict with per-gate booleans and ``passed`` (all five must hold).
    """
    last_metrics = last_metrics or {}
    step_log = step_log or []
    stats = collector_stats or {}
    ev = eval_result or {}
    rnd = random_eval or {}

    finite = True
    sources: list[dict[str, Any]] = [last_metrics]
    sources.extend(row for row in step_log if isinstance(row, dict))
    for src in sources:
        for key in ("critic_loss", "actor_loss", "q1_mean", "q2_mean"):
            if key in src and src[key] is not None and not _is_finite_number(src[key]):
                finite = False
                break
        if not finite:
            break

    post = int(stats.get("post_step_hard_constraint_violation_count", 0) or 0)
    unsafe = int(stats.get("main_fmu_unsafe_execution_count", 0) or 0)
    eval_fmu = int(ev.get("fmu_failure_count", 0) or 0)
    fmu_hard_ok = post == 0 and unsafe == 0 and eval_fmu == 0

    eval_failed = bool(ev.get("eval_failed"))
    eval_status = str(ev.get("eval_status") or ("failed" if eval_failed else "ok"))
    no_safe_ok = eval_status != "failed_no_safe_action" and not (
        eval_failed and "no_safe" in eval_status
    )
    eval_steps = int(ev.get("valid_steps") or ev.get("steps") or 0)
    random_steps = int(rnd.get("valid_steps") or rnd.get("steps") or 0)
    complete_week = bool(no_safe_ok and eval_steps >= 168)
    random_complete = (not bool(rnd.get("eval_failed"))) and random_steps >= 168

    metrics = ev.get("metrics") or {}
    unserved = float(metrics.get("unserved_energy_mwh") or 0.0)
    unserved_ok = bool(complete_week and unserved <= float(unserved_eps_mwh))

    policy_cost = ev.get("weekly_raw_total_cost")
    random_cost = rnd.get("weekly_raw_total_cost")
    cost_better = False
    if complete_week and random_complete and policy_cost is not None and random_cost is not None:
        cost_better = float(policy_cost) < float(random_cost)

    passed = bool(finite and fmu_hard_ok and no_safe_ok and unserved_ok and cost_better)
    return {
        "c1_nan_inf_zero": finite,
        "c2_fmu_hard_zero": fmu_hard_ok,
        "c3_heldout_nosafe_zero": no_safe_ok,
        "c4_unserved_approx_zero": unserved_ok,
        "c5_cost_better_than_random": cost_better,
        "passed": passed,
        "policy_cost": policy_cost,
        "random_cost": random_cost,
        "unserved_energy_mwh": unserved,
        "unserved_eps_mwh": float(unserved_eps_mwh),
        "post_step_hard_constraint_violation_count": post,
        "main_fmu_unsafe_execution_count": unsafe,
        "eval_fmu_failure_count": eval_fmu,
        "eval_status": eval_status,
        "random_eval_status": rnd.get("eval_status"),
        "eval_valid_steps": eval_steps,
        "random_valid_steps": random_steps,
        "complete_week": complete_week,
    }


def run_hybrid_training(
    total_valid_steps: int = 5000,
    run_dir: str | Path = "runs/givesafe_td3_smoke",
    seed: int = 0,
    learning_starts: int = 1024,
    batch_size: int = 64,
    formal: bool = False,
    enable_shadow: bool | None = None,
    forecast_enabled: bool | None = None,
    annual_evaluation: bool = False,
    random_explore_start: float = 1.0,
    random_explore_end: float = 0.05,
    gradient_steps: int = 2,
    resume_from: str | Path | None = None,
    reset_critic_on_resume: bool = False,
    rule_demo_fraction: float = 0.0,
    soft_shell: bool | None = None,
    parameterized_caes: bool = True,
    use_dynamic_support: bool = True,
    forecast_mode: str | None = None,
    require_gas_swing: float | None = None,
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
        soft_shell: 训练/终评是否启用软约束外壳；None 时读 ``SOFT_SHELL`` 环境变量。

    Returns:
        含 status、stats、eval、formal_gate_blockers 等的 summary 字典。
    """
    use_soft_shell = _soft_shell_enabled(soft_shell)
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
        "paper_pc_hybrid_td3.yaml",
        "experiment_protocol.yaml",
    ):
        src = root / "src/config" / cfg_name
        if src.exists():
            shutil.copy2(src, run_dir / "config" / cfg_name)

    gs_cfg = load_givesafe_config(root / "src/config/givesafe_config.yaml")
    if gs_cfg.get("givesafe", {}).get("use_fallback", False):
        raise RuntimeError("禁止 use_fallback")
    replay_cfg = gs_cfg.get("replay_sampling") or {}
    paper_replay = dict(_paper_cfg(root).get("replay") or {})
    if paper_replay:
        replay_cfg = {**replay_cfg, **paper_replay}
    shadow_cfg = (gs_cfg.get("givesafe") or {}).get("shadow_validation") or {}
    use_shadow = bool(shadow_cfg.get("enabled", True)) if enable_shadow is None else bool(enable_shadow)

    np.random.seed(seed)
    try:
        env = PowerSystemEnv(
            require_complete_reward=formal,
            run_id=run_dir.name,
            forecast_enabled=forecast_enabled,
            forecast_mode=forecast_mode,
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
        physical_fraction=float(replay_cfg.get("physical_fraction", 1.0)),
        givesafe_fraction=float(replay_cfg.get("givesafe_fraction", 0.0)),
    )
    safety_dataset = SafetyDataset()
    collector = GiveSafeTransitionCollector(
        buffer, controller, shadow=shadow, safety_dataset=safety_dataset, soft_shell=use_soft_shell
    )
    rew_cfg = env.reward_calculator.config
    paper_algo = _paper_algo_cfg(root)
    agent = HybridTD3(
        obs_dim=int(np.prod(env.observation_space.shape)),
        gamma=float(paper_algo.get("gamma", rew_cfg.get("gamma", 0.99))),
        actor_lr=float(paper_algo.get("actor_lr", 3e-4)),
        critic_lr=float(paper_algo.get("critic_lr", 3e-4)),
        tau=float(paper_algo.get("tau", 0.005)),
        policy_delay=int(paper_algo.get("policy_delay", 2)),
        explore_noise=0.1,
        q_clip=float(paper_algo.get("q_clip", 200.0)),
        parameterized_caes=bool(parameterized_caes),
        use_dynamic_support=bool(use_dynamic_support),
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
        "soft_shell": use_soft_shell,
        "shadow_validation": shadow.capabilities() if shadow else {"enabled": False},
        "oracle_version": env.oracle.oracle_version,
        "git_commit": _git_commit(root),
        **{k: v for k, v in describe_fmu(env.fmu_path).items()},
        "annual_horizon_hours": env.config["fmu"].get("annual_horizon_hours"),
        "episode_start_schedule": "annual_cycling_windows",
        "forecast_enabled": env.forecast_enabled,
        "forecast_horizon_hours": env.forecast_provider.horizon_hours if env.forecast_provider else 0,
        "forecast_mode": forecast_mode or (env.forecast_provider.mode if env.forecast_provider else None),
        "observation_dim": int(np.prod(env.observation_space.shape)),
        "parameterized_caes": bool(agent.parameterized_caes),
        "use_dynamic_support": bool(agent.use_dynamic_support),
        "algorithm": "PC-HybridTD3",
        "training_recipe": {
            "random_explore_start": random_explore_start,
            "random_explore_end": random_explore_end,
            "gradient_steps": n_grad,
            "actor_prior": "no_idle_bias",
            "random_idle_bias": False,
            "rule_policy": rule_policy_name,
            "resume_from": resumed,
            "reset_critic_on_resume": bool(reset_critic_on_resume and resumed),
            "rule_demo_fraction": float(np.clip(rule_demo_fraction, 0.0, 1.0)),
            "soc_shaping": (rew_cfg.get("terminal_soc") or {}).get("shaping"),
        },
    }

    warmup_checked = False
    gas_soc_min: float | None = None
    gas_soc_max: float | None = None
    try:
        pbar = tqdm(total=total_valid_steps, desc="PC-HybridTD3", unit="step", dynamic_ncols=True)
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
                if valid_steps < learning_starts:
                    return random_policy.predict(obs, feasible=feasible)
                if np.random.rand() < eps:
                    return random_policy.predict(obs, feasible=feasible)
                return agent.select_action(obs, feasible, deterministic=False)

            obs, reward, terminated, truncated, info = collector.step_with_givesafe(
                env, propose, deterministic=False
            )
            if info.get("transition_type") == "physical" and info.get("transition_valid"):
                valid_steps += 1
                outputs = info.get("last_valid_outputs") or getattr(env, "last_outputs", None) or {}
                soc_g = outputs.get("caes_gas_soc")
                if soc_g is not None:
                    sg = float(soc_g)
                    gas_soc_min = sg if gas_soc_min is None else min(gas_soc_min, sg)
                    gas_soc_max = sg if gas_soc_max is None else max(gas_soc_max, sg)
                progress = valid_steps / max(total_valid_steps, 1)
                agent.explore_noise = float(0.1 * (1.0 - 0.6 * progress))
                agent.actor.gumbel_tau = float(max(0.2, 1.0 - 0.8 * progress))
                if (not warmup_checked) and valid_steps >= learning_starts:
                    counts = collector.stats.get("caes_mode_counts") or {}
                    n_d = int(counts.get(0, 0))
                    n_i = int(counts.get(1, 0))
                    n_c = int(counts.get(2, 0))
                    print(f"WARMUP modes discharge={n_d} idle={n_i} charge={n_c}", flush=True)
                    if n_d <= 100 or n_c <= 100:
                        raise RuntimeError(
                            f"warm-up CAES coverage failed: D={n_d} C={n_c} I={n_i} "
                            "(need D>100 and C>100 before gradient updates)"
                        )
                    warmup_checked = True
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
        swing = (
            None
            if gas_soc_min is None or gas_soc_max is None
            else float(gas_soc_max - gas_soc_min)
        )
        result["caes_gas_soc_range_train"] = swing
        result["stage_b_interaction"] = "passed"
        if require_gas_swing is not None and (swing is None or swing <= float(require_gas_swing)):
            result["stage_b_interaction"] = "failed"
            result["error"] = f"ΔSOC_gas={swing} (need > {require_gas_swing})"
        agent.save(run_dir / "checkpoints" / "hybrid_givesafe_td3.pt")
        safety_dataset.save(run_dir / "train" / "safety_dataset.json")

        eval_opts = {"start_time": eval_start_seconds(env.config["fmu"])}
        rule_env = PowerSystemEnv(
            run_id=f"{run_dir.name}_rule",
            forecast_enabled=forecast_enabled,
            forecast_mode=forecast_mode,
        )
        rule_result = evaluate_policy(
            rule_env,
            PriceAwareRuleController(rule_env),
            run_dir / "trajectories" / "rule.csv",
            reset_options=eval_opts,
        )
        rule_env.close()

        eval_env = PowerSystemEnv(
            run_id=f"{run_dir.name}_eval",
            forecast_enabled=forecast_enabled,
            forecast_mode=forecast_mode,
        )
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
        eval_policy = HybridPolicyWrapper(
            agent, eval_env, eval_ctrl, deterministic=True, soft_shell=use_soft_shell
        )
        try:
            eval_result = evaluate_policy(
                eval_env,
                eval_policy,
                run_dir / "trajectories" / "eval.csv",
                reset_options=eval_opts,
                soft_shell=use_soft_shell,
            )
        finally:
            if eval_shadow is not None:
                eval_shadow.close()
            eval_env.close()
        result["eval_start_time_seconds"] = eval_opts["start_time"]
        eval_steps = int(eval_result.get("valid_steps") or eval_result.get("steps") or 0)
        fmu_fail = int(eval_result.get("fmu_failure_count") or 0)
        greedy_ok = (
            not bool(eval_result.get("eval_failed"))
            and fmu_fail == 0
            and eval_steps >= 168
        )
        if not greedy_ok:
            result["stage_b_greedy_eval"] = "failed"
            result["greedy_eval"] = "failed"
            if result.get("stage_b_interaction") == "passed":
                result["status"] = "partial_pass"
            fail = eval_result.get("failure") or {
                "eval_status": eval_result.get("eval_status"),
                "valid_steps": eval_steps,
                "fmu_failure_count": fmu_fail,
            }
            (run_dir / "eval_failure.json").write_text(
                json.dumps(fail, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        else:
            result["stage_b_greedy_eval"] = "passed"
            result["greedy_eval"] = "passed"
        random_eval_result: dict[str, Any] | None = None
        try:
            rnd_env = PowerSystemEnv(
                run_id=f"{run_dir.name}_random",
                forecast_enabled=forecast_enabled,
                forecast_mode=forecast_mode,
            )
            try:
                random_eval_result = evaluate_policy(
                    rnd_env,
                    RandomFeasiblePolicy(rnd_env),
                    run_dir / "trajectories" / "random.csv",
                    reset_options=eval_opts,
                    deterministic=False,
                )
            finally:
                rnd_env.close()
        except Exception as exc:  # noqa: BLE001
            random_eval_result = {
                "eval_status": "failed",
                "eval_failed": True,
                "error": str(exc),
            }
        result["random"] = random_eval_result

        last_m = agent.last_metrics or {}
        counts = (collector.stats or {}).get("caes_mode_counts") or {}
        n_d = int(counts.get(0, 0) or 0)
        n_c = int(counts.get(2, 0) or 0)
        stage_c = compute_stage_c_gates(
            last_metrics=last_m,
            step_log=step_log,
            collector_stats=collector.stats or {},
            eval_result=eval_result,
            random_eval=random_eval_result,
        )
        stage_c["c1_learning_stability"] = stage_c["c1_nan_inf_zero"]
        stage_c["c2_greedy_deployability"] = stage_c["c3_heldout_nosafe_zero"]
        stage_c["storage_effectiveness_diagnostic"] = bool(
            (swing is not None and swing > 0.05) and n_d > 0 and n_c > 0
        )
        stage_c["caes_mode_counts"] = {
            "D": n_d,
            "I": int(counts.get(1, 0) or 0),
            "C": n_c,
        }
        stage_c["gas_soc_range"] = swing
        stage_c["device_diagnostic_note"] = (
            "CAES idle is not a Stage C gate; compare with rolling MILP."
        )
        result["stage_c_gates"] = stage_c
        result["stage_c_passed"] = bool(stage_c.get("passed"))

        annual_eval_result = None
        if annual_evaluation:
            annual_env = PowerSystemEnv(
                run_id=f"{run_dir.name}_annual_eval",
                forecast_enabled=forecast_enabled,
                forecast_mode=forecast_mode,
            )
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
            annual_policy = HybridPolicyWrapper(
                agent, annual_env, annual_ctrl, deterministic=True, soft_shell=use_soft_shell
            )
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
                "numerical_endpoint_snap_count": int(
                    collector.stats.get("numerical_endpoint_snap_count", 0) or 0
                ),
                "max_endpoint_snap_abs": float(
                    collector.stats.get("max_endpoint_snap_abs", 0.0) or 0.0
                ),
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
    except Exception as exc:
        if str(result.get("status") or "") in ("running", "", "unknown"):
            result["status"] = "training_failed"
        result["error"] = repr(exc)
        result["training_status"] = "failed"
        raise
    finally:
        if shadow is not None:
            shadow.close()
        try:
            env.close()
        except Exception:
            pass
        try:
            (run_dir / "train").mkdir(parents=True, exist_ok=True)
            (run_dir / "train" / "step_log.json").write_text(
                json.dumps(step_log, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            result.setdefault("algo", "hybrid_givesafe_td3")
            _stamp_summary_status(result)
            (run_dir / "summary.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
        except Exception:
            pass

    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "step_log.json").write_text(
        json.dumps(step_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result.setdefault("algo", "hybrid_givesafe_td3")
    _stamp_summary_status(result)
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


def run_td3_scratch(total_valid_steps: int = 35000, **kwargs) -> dict[str, Any]:
    """论文典型单层 TD3：从零训练，无规则 BC 示范主导，GiveSafe 开启。

    与 Hybrid-BC→RL 强教师路径区分：默认 rule_demo_fraction=0，
    较高早期随机探索，run 目录命名 td3_scratch。

    Args:
        total_valid_steps: 有效步数目标。
        **kwargs: 传给 ``run_hybrid_training`` 的额外参数。

    Returns:
        训练 summary 字典。
    """
    kwargs.setdefault("rule_demo_fraction", 0.0)
    kwargs.setdefault("random_explore_start", 1.0)
    kwargs.setdefault("random_explore_end", 0.05)
    kwargs.setdefault("enable_shadow", False)
    return run_hybrid_training(
        total_valid_steps=total_valid_steps,
        run_dir=kwargs.pop("run_dir", "runs/td3_scratch"),
        formal=False,
        **kwargs,
    )
