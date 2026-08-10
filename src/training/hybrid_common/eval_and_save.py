"""训练后规则/策略评估、summary 落盘与可读报告。"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Callable

from controllers.rule_based_controller import RuleBasedController
from envs.power_system_env import PowerSystemEnv
from fmu import FmuAdapter
from safety import GiveSafeController, ShadowFmuValidator
from training.episode_starts import eval_start_seconds
from training.evaluate_td3 import evaluate_annual_policy, evaluate_policy

from .policy_wrapper import HybridGiveSafePolicyWrapper

logger = logging.getLogger(__name__)


def prepare_run_dir(run_dir: Path, root: Path) -> None:
    """创建运行目录结构并复制配置文件快照。

    Args:
        run_dir: 本次训练运行根目录。
        root: 项目根目录，用于定位 ``src/config``。

    Returns:
        无。
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("config", "train", "checkpoints", "trajectories"):
        (run_dir / name).mkdir(exist_ok=True)
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


def finalize_training_run(
    *,
    run_dir: Path,
    agent: Any,
    checkpoint_name: str,
    gs_cfg: dict,
    use_shadow: bool,
    forecast_enabled: bool | None,
    annual_evaluation: bool,
    result: dict[str, Any],
    step_log: list[dict],
    collector_stats: dict[str, Any] | None = None,
    extra_result: dict[str, Any] | None = None,
    make_shadow: Callable[..., ShadowFmuValidator] | None = None,
) -> dict[str, Any]:
    """规则评估 + GiveSafe 策略评估 + 写 summary + 生成可读报告。

    Args:
        run_dir: 训练运行目录。
        agent: 需实现 ``save`` 与 ``select_action`` 的智能体。
        checkpoint_name: 检查点文件名。
        gs_cfg: GiveSafe 配置字典。
        use_shadow: 评估时是否启用影子 FMU 校验。
        forecast_enabled: 评估环境是否启用预测。
        annual_evaluation: 是否执行全年滑动窗口评估。
        result: 训练过程已收集的结果字典（会被更新）。
        step_log: 训练步日志列表。
        collector_stats: 可选收集器统计，用于拒绝率等指标。
        extra_result: 额外合并进 summary 的字段。
        make_shadow: 可选自定义影子校验器工厂。

    Returns:
        更新后的完整 result 字典。
    """
    run_dir = Path(run_dir)
    agent.save(run_dir / "checkpoints" / checkpoint_name)

    # Temporary env only to read fmu config for eval start; closed immediately.
    _cfg_env = PowerSystemEnv(run_id=f"{run_dir.name}_cfg", forecast_enabled=forecast_enabled)
    eval_opts = {"start_time": eval_start_seconds(_cfg_env.config["fmu"])}
    _cfg_env.close()

    rule_env = PowerSystemEnv(run_id=f"{run_dir.name}_rule", forecast_enabled=forecast_enabled)
    rule_result = evaluate_policy(
        rule_env,
        RuleBasedController(rule_env),
        run_dir / "trajectories" / "rule.csv",
        reset_options=eval_opts,
    )
    rule_env.close()

    eval_env = PowerSystemEnv(run_id=f"{run_dir.name}_eval", forecast_enabled=forecast_enabled)
    eval_shadow = None
    if use_shadow:
        fmu_path = eval_env.root / eval_env.config["fmu"]["path"]
        step = float(eval_env.config["fmu"]["communication_step_seconds"])

        def efactory():
            """构造评估用功能模型单元适配器(FmuAdapter)。"""
            return FmuAdapter(fmu_path, step, eval_env.registry)

        if make_shadow is not None:
            eval_shadow = make_shadow(eval_env, efactory)
        else:
            shadow_cfg = (gs_cfg.get("givesafe") or {}).get("shadow_validation") or {}
            eval_shadow = ShadowFmuValidator(
                factory=efactory,
                oracle=eval_env.oracle,
                enabled=True,
                mode=str(shadow_cfg.get("mode", "always")),
            )
    eval_ctrl = GiveSafeController(oracle=eval_env.oracle, shadow=eval_shadow, config=gs_cfg)
    eval_policy = HybridGiveSafePolicyWrapper(agent, eval_env, eval_ctrl, deterministic=True)
    try:
        eval_result = evaluate_policy(
            eval_env,
            eval_policy,
            run_dir / "trajectories" / "eval.csv",
            reset_options=eval_opts,
        )
    finally:
        if eval_shadow is not None:
            eval_shadow.close()
        eval_env.close()
    result["eval_start_time_seconds"] = eval_opts["start_time"]

    annual_eval_result = None
    if annual_evaluation:
        annual_env = PowerSystemEnv(
            run_id=f"{run_dir.name}_annual_eval", forecast_enabled=forecast_enabled
        )
        annual_shadow = None
        if use_shadow:
            fmu_path = annual_env.root / annual_env.config["fmu"]["path"]
            step = float(annual_env.config["fmu"]["communication_step_seconds"])

            def afactory():
                """构造年评估用功能模型单元适配器(FmuAdapter)。"""
                return FmuAdapter(fmu_path, step, annual_env.registry)

            shadow_cfg = (gs_cfg.get("givesafe") or {}).get("shadow_validation") or {}
            annual_shadow = ShadowFmuValidator(
                factory=afactory,
                oracle=annual_env.oracle,
                enabled=True,
                mode=str(shadow_cfg.get("mode", "always")),
            )
        annual_ctrl = GiveSafeController(
            oracle=annual_env.oracle, shadow=annual_shadow, config=gs_cfg
        )
        annual_policy = HybridGiveSafePolicyWrapper(
            agent, annual_env, annual_ctrl, deterministic=True
        )
        try:
            annual_eval_result = evaluate_annual_policy(
                annual_env,
                annual_policy,
                annual_horizon_hours=int(annual_env.config["fmu"]["annual_horizon_hours"]),
                output_dir=run_dir / "trajectories" / "annual_eval",
            )
        finally:
            if annual_shadow is not None:
                annual_shadow.close()
            annual_env.close()

    result = dict(result)
    result.update(
        {
            "eval": eval_result,
            "annual_eval": annual_eval_result,
            "rule": rule_result,
            "last_metrics": getattr(agent, "last_metrics", {}),
        }
    )
    if collector_stats is not None:
        attempts = max(collector_stats.get("policy_attempt_count", 1), 1)
        rej = collector_stats.get("givesafe_rejection_count", 0)
        main_exec = max(collector_stats.get("main_fmu_execution_count", 1), 1)
        post = collector_stats.get("post_step_hard_constraint_violation_count", 0)
        result.update(
            {
                "stats": collector_stats,
                "proposal_rejection_rate": rej / attempts,
                "false_safe_rate": collector_stats.get("givesafe_false_safe_count", 0) / main_exec,
                "main_fmu_execution_safety_rate": 1.0 - post / main_exec,
            }
        )
    if extra_result:
        result.update(extra_result)

    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "step_log.json").write_text(
        json.dumps(step_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return result


def write_summary_and_report(
    run_dir: Path, result: dict[str, Any], step_log: list[dict] | None = None
) -> dict[str, Any]:
    """最终 summary 落盘并生成可读报告。

    Args:
        run_dir: 训练运行目录。
        result: 待写入的 summary 内容。
        step_log: 可选训练步日志，写入 train/step_log.json（完整周期采样，不截断尾窗）。

    Returns:
        可能含 ``report_path`` 或 ``report_error`` 的 result 字典。
    """
    run_dir = Path(run_dir)
    if step_log is not None:
        (run_dir / "train").mkdir(parents=True, exist_ok=True)
        (run_dir / "train" / "step_log.json").write_text(
            json.dumps(step_log, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (run_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    try:
        from training.report_policy_run import generate_policy_report

        report_path = generate_policy_report(run_dir)
        result["report_path"] = str(Path(report_path).as_posix())
    except Exception as exc:  # noqa: BLE001
        logger.warning("策略报告生成失败: %s", exc)
        result["report_error"] = str(exc)
    (run_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return result
