"""安全给予控制器(GiveSafeController)：同状态重采样，禁止任何 fallback。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import yaml

from actions import DynamicFeasibleActionSet, FeasibilityOracle, HybridActionValidator
from actions.validator import hybrid_from_dict

from .constraint_checker import GiveSafeConstraintChecker
from .constraint_reward import ConstraintRewardCalculator
from .no_safe_action import NoSafeActionFoundError
from .safety_result import GiveSafeResult, SafetyCheckResult
from .shadow_fmu_validator import ShadowFmuValidator


def load_givesafe_config(path: str | Path | None = None) -> dict[str, Any]:
    """从 YAML 文件加载安全给予(GiveSafe) 配置。

    Args:
        path: 配置文件路径；为 None 时使用项目默认 givesafe_config.yaml。

    Returns:
        解析后的配置字典。

    Raises:
        OSError: 配置文件不存在或无法读取时。
        yaml.YAMLError: YAML 格式无效时。
    """
    root = Path(__file__).resolve().parents[2]
    p = Path(path) if path else root / "src" / "config" / "givesafe_config.yaml"
    with Path(p).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class GiveSafeController:
    """安全给予控制器(GiveSafeController)：策略提出动作 → 两级安全检查 → 拒绝则自环记录并重采样；禁止 fallback。"""

    def __init__(
        self,
        oracle: FeasibilityOracle | None = None,
        shadow: ShadowFmuValidator | None = None,
        config: Mapping[str, Any] | None = None,
        config_path: str | Path | None = None,
    ):
        """初始化控制器、约束检查器与可选影子仿真校验器。

        Args:
            oracle: 可行性神谕(FeasibilityOracle)；为 None 时从项目根加载默认实例。
            shadow: 可选影子仿真校验器(ShadowFmuValidator)。
            config: 内联配置字典；与 config_path 二选一或合并使用。
            config_path: YAML 配置文件路径。

        Returns:
            无。

        Raises:
            RuntimeError: 配置中 use_fallback 为 true 时（安全给予禁止 fallback）。
        """
        full = dict(config) if config is not None else load_givesafe_config(config_path)
        if "givesafe" in full:
            self.cfg = dict(full["givesafe"])
            self.full_config = full
        else:
            self.cfg = dict(full)
            self.full_config = {"givesafe": self.cfg}
        if self.cfg.get("use_fallback", False):
            raise RuntimeError("GiveSafe 禁止 use_fallback=true")
        self.oracle = oracle or FeasibilityOracle.from_root()
        self.checker = GiveSafeConstraintChecker(self.oracle)
        self.shadow = shadow
        self.reward_calc = ConstraintRewardCalculator(self.cfg.get("constraint_reward"))
        self.max_attempts = int(self.cfg.get("max_attempts_per_env_step", 64))

    def select_safe_action(
        self,
        observation_outputs: Mapping[str, float],
        previous_thermal_w: float,
        policy_sample_fn: Callable[[], dict],
        *,
        deterministic: bool = False,
        on_rejection: Callable[[dict, SafetyCheckResult, dict[str, float]], None] | None = None,
        feasible_override: DynamicFeasibleActionSet | None = None,
    ) -> GiveSafeResult:
        """在当前状态下反复采样候选动作直至通过两级安全检查或耗尽尝试次数。

        Args:
            observation_outputs: 当前环境观测输出。
            previous_thermal_w: 上一时刻热功率（W）。
            policy_sample_fn: 无参回调，返回策略采样的候选动作 dict。
            deterministic: 预留确定性采样标志（当前未改变采样逻辑）。
            on_rejection: 每次拒绝时的可选回调 (proposed, safety, reward_terms)。
            feasible_override: 可选动态可行集，用于 CAES 最小运行等额外 Oracle 校验。

        Returns:
            含 safe_action 与完整重采样轨迹的 GiveSafeResult。

        Raises:
            NoSafeActionFoundError: 在 max_attempts 次内未找到安全动作时。
        """
        result = GiveSafeResult(safe_action=None, oracle_version=self.oracle.oracle_version)
        for attempt in range(self.max_attempts):
            proposed = policy_sample_fn()
            result.proposed_actions.append(proposed)
            result.attempt_count = attempt + 1
            if feasible_override is not None:
                try:
                    HybridActionValidator().validate(hybrid_from_dict(proposed), feasible_override)
                except Exception as exc:
                    level1 = SafetyCheckResult(
                        safe=False,
                        rejection_stage="oracle",
                        violation_type="forbidden_mode",
                        violation_severity=1.0,
                        normalized_violations={"forbidden_mode": 1.0},
                        mode_mask=feasible_override.mode_mask.as_dict(),
                        oracle_safe=False,
                        oracle_rejection_reason=str(exc),
                        metadata={"oracle_version": self.oracle.oracle_version, "caes_min_run": True},
                    )
                else:
                    level1 = self.checker.check(observation_outputs, proposed, previous_thermal_w)
            else:
                level1 = self.checker.check(observation_outputs, proposed, previous_thermal_w)
            safety = level1
            if level1.safe and self.shadow is not None:
                safety = self.shadow.validate(proposed, level1)
            result.safety_check_metadata.append(safety)
            if not safety.safe:
                terms = self.reward_calc.calculate(safety)
                result.rejected_actions.append(proposed)
                result.rejection_reasons.append(
                    f"{safety.rejection_stage}:{safety.violation_type}:{safety.oracle_rejection_reason or safety.shadow_failure_reason}"
                )
                result.constraint_rewards.append(float(terms["constraint_reward"]))
                if on_rejection is not None:
                    on_rejection(proposed, safety, terms)
                continue
            result.safe_action = proposed
            return result
        result.no_safe_action = True
        raise NoSafeActionFoundError(
            f"在 {self.max_attempts} 次尝试后仍无安全动作（无 fallback）",
            attempts=self.max_attempts,
            rejected=list(result.rejected_actions),
        )
