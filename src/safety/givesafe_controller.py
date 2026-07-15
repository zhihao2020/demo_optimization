"""GiveSafe 控制器：同状态重采样，禁止任何 fallback。"""

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
    root = Path(__file__).resolve().parents[2]
    p = Path(path) if path else root / "src" / "config" / "givesafe_config.yaml"
    with Path(p).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class GiveSafeController:
    """策略提出动作 → 两级安全检查 → 拒绝则自环记录并重采样；禁止 fallback。"""

    def __init__(
        self,
        oracle: FeasibilityOracle | None = None,
        shadow: ShadowFmuValidator | None = None,
        config: Mapping[str, Any] | None = None,
        config_path: str | Path | None = None,
    ):
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
        """policy_sample_fn: 无参，在当前状态采样候选动作 dict。"""
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
