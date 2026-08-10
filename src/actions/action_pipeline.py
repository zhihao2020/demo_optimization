"""安全动作生成流水线(SafeActionGenerator)：无 fallback，拒绝时仅重采样。"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from envs.failures import FeasibleSetEmpty

from .feasible_set import DynamicFeasibleActionSet
from .feasibility_oracle import FeasibilityOracle
from .safety_classifier import SafetyClassifier
from .types import PhysicalFmuAction
from .validator import physical_from_dict


class SafeActionGenerator:
    """安全动作生成器：拒绝时仅重采样，耗尽则抛出 FeasibleSetEmpty。"""

    def __init__(
        self,
        oracle: FeasibilityOracle,
        classifier: SafetyClassifier | None = None,
        *,
        safety_threshold: float | None = None,
        max_resamples: int = 32,
    ):
        self.oracle = oracle
        self.classifier = classifier
        if classifier is not None and safety_threshold is not None:
            classifier.threshold = float(safety_threshold)
        self.max_resamples = int(max_resamples)

    def generate(
        self,
        outputs: Mapping[str, float],
        previous_thermal_w: float,
        propose_fn: Callable[[DynamicFeasibleActionSet], dict | PhysicalFmuAction],
        *,
        deterministic: bool = False,
        feasible_override: DynamicFeasibleActionSet | None = None,
    ) -> tuple[dict, dict[str, Any]]:
        """在动态可行域内重采样直至 Oracle 可执行且（可选）分类器安全。"""
        _ = deterministic
        feasible = feasible_override or self.oracle.compute(outputs, previous_thermal_w)
        if self.oracle.is_feasible_set_empty(feasible):
            raise FeasibleSetEmpty("可行域为空：无法生成合法动作")
        physical_dist, safe_dist = self.oracle.distances_to_bounds(outputs)

        meta: dict[str, Any] = {
            "oracle_version": self.oracle.oracle_version,
            "safety_model_version": (
                None if self.classifier is None else self.classifier.model_version
            ),
            "safety_threshold": (
                None if self.classifier is None else self.classifier.threshold
            ),
            "safety_probability": None,
            "distance_to_physical_boundary": physical_dist,
            "distance_to_safe_boundary": safe_dist,
            "resample_count": 0,
            "feasible_action_spec": feasible.as_dict(),
            "fallback": None,
        }
        last_action: dict | None = None
        for i in range(self.max_resamples):
            raw = propose_fn(feasible)
            if isinstance(raw, PhysicalFmuAction):
                action = raw.as_env_dict()
            else:
                action = raw
            last_action = action
            physical = physical_from_dict(action)
            ok, reason = self.oracle.check_action_executable(
                physical, outputs, feasible, previous_thermal_w
            )
            meta["resample_count"] = i + 1
            if not ok:
                meta["last_reject_reason"] = reason
                continue
            if self.classifier is None:
                predicted = self.oracle.predict_next_state(
                    outputs, physical, previous_thermal_w
                )
                meta["oracle_predicted_next_state"] = {
                    k: predicted[k]
                    for k in predicted
                    if k not in ("caes_mode",)
                }
                meta["safety_probability"] = 1.0
                return action, meta
            is_safe, p = self.classifier.is_safe(outputs, action, physical_dist)
            meta["safety_probability"] = p
            if is_safe:
                predicted = self.oracle.predict_next_state(
                    outputs, physical, previous_thermal_w
                )
                meta["oracle_predicted_next_state"] = {
                    k: predicted[k]
                    for k in predicted
                    if k not in ("caes_mode",)
                }
                return action, meta
        raise FeasibleSetEmpty(
            f"在 {self.max_resamples} 次重采样后仍无安全动作（无 fallback）"
            + (f"；last={last_action}" if last_action else "")
        )
