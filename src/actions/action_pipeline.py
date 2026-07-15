"""SafeActionGenerator：无 fallback。拒绝时仅重采样，耗尽则 FeasibleSetEmpty。

GiveSafe 主路径请用 safety.GiveSafeController（会写入自环样本）。
本类保留给边界工具等只需要动作过滤、不需要 replay 写入的场景。
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from envs.failures import FeasibleSetEmpty

from .feasible_set import DynamicFeasibleActionSet
from .feasibility_oracle import FeasibilityOracle
from .safety_classifier import SafetyClassifier
from .types import HybridAction
from .validator import hybrid_from_dict


class SafeActionGenerator:
    """Actor -> ModeMask/FeasibleSet ->（可选）SafetyClassifier。禁止 idle/rule fallback。"""

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
        propose_fn: Callable[[DynamicFeasibleActionSet], dict | HybridAction],
        *,
        deterministic: bool = False,
        feasible_override: DynamicFeasibleActionSet | None = None,
    ) -> tuple[dict, dict[str, Any]]:
        # 环境可传入其已叠加 CAES 最短运行规则的可行域；否则保持原 Oracle 行为。
        feasible = feasible_override or self.oracle.compute(outputs, previous_thermal_w)
        if self.oracle.is_feasible_set_empty(feasible):
            raise FeasibleSetEmpty("DynamicFeasibleActionSet 为空：无法生成合法动作")
        physical_dist, safe_dist = self.oracle.distances_to_bounds(outputs)
        meta: dict[str, Any] = {
            "oracle_version": self.oracle.oracle_version,
            "safety_model_version": None if self.classifier is None else self.classifier.model_version,
            "safety_threshold": None if self.classifier is None else self.classifier.threshold,
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
            action = raw if isinstance(raw, dict) else {
                "u_tp": np.asarray([raw.u_tp], dtype=np.float32),
                "u_battery": np.asarray([raw.u_battery], dtype=np.float32),
                "caes_mode": int(raw.caes_mode),
                "caes_magnitude": np.asarray([raw.caes_magnitude], dtype=np.float32),
            }
            last_action = action
            hybrid = hybrid_from_dict(action)
            ok, reason = self.oracle.check_action_executable(hybrid, outputs, feasible, previous_thermal_w)
            meta["resample_count"] = i + 1
            if not ok:
                meta["last_reject_reason"] = reason
                continue
            if self.classifier is None:
                predicted = self.oracle.predict_next_state(outputs, hybrid, previous_thermal_w)
                meta["oracle_predicted_next_state"] = {
                    k: predicted[k] for k in predicted if k not in ("caes_mode", "caes_magnitude")
                }
                meta["safety_probability"] = 1.0
                return action, meta
            is_safe, p = self.classifier.is_safe(outputs, action, physical_dist)
            meta["safety_probability"] = p
            if is_safe:
                predicted = self.oracle.predict_next_state(outputs, hybrid, previous_thermal_w)
                meta["oracle_predicted_next_state"] = {
                    k: predicted[k] for k in predicted if k not in ("caes_mode", "caes_magnitude")
                }
                return action, meta
        raise FeasibleSetEmpty(
            f"在 {self.max_resamples} 次重采样后仍无安全动作（无 fallback）"
            + (f"；last={last_action}" if last_action else "")
        )
