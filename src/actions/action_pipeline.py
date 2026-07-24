"""安全动作生成流水线(SafeActionGenerator)：无 fallback，拒绝时仅重采样。"""

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
    """安全动作生成器(SafeActionGenerator)：拒绝时仅重采样，耗尽则抛出 FeasibleSetEmpty。"""

    def __init__(
        self,
        oracle: FeasibilityOracle,
        classifier: SafetyClassifier | None = None,
        *,
        safety_threshold: float | None = None,
        max_resamples: int = 32,
    ):
        """初始化安全动作生成器。

        Args:
            oracle: 可行性神谕(FeasibilityOracle)，根据物理观测、动态可行域与 CAES 最短运行约束计算可行域。
            classifier: 安全性分类器(SafetyClassifier)，可选；为 None 时仅依赖 Oracle 预检。
            safety_threshold: 安全概率阈值；仅当 classifier 非 None 时写入其 threshold。
            max_resamples: 动作被拒绝后的最大重采样次数。
        """
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
        """在动态可行域内重采样直至得到 Oracle 可执行且（可选）分类器判定的安全动作。

        Args:
            outputs: 当前环境观测(FMU 输出)字典。
            previous_thermal_w: 上一决策步 FMU 实际火电功率 p_thermal（W）。
            propose_fn: 提案函数，输入动态可行域(DynamicFeasibleActionSet)，返回动作 dict 或 HybridAction。
            deterministic: 是否要求提案函数使用确定性策略（由 propose_fn 自行解释）。
            feasible_override: 可选，覆盖 Oracle 计算的动态可行域。

        Returns:
            (action, meta)：动作字典与审计元数据（含 resample_count、safety_probability、距离界等）。

        Raises:
            FeasibleSetEmpty: 可行域为空，或在 max_resamples 次重采样后仍无安全动作（无 fallback）。
        """
        # 计算可行域
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
            action = (
                raw
                if isinstance(raw, dict)
                else {
                    "u_tp": np.asarray([raw.u_tp], dtype=np.float32),
                    "u_battery": np.asarray([raw.u_battery], dtype=np.float32),
                    "caes_mode": int(raw.caes_mode),
                    "caes_magnitude": np.asarray(
                        [raw.caes_magnitude], dtype=np.float32
                    ),
                }
            )
            last_action = action
            hybrid = hybrid_from_dict(action)  # 将动作转换为混合动作，方便后续检查
            ok, reason = self.oracle.check_action_executable(
                hybrid, outputs, feasible, previous_thermal_w
            )  # 检查动作是否可执行
            meta["resample_count"] = i + 1
            if not ok:
                meta["last_reject_reason"] = reason  # 记录拒绝原因
                continue
            if self.classifier is None:
                predicted = self.oracle.predict_next_state(
                    outputs, hybrid, previous_thermal_w
                )  # 预测下一个状态
                meta["oracle_predicted_next_state"] = {
                    k: predicted[k]
                    for k in predicted
                    if k not in ("caes_mode", "caes_magnitude")
                }
                meta["safety_probability"] = 1.0
                return action, meta
            is_safe, p = self.classifier.is_safe(outputs, action, physical_dist)
            meta["safety_probability"] = p
            if is_safe:
                predicted = self.oracle.predict_next_state(
                    outputs, hybrid, previous_thermal_w
                )
                meta["oracle_predicted_next_state"] = {
                    k: predicted[k]
                    for k in predicted
                    if k not in ("caes_mode", "caes_magnitude")
                }
                return action, meta
        raise FeasibleSetEmpty(
            f"在 {self.max_resamples} 次重采样后仍无安全动作（无 fallback）"
            + (f"；last={last_action}" if last_action else "")
        )

