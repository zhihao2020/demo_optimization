"""统一失败分类：硬约束与环境异常，不进入经济 reward。"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


class ConstraintFailure(Exception):
    """约束/环境失败基类(ConstraintFailure)。

    所有预检、后验与 FMU 异常均继承此类；不触发经济 reward。
    """

    failure_type: str = "ConstraintFailure"

    def __init__(
        self,
        reason: str,
        *,
        fine_type: str | None = None,
        triggering_constraint: str | None = None,
    ) -> None:
        """构造失败异常。

        Args:
            reason: 人类可读失败原因。
            fine_type: 细粒度失败类型，默认 ``unknown``。
            triggering_constraint: 触发的约束名，默认同 ``fine_type``。
        """
        self.reason = reason
        self.fine_type = fine_type or "unknown"
        self.triggering_constraint = triggering_constraint or self.fine_type
        super().__init__(f"[{self.failure_type}/{self.fine_type}] {reason}")


class StaticActionViolation(ConstraintFailure):
    """静态动作违反(StaticActionViolation)：形状、类型或静态边界错误。"""

    failure_type = "StaticActionViolation"


class ForbiddenModeViolation(ConstraintFailure):
    """禁止模式违反(ForbiddenModeViolation)：CAES 模式不在允许集合内。"""

    failure_type = "ForbiddenModeViolation"


class DynamicStateConstraintViolation(ConstraintFailure):
    """动态状态约束违反(DynamicStateConstraintViolation)：Oracle 预检不可执行。"""

    failure_type = "DynamicStateConstraintViolation"


class PostStepHardConstraintViolation(ConstraintFailure):
    """步后硬约束违反(PostStepHardConstraintViolation)：FMU 输出越物理/安全界。"""

    failure_type = "PostStepHardConstraintViolation"


class FmuNumericalFailure(ConstraintFailure):
    """FMU 数值失败(FmuNumericalFailure)：求解器或 doStep 数值异常。"""

    failure_type = "FmuNumericalFailure"


class FmiLifecycleFailure(ConstraintFailure):
    """FMI 生命周期失败(FmiLifecycleFailure)：实例化、reset 或 terminate 失败。"""

    failure_type = "FmiLifecycleFailure"


class NonFiniteOutputFailure(ConstraintFailure):
    """非有限输出失败(NonFiniteOutputFailure)：FMU 输出含 NaN/Inf。"""

    failure_type = "NonFiniteOutputFailure"


class FeasibleSetEmpty(ConstraintFailure):
    """可行集为空(FeasibleSetEmpty)。

    动态可行集为空时中止 episode，不伪造合法动作。
    """

    failure_type = "FeasibleSetEmpty"


# 细粒度后验失败类型（不是仅 PostStepHardConstraintViolation）
FINE_FAILURE_TYPES = (
    "battery_soc_high",
    "battery_soc_low",
    "caes_gas_soc_high",
    "caes_gas_soc_low",
    "caes_hot_soc_high",
    "caes_hot_soc_low",
    "caes_cold_soc_high",
    "caes_cold_soc_low",
    "caes_pressure_high",
    "caes_pressure_low",
    "caes_temperature_high",
    "caes_temperature_low",
    "thermal_ramp_violation",
    "grid_capacity_violation",
    "nonfinite_output",
    "nonlinear_solver_failure",
    "feasible_set_empty",
    "unknown",
)


@dataclass
class FailureRecord:
    """结构化失败记录(FailureRecord)。

    供审计、SafetyDataset 与离线分析；涵盖 Oracle 预测与实际 FMU 输出对比。
    """

    run_id: str
    episode: int
    step: int
    simulation_time: float
    failure_type: str
    fine_failure_type: str
    triggering_constraint: str
    previous_observation: dict[str, float] | None = None
    hybrid_action: dict[str, Any] | None = None
    decoded_fmu_action: dict[str, float] | None = None
    oracle_dynamic_bounds: dict[str, float] | None = None
    oracle_mode_mask: dict[str, bool] | None = None
    oracle_predicted_next_state: dict[str, float] | None = None
    actual_fmu_outputs: dict[str, float] | None = None
    last_valid_state: dict[str, float] | None = None
    distance_to_physical_boundary: dict[str, float] | None = None
    distance_to_safe_boundary: dict[str, float] | None = None
    residuals: dict[str, float] | None = None
    dangerous_residual: dict[str, float] | None = None
    fmu_status: str | None = None
    modelica_assert_message: str | None = None
    oracle_version: str | None = None
    safety_probability: float | None = None
    safety_threshold: float | None = None
    safety_model_version: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转为可序列化字典。

        Returns:
            与 dataclass 字段一一对应的 ``dict``。
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FailureRecord":
        """从字典反序列化，忽略未知键。

        Args:
            data: 含 FailureRecord 字段的映射。

        Returns:
            重建的 FailureRecord 实例。
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        payload = {k: v for k, v in data.items() if k in known}
        return cls(**payload)  # type: ignore[arg-type]
