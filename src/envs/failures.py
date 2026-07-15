"""统一失败分类：硬约束与环境异常，不进入经济 reward。"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping
class ConstraintFailure(Exception):
    """基础约束/环境失败。"""
    failure_type: str = "ConstraintFailure"
    def __init__(self, reason: str, *, fine_type: str | None = None, triggering_constraint: str | None = None) -> None:
        self.reason = reason
        self.fine_type = fine_type or "unknown"
        self.triggering_constraint = triggering_constraint or self.fine_type
        super().__init__(f"[{self.failure_type}/{self.fine_type}] {reason}")
class StaticActionViolation(ConstraintFailure):
    failure_type = "StaticActionViolation"
class ForbiddenModeViolation(ConstraintFailure):
    failure_type = "ForbiddenModeViolation"
class DynamicStateConstraintViolation(ConstraintFailure):
    failure_type = "DynamicStateConstraintViolation"
class PostStepHardConstraintViolation(ConstraintFailure):
    failure_type = "PostStepHardConstraintViolation"
class FmuNumericalFailure(ConstraintFailure):
    failure_type = "FmuNumericalFailure"
class FmiLifecycleFailure(ConstraintFailure):
    failure_type = "FmiLifecycleFailure"
class NonFiniteOutputFailure(ConstraintFailure):
    failure_type = "NonFiniteOutputFailure"
class FeasibleSetEmpty(ConstraintFailure):
    """动态可行集为空：无法生成合法安全动作，中止 episode，不伪造动作。"""
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
    """结构化后验/预检失败记录，供审计与 SafetyDataset 使用。"""
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
        return asdict(self)
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FailureRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        payload = {k: v for k, v in data.items() if k in known}
        return cls(**payload)  # type: ignore[arg-type]
