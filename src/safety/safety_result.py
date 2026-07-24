"""安全给予(GiveSafe) 结果与候选动作元数据数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SafetyCheckResult:
    """单次安全检查(Oracle / Shadow) 的完整判定结果。"""

    safe: bool  # 是否通过安全检查
    rejection_stage: str | None = None  # 拒绝阶段：oracle | shadow | false_safe | None
    violation_type: str | None = None  # 违规类型标识
    violation_severity: float = 0.0  # 违规严重度（归一化）
    normalized_violations: dict[str, float] = field(default_factory=dict)  # 各约束归一化越界量
    predicted_next_state: dict[str, float] = field(default_factory=dict)  # 预测下一状态
    dynamic_bounds: dict[str, float] = field(default_factory=dict)  # 动态动作上下界
    boundary_margins: dict[str, float] = field(default_factory=dict)  # 物理/安全边界余量
    mode_mask: dict[str, bool] = field(default_factory=dict)  # 可行模式掩码
    shadow_validation_used: bool = False  # 是否启用了影子仿真校验
    shadow_safe: bool | None = None  # 影子仿真是否通过
    shadow_failure_reason: str | None = None  # 影子仿真失败原因
    oracle_safe: bool | None = None  # 可行性神谕(Oracle) 是否通过
    oracle_rejection_reason: str | None = None  # 神谕拒绝原因
    metadata: dict[str, Any] = field(default_factory=dict)  # 附加元数据


@dataclass
class GiveSafeResult:
    """一次 select_safe_action 调用的完整重采样轨迹与最终结果。"""

    safe_action: dict | None  # 最终选中的安全动作，失败时为 None
    proposed_actions: list[dict] = field(default_factory=list)  # 所有候选动作
    rejected_actions: list[dict] = field(default_factory=list)  # 被拒绝的候选动作
    rejection_reasons: list[str] = field(default_factory=list)  # 各次拒绝原因
    constraint_rewards: list[float] = field(default_factory=list)  # 各次拒绝对应的约束奖励
    attempt_count: int = 0  # 实际尝试次数
    oracle_version: str | None = None  # 所用神谕版本
    safety_check_metadata: list[SafetyCheckResult] = field(default_factory=list)  # 各次检查元数据
    no_safe_action: bool = False  # 是否在最大尝试后仍未找到安全动作
