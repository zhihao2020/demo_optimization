"""GiveSafe 安全结果与候选元数据。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SafetyCheckResult:
    safe: bool
    rejection_stage: str | None = None  # oracle | shadow | false_safe | None
    violation_type: str | None = None
    violation_severity: float = 0.0
    normalized_violations: dict[str, float] = field(default_factory=dict)
    predicted_next_state: dict[str, float] = field(default_factory=dict)
    dynamic_bounds: dict[str, float] = field(default_factory=dict)
    boundary_margins: dict[str, float] = field(default_factory=dict)
    mode_mask: dict[str, bool] = field(default_factory=dict)
    shadow_validation_used: bool = False
    shadow_safe: bool | None = None
    shadow_failure_reason: str | None = None
    oracle_safe: bool | None = None
    oracle_rejection_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GiveSafeResult:
    safe_action: dict | None
    proposed_actions: list[dict] = field(default_factory=list)
    rejected_actions: list[dict] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)
    constraint_rewards: list[float] = field(default_factory=list)
    attempt_count: int = 0
    oracle_version: str | None = None
    safety_check_metadata: list[SafetyCheckResult] = field(default_factory=list)
    no_safe_action: bool = False
