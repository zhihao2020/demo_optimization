"""Shadow FMU 二级验证。当前 FMU canGetAndSetFMUstate=False，采用同步独立实例。

Shadow 只作执行前验证，绝不是 fallback，也不推进主 FMU。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping

from actions import FeasibilityOracle, HybridAction, HybridActionDecoder
from actions.validator import hybrid_from_dict
from fmu.exceptions import FmuSolverError

from .safety_result import SafetyCheckResult


class ShadowFmuValidator:
    """同步 Shadow：正常路径与主 FMU 并行推进，失步后才 reset+重放恢复。"""

    def __init__(
        self,
        *,
        factory: Callable[[], Any] | None = None,
        oracle: FeasibilityOracle | None = None,
        enabled: bool = True,
        mode: str = "always",
        near_boundary_fraction: float = 0.15,
    ):
        self.factory = factory
        self.oracle = oracle or FeasibilityOracle.from_root()
        self.enabled = enabled
        self.mode = mode
        self.near_boundary_fraction = float(near_boundary_fraction)
        self.decoder = HybridActionDecoder()
        self.episode_start_time = 0.0
        self.physical_action_history: list[dict[str, float]] = []
        self._shadow: Any | None = None
        self._pending_action: dict[str, float] | None = None
        self.fmu_supports_state = False  # 实测 canGetAndSetFMUstate=False
        self._capabilities = {
            "canGetAndSetFMUstate": False,
            "canSerializeFMUstate": False,
            "strategy": "synchronized_shadow_with_replay_recovery",
        }

    def on_episode_reset(self, start_time: float) -> None:
        self._dispose_shadow()
        self.episode_start_time = float(start_time)
        self.physical_action_history = []
        self._pending_action = None

    def on_physical_success(self, physical_action: Mapping[str, float]) -> None:
        physical = self._physical_action(physical_action)
        self.physical_action_history.append(physical)
        # 已经由 Shadow 执行并被主 FMU确认的候选，保留同步实例，不再重放历史。
        if self._pending_action == physical:
            self._pending_action = None
        else:
            # near_boundary 模式下未经 Shadow 的动作，或调用顺序异常；下次验证安全重建。
            self._dispose_shadow()

    @staticmethod
    def _physical_action(physical_action: Mapping[str, float]) -> dict[str, float]:
        return {
            "u_tp": float(physical_action["u_tp"]),
            "u_battery": float(physical_action["u_battery"]),
            "u_caes": float(physical_action["u_caes"]),
        }

    def _dispose_shadow(self) -> None:
        if self._shadow is not None:
            try:
                self._shadow.close()
            except Exception:
                pass
        self._shadow = None
        self._pending_action = None

    def _synchronized_shadow(self):
        """取得与已确认主 FMU历史一致的 Shadow；仅在失步/拒绝后重放。"""
        if self._shadow is None:
            shadow = self.factory()
            shadow.reset(self.episode_start_time)
            for past in self.physical_action_history:
                shadow.step(past)
            self._shadow = shadow
        return self._shadow

    def should_validate(self, level1: SafetyCheckResult) -> bool:
        if not self.enabled or self.mode == "disabled" or self.factory is None:
            return False
        if self.mode == "always":
            return True
        if self.mode == "near_boundary":
            margins = level1.boundary_margins or {}
            # 任一 safe 距离相对尺度较小则触发
            for key, val in margins.items():
                if key.startswith("safe_") and abs(float(val)) < self.near_boundary_fraction:
                    return True
            return False
        return False

    def validate(
        self,
        action: dict | HybridAction,
        level1: SafetyCheckResult,
    ) -> SafetyCheckResult:
        if not self.should_validate(level1):
            result = deepcopy(level1)
            result.shadow_validation_used = False
            result.shadow_safe = None
            return result
        hybrid = action if isinstance(action, HybridAction) else hybrid_from_dict(action)
        physical = self.decoder.decode(hybrid).as_dict()
        # 上一次候选若未获主 FMU确认，不能复用其已推进的 Shadow 状态。
        if self._pending_action is not None:
            self._dispose_shadow()
        try:
            shadow = self._synchronized_shadow()
            outputs = shadow.step(physical)
            ok, reason = self.oracle.post_step_hard_ok(outputs)
            if not ok:
                self._dispose_shadow()
                result = deepcopy(level1)
                result.safe = False
                result.rejection_stage = "shadow"
                result.violation_type = "shadow_fmu_rejection"
                result.violation_severity = 1.0
                result.normalized_violations = {"shadow_fmu_rejection": 1.0}
                result.shadow_validation_used = True
                result.shadow_safe = False
                result.shadow_failure_reason = reason or "post_step_hard_constraint"
                result.metadata = {**(result.metadata or {}), "shadow_outputs": dict(outputs), **self._capabilities}
                return result
            result = deepcopy(level1)
            result.safe = True
            result.shadow_validation_used = True
            result.shadow_safe = True
            result.shadow_failure_reason = None
            result.metadata = {**(result.metadata or {}), **self._capabilities}
            self._pending_action = self._physical_action(physical)
            return result
        except Exception as exc:
            self._dispose_shadow()
            result = deepcopy(level1)
            result.safe = False
            result.rejection_stage = "shadow"
            result.violation_type = "shadow_fmu_rejection"
            result.violation_severity = 1.0
            result.normalized_violations = {"shadow_fmu_rejection": 1.0}
            result.shadow_validation_used = True
            result.shadow_safe = False
            result.shadow_failure_reason = str(exc)
            result.metadata = {**(result.metadata or {}), **self._capabilities, "exception_type": type(exc).__name__}
            return result

    def close(self) -> None:
        self._dispose_shadow()

    def capabilities(self) -> dict[str, Any]:
        return dict(self._capabilities)
