"""影子仿真校验器(ShadowFmuValidator) 二级验证。

当前 FMU canGetAndSetFMUstate=False，采用同步独立实例。
Shadow 仅作执行前验证，绝不是 fallback，也不推进主 FMU。
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
    """同步影子仿真(Shadow)：正常路径与主 FMU 并行推进，失步后 reset 并重放历史恢复。"""

    def __init__(
        self,
        *,
        factory: Callable[[], Any] | None = None,
        oracle: FeasibilityOracle | None = None,
        enabled: bool = True,
        mode: str = "always",
        near_boundary_fraction: float = 0.15,
    ):
        """配置影子 FMU 工厂、神谕与触发模式。

        Args:
            factory: 无参工厂，每次创建独立 Shadow FMU 实例。
            oracle: 可行性神谕(FeasibilityOracle)；为 None 时使用默认实例。
            enabled: 是否启用二级 Shadow 校验。
            mode: 触发模式：always | near_boundary | disabled。
            near_boundary_fraction: near_boundary 模式下判定近边界的相对距离阈值。

        Returns:
            无。

        Raises:
            无。
        """
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
        """新 episode 开始时重置 Shadow 状态与已确认动作历史。

        Args:
            start_time: episode 起始仿真时间。

        Returns:
            无。

        Raises:
            无。
        """
        self._dispose_shadow()
        self.episode_start_time = float(start_time)
        self.physical_action_history = []
        self._pending_action = None

    def on_physical_success(self, physical_action: Mapping[str, float]) -> None:
        """主 FMU 成功执行物理动作后同步 Shadow 跟踪状态。

        Args:
            physical_action: 主 FMU 已确认执行的物理动作字典。

        Returns:
            无。

        Raises:
            无。
        """
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
        """提取并规范化物理动作三元组 (u_tp, u_battery, u_caes)。

        Args:
            physical_action: 含控制量的动作映射。

        Returns:
            仅含 u_tp、u_battery、u_caes 的 float 字典。

        Raises:
            KeyError: 缺少必需控制键时。
        """
        return {
            "u_tp": float(physical_action["u_tp"]),
            "u_battery": float(physical_action["u_battery"]),
            "u_caes": float(physical_action["u_caes"]),
        }

    def _dispose_shadow(self) -> None:
        """关闭并释放当前 Shadow FMU 实例。

        Args:
            无。

        Returns:
            无。

        Raises:
            无：关闭失败时静默忽略。
        """
        if self._shadow is not None:
            try:
                self._shadow.close()
            except Exception:
                pass
        self._shadow = None
        self._pending_action = None

    def _synchronized_shadow(self):
        """取得与已确认主 FMU 历史一致的 Shadow 实例；仅在失步或拒绝后重放。

        Args:
            无。

        Returns:
            已与 physical_action_history 对齐的 Shadow FMU 实例。

        Raises:
            由 factory 或 shadow.reset/step 抛出的 FMU 相关异常。
        """
        if self._shadow is None:
            shadow = self.factory()
            shadow.reset(self.episode_start_time)
            for past in self.physical_action_history:
                shadow.step(past)
            self._shadow = shadow
        return self._shadow

    def should_validate(self, level1: SafetyCheckResult) -> bool:
        """根据配置与一级结果判断本步是否执行 Shadow 校验。

        Args:
            level1: 一级 Oracle 安全检查已通过的结果。

        Returns:
            True 表示需要执行 Shadow validate。

        Raises:
            无。
        """
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
        """在 Shadow FMU 上试探性执行候选动作并做硬约束后验检查。

        Args:
            action: 候选混合动作（dict 或 HybridAction）。
            level1: 一级 Oracle 已通过的安全检查结果（将被 deepcopy 扩展）。

        Returns:
            更新 shadow_* 字段后的 SafetyCheckResult；跳过时返回未启用 Shadow 的拷贝。

        Raises:
            无：FMU 异常转为 safe=False 的结果返回。
        """
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
        """释放 Shadow FMU 资源，供环境 teardown 调用。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """
        self._dispose_shadow()

    def capabilities(self) -> dict[str, Any]:
        """返回 Shadow 策略与 FMU 能力声明的只读拷贝。

        Args:
            无。

        Returns:
            含 canGetAndSetFMUstate、strategy 等键的能力字典。

        Raises:
            无。
        """
        return dict(self._capabilities)
