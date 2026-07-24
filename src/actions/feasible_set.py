"""用于约束火电、电池与压空模式的动态可行动作集。"""

from __future__ import annotations

from dataclasses import dataclass

from .mode_mask import ModeMask


@dataclass(frozen=True)
class DynamicFeasibleActionSet:
    """状态相关的动态可行动作集(DynamicFeasibleActionSet)。

    由可行性神谕(FeasibilityOracle)根据当前观测计算，给出火电/电池指令上下界
    与压空模式掩码(ModeMask)。
    """

    u_tp_low: float
    u_tp_high: float
    u_battery_low: float
    u_battery_high: float
    mode_mask: ModeMask
    grid_violation_predicted: bool = False
    metadata: dict | None = None

    def as_dict(self) -> dict:
        """序列化为日志/诊断用字典。

        Returns:
            含动态界、模式允许标志与元数据的字典。
        """
        return {
            "u_tp_dynamic_low": self.u_tp_low,
            "u_tp_dynamic_high": self.u_tp_high,
            "u_battery_dynamic_low": self.u_battery_low,
            "u_battery_dynamic_high": self.u_battery_high,
            "caes_discharge_allowed": self.mode_mask.discharge,
            "caes_idle_allowed": self.mode_mask.idle,
            "caes_charge_allowed": self.mode_mask.charge,
            "grid_violation_predicted": self.grid_violation_predicted,
            **(self.metadata or {}),
        }
