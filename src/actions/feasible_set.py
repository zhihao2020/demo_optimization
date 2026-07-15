"""状态相关动态可行动作集合 A(s)：连续边界 + CAES ModeMask。

由 FeasibilityOracle.compute 生成；Actor 与校验器只消费此结构，不各自猜边界。
"""

from __future__ import annotations

from dataclasses import dataclass

from .mode_mask import ModeMask


@dataclass(frozen=True)
class DynamicFeasibleActionSet:
    u_tp_low: float
    u_tp_high: float
    u_battery_low: float
    u_battery_high: float
    mode_mask: ModeMask
    grid_violation_predicted: bool = False
    metadata: dict | None = None

    def as_dict(self) -> dict:
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
