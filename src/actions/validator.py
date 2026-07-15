"""混合动作与动态可行域的严格校验；绝不静默修正动作。"""

from __future__ import annotations

import math
from typing import Any

from envs.failures import (
    DynamicStateConstraintViolation,
    ForbiddenModeViolation,
    StaticActionViolation,
)

from .feasible_set import DynamicFeasibleActionSet
from .types import CaesMode, HybridAction, PhysicalFmuAction

_EPS = 1e-6


class HybridActionValidator:
    """校验 HybridAction 相对当前 DynamicFeasibleActionSet。"""

    def validate(self, action: HybridAction, feasible: DynamicFeasibleActionSet) -> HybridAction:
        for name, value in (("u_tp", action.u_tp), ("u_battery", action.u_battery), ("caes_magnitude", action.caes_magnitude)):
            if not math.isfinite(value):
                raise StaticActionViolation(f"{name} 非有限: {value}")
        if action.caes_mode == CaesMode.IDLE:
            # magnitude 在 IDLE 下被忽略，但不要求调用方传 0
            pass
        elif not (0.0 - _EPS <= action.caes_magnitude <= 1.0 + _EPS):
            raise StaticActionViolation(f"caes_magnitude={action.caes_magnitude} 不在 [0,1]")

        mask = feasible.mode_mask
        if action.caes_mode == CaesMode.DISCHARGE and not mask.discharge:
            raise ForbiddenModeViolation("CAES DISCHARGE 当前被 mode mask 禁止")
        if action.caes_mode == CaesMode.CHARGE and not mask.charge:
            raise ForbiddenModeViolation("CAES CHARGE 当前被 mode mask 禁止")
        if action.caes_mode == CaesMode.IDLE and not mask.idle:
            raise ForbiddenModeViolation("CAES IDLE 当前被 mode mask 禁止")
        if action.caes_mode not in (CaesMode.DISCHARGE, CaesMode.IDLE, CaesMode.CHARGE):
            raise StaticActionViolation(f"非法 caes_mode={action.caes_mode}")

        if not (feasible.u_tp_low - _EPS <= action.u_tp <= feasible.u_tp_high + _EPS):
            raise DynamicStateConstraintViolation(
                f"u_tp={action.u_tp} 超出动态范围 [{feasible.u_tp_low}, {feasible.u_tp_high}]"
            )
        if not (feasible.u_battery_low - _EPS <= action.u_battery <= feasible.u_battery_high + _EPS):
            raise DynamicStateConstraintViolation(
                f"u_battery={action.u_battery} 超出动态范围 [{feasible.u_battery_low}, {feasible.u_battery_high}]"
            )
        return action

    def validate_physical_static(self, physical: PhysicalFmuAction) -> PhysicalFmuAction:
        """仅检查静态合法集合（用于 step_physical_for_test）。"""
        if not all(math.isfinite(v) for v in (physical.u_tp, physical.u_battery, physical.u_caes)):
            raise StaticActionViolation(f"物理动作含非有限值: {physical}")
        if not (1.0 / 3.0 - _EPS <= physical.u_tp <= 1.0 + _EPS):
            raise StaticActionViolation(f"u_tp={physical.u_tp} 不在 [1/3,1]")
        if not (-1.0 - _EPS <= physical.u_battery <= 1.0 + _EPS):
            raise StaticActionViolation(f"u_battery={physical.u_battery} 不在 [-1,1]")
        u = physical.u_caes
        in_discharge = DISCHARGE_OK(u)
        in_idle = abs(u) <= _EPS
        in_charge = CHARGE_OK(u)
        if not (in_discharge or in_idle or in_charge):
            raise StaticActionViolation(
                f"u_caes={u} 不在 [-1,-0.33]∪{{0}}∪[0.86,1]（forbidden area）"
            )
        return physical


def DISCHARGE_OK(u: float) -> bool:
    return -1.0 - _EPS <= u <= -0.33 + _EPS


def CHARGE_OK(u: float) -> bool:
    return 0.86 - _EPS <= u <= 1.0 + _EPS


def hybrid_from_dict(action: dict[str, Any]) -> HybridAction:
    mode = action["caes_mode"]
    if not isinstance(mode, CaesMode):
        mode = CaesMode(int(np_as_scalar(mode)))
    return HybridAction(
        u_tp=float(np_as_scalar(action["u_tp"])),
        u_battery=float(np_as_scalar(action["u_battery"])),
        caes_mode=mode,
        caes_magnitude=float(np_as_scalar(action["caes_magnitude"])),
    )


def np_as_scalar(value: Any) -> float:
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        return float(value[0] if len(value) else 0.0)
    return float(value)
