"""对物理三元组动作与动态可行域做严格校验；绝不静默修正动作。"""

from __future__ import annotations

import math
from typing import Any

from envs.failures import (
    DynamicStateConstraintViolation,
    ForbiddenModeViolation,
    StaticActionViolation,
)

from .caes_u import (
    in_charge_band,
    in_discharge_band,
    in_idle,
    is_legal_u_caes,
    mode_from_u,
    np_as_scalar,
    project_u_caes,
)
from .feasible_set import DynamicFeasibleActionSet
from .types import CaesMode, PhysicalFmuAction

_EPS = 1e-6


class PhysicalActionValidator:
    """物理动作验证器：相对动态可行动作集做硬校验。"""

    def validate(
        self, action: PhysicalFmuAction, feasible: DynamicFeasibleActionSet
    ) -> PhysicalFmuAction:
        for name, value in (
            ("u_tp", action.u_tp),
            ("u_battery", action.u_battery),
            ("u_caes", action.u_caes),
        ):
            if not math.isfinite(value):
                raise StaticActionViolation(f"{name} 非有限: {value}")

        if not is_legal_u_caes(action.u_caes):
            raise StaticActionViolation(
                f"u_caes={action.u_caes} 不在 [-1,-0.33]∪{{0}}∪[0.86,1]"
            )

        mode = mode_from_u(action.u_caes)
        mask = feasible.mode_mask
        if mode == CaesMode.DISCHARGE and not mask.discharge:
            raise ForbiddenModeViolation("CAES DISCHARGE 当前被 mode mask 禁止")
        if mode == CaesMode.CHARGE and not mask.charge:
            raise ForbiddenModeViolation("CAES CHARGE 当前被 mode mask 禁止")
        if mode == CaesMode.IDLE and not mask.idle:
            raise ForbiddenModeViolation("CAES IDLE 当前被 mode mask 禁止")

        if not (feasible.u_tp_low - _EPS <= action.u_tp <= feasible.u_tp_high + _EPS):
            raise DynamicStateConstraintViolation(
                f"u_tp={action.u_tp} 超出动态范围 [{feasible.u_tp_low}, {feasible.u_tp_high}]"
            )
        if not (
            feasible.u_battery_low - _EPS
            <= action.u_battery
            <= feasible.u_battery_high + _EPS
        ):
            raise DynamicStateConstraintViolation(
                f"u_battery={action.u_battery} 超出动态范围 "
                f"[{feasible.u_battery_low}, {feasible.u_battery_high}]"
            )
        return action

    def validate_physical_static(self, physical: PhysicalFmuAction) -> PhysicalFmuAction:
        """仅检查物理动作的静态合法集合（测试步进用）。"""
        if not all(
            math.isfinite(v)
            for v in (physical.u_tp, physical.u_battery, physical.u_caes)
        ):
            raise StaticActionViolation(f"物理动作含非有限值: {physical}")
        if not (1.0 / 3.0 - _EPS <= physical.u_tp <= 1.0 + _EPS):
            raise StaticActionViolation(f"u_tp={physical.u_tp} 不在 [1/3,1]")
        if not (-1.0 - _EPS <= physical.u_battery <= 1.0 + _EPS):
            raise StaticActionViolation(f"u_battery={physical.u_battery} 不在 [-1,1]")
        if not is_legal_u_caes(physical.u_caes):
            raise StaticActionViolation(
                f"u_caes={physical.u_caes} 不在 [-1,-0.33]∪{{0}}∪[0.86,1]"
            )
        return physical


# 兼容旧测试名
def DISCHARGE_OK(u: float) -> bool:
    return in_discharge_band(u)


def CHARGE_OK(u: float) -> bool:
    return in_charge_band(u)


def physical_from_dict(action: dict[str, Any]) -> PhysicalFmuAction:
    """从环境字典构造物理动作。执行量只认 u_tp / u_battery / u_caes。

    Actor 诊断字段（caes_mode_onehot / caes_magnitude / mag）允许并存；
    若缺少 u_caes 则拒绝走旧 hybrid 字段。
    """
    if "u_caes" not in action:
        raise KeyError("动作必须含 u_caes（物理路径；不再接受 caes_mode/caes_magnitude）")
    return PhysicalFmuAction(
        u_tp=float(np_as_scalar(action["u_tp"])),
        u_battery=float(np_as_scalar(action["u_battery"])),
        u_caes=float(project_u_caes(np_as_scalar(action["u_caes"]))),
    )
