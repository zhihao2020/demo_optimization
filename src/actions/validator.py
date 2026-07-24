"""对混合动作与动态可行域做严格校验；绝不静默修正动作。"""

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
    """混合动作验证器(HybridActionValidator)：相对动态可行动作集做硬校验。"""

    def validate(self, action: HybridAction, feasible: DynamicFeasibleActionSet) -> HybridAction:
        """校验混合动作是否落在当前动态可行域内。

        Args:
            action: 混合动作(HybridAction)。
            feasible: 动态可行动作集(DynamicFeasibleActionSet)。

        Returns:
            通过校验的同一混合动作。

        Raises:
            StaticActionViolation: 非有限值、非法幅值或非法模式。
            ForbiddenModeViolation: 模式被掩码禁止。
            DynamicStateConstraintViolation: 火电或电池指令越动态界。
        """
        for name, value in (
            ("u_tp", action.u_tp),
            ("u_battery", action.u_battery),
            ("caes_magnitude", action.caes_magnitude),
        ):
            if not math.isfinite(value):
                raise StaticActionViolation(f"{name} 非有限: {value}")
        if action.caes_mode == CaesMode.IDLE:
            # 待机时幅值被忽略，不要求调用方传 0
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
        if not (
            feasible.u_battery_low - _EPS <= action.u_battery <= feasible.u_battery_high + _EPS
        ):
            raise DynamicStateConstraintViolation(
                f"u_battery={action.u_battery} 超出动态范围 "
                f"[{feasible.u_battery_low}, {feasible.u_battery_high}]"
            )
        return action

    def validate_physical_static(self, physical: PhysicalFmuAction) -> PhysicalFmuAction:
        """仅检查物理动作的静态合法集合（测试步进用）。

        Args:
            physical: 物理动作(PhysicalFmuAction)。

        Returns:
            通过校验的同一物理动作。

        Raises:
            StaticActionViolation: 非有限或落入禁止区间。
        """
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
                f"u_caes={u} 不在 [-1,-0.33]∪{{0}}∪[0.86,1]（禁止区间）"
            )
        return physical


def DISCHARGE_OK(u: float) -> bool:
    """判断压空指令是否落在放电合法区间。

    Args:
        u: 压空连续指令。

    Returns:
        在放电区间内则为真。
    """
    return -1.0 - _EPS <= u <= -0.33 + _EPS


def CHARGE_OK(u: float) -> bool:
    """判断压空指令是否落在充电合法区间。

    Args:
        u: 压空连续指令。

    Returns:
        在充电区间内则为真。
    """
    return 0.86 - _EPS <= u <= 1.0 + _EPS


def hybrid_from_dict(action: dict[str, Any]) -> HybridAction:
    """从环境字典构造混合动作。

    Args:
        action: 含 u_tp、u_battery、caes_mode、caes_magnitude 的字典。

    Returns:
        混合动作(HybridAction)。
    """
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
    """将标量或长度为 1 的数组转为浮点。

    Args:
        value: 数值或可索引序列。

    Returns:
        浮点标量。
    """
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        return float(value[0] if len(value) else 0.0)
    return float(value)
