"""混合动作与物理仿真动作的类型定义。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class CaesMode(IntEnum):
    """压缩空气储能(CAES)离散运行模式。"""

    DISCHARGE = 0
    IDLE = 1
    CHARGE = 2


@dataclass(frozen=True)
class HybridAction:
    """
    策略侧混合动作：火电指令 + 电池指令 + 压空模式 + 压空模式强度。
    u_tp: 火电指令，范围为 0.0 ~ 1.2
    u_battery: 电池指令，范围为 -1.5 ~ 1.5
    caes_mode: 压空模式，0: 放电，1: 空闲，2: 充电
    caes_magnitude: 压空模式强度，范围为 0.0 ~ 1.0
    """

    u_tp: float # 火电指令
    u_battery: float # 电池指令
    caes_mode: CaesMode # 压空模式
    caes_magnitude: float # 压空模式强度


@dataclass(frozen=True)
class PhysicalFmuAction:
    """写入功能模型单元(FMU)的原生物理指令三元组。"""

    u_tp: float
    u_battery: float
    u_caes: float

    def as_dict(self) -> dict[str, float]:
        """转为功能模型单元(FMU)输入字典。

        Returns:
            含 u_tp、u_battery、u_caes 的浮点字典。
        """
        return {
            "u_tp": float(self.u_tp),
            "u_battery": float(self.u_battery),
            "u_caes": float(self.u_caes),
        }
