"""物理 FMU 动作类型：与 Modelica 三个 RealInput 一一对应。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class CaesMode(IntEnum):
    """CAES 运行方向（由 u_caes 派生，用于锁/最短运行/日志，非策略动作维）。"""

    DISCHARGE = 0
    IDLE = 1
    CHARGE = 2


@dataclass(frozen=True)
class PhysicalFmuAction:
    """写入 FMU 的原生物理指令三元组：u_tp, u_battery, u_caes。"""

    u_tp: float
    u_battery: float
    u_caes: float

    def as_dict(self) -> dict[str, float]:
        return {
            "u_tp": float(self.u_tp),
            "u_battery": float(self.u_battery),
            "u_caes": float(self.u_caes),
        }

    def as_env_dict(self) -> dict[str, object]:
        """Gymnasium / 训练侧常用的长度为 1 的 array 字典。"""
        import numpy as np

        return {
            "u_tp": np.asarray([float(self.u_tp)], dtype=np.float32),
            "u_battery": np.asarray([float(self.u_battery)], dtype=np.float32),
            "u_caes": np.asarray([float(self.u_caes)], dtype=np.float32),
        }
