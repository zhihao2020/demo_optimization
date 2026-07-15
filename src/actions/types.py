"""混合动作与 FMU 物理动作类型。

策略侧使用 HybridAction（模式+幅值）；写入 FMU 前必须经 Decoder 变为 PhysicalFmuAction。
二者分离是为了把非凸 CAES 区间显式建模，而不是在连续 Box 里静默投影。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class CaesMode(IntEnum):
    """CAES 离散模式：与 ModeMask / Actor mode_head 的索引一致。"""

    DISCHARGE = 0
    IDLE = 1
    CHARGE = 2


@dataclass(frozen=True)
class HybridAction:
    """策略输出：火电/电池连续指令 + CAES 模式与幅值（幅值∈[0,1]，IDLE 时忽略）。"""

    u_tp: float
    u_battery: float
    caes_mode: CaesMode
    caes_magnitude: float


@dataclass(frozen=True)
class PhysicalFmuAction:
    """FMU 三输入无量纲指令；边界见 ``fmu.validate`` / docs/FMU输入上下限.md。"""

    u_tp: float
    u_battery: float
    u_caes: float

    def as_dict(self) -> dict[str, float]:
        return {"u_tp": float(self.u_tp), "u_battery": float(self.u_battery), "u_caes": float(self.u_caes)}

    def as_array(self):
        import numpy as np
        return np.asarray([self.u_tp, self.u_battery, self.u_caes], dtype=np.float32)
