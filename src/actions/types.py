"""混合动作与 FMU 物理动作类型。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class CaesMode(IntEnum):
    DISCHARGE = 0
    IDLE = 1
    CHARGE = 2


@dataclass(frozen=True)
class HybridAction:
    u_tp: float
    u_battery: float
    caes_mode: CaesMode
    caes_magnitude: float


@dataclass(frozen=True)
class PhysicalFmuAction:
    u_tp: float
    u_battery: float
    u_caes: float

    def as_dict(self) -> dict[str, float]:
        return {"u_tp": float(self.u_tp), "u_battery": float(self.u_battery), "u_caes": float(self.u_caes)}

    def as_array(self):
        import numpy as np
        return np.asarray([self.u_tp, self.u_battery, self.u_caes], dtype=np.float32)
