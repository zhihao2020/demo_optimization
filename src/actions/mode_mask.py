"""CAES 模式可用性 mask。"""

from __future__ import annotations

from dataclasses import dataclass

from .types import CaesMode


@dataclass(frozen=True)
class ModeMask:
    discharge: bool = True
    idle: bool = True
    charge: bool = True

    def allows(self, mode: CaesMode) -> bool:
        if mode == CaesMode.DISCHARGE:
            return self.discharge
        if mode == CaesMode.IDLE:
            return self.idle
        if mode == CaesMode.CHARGE:
            return self.charge
        return False

    def as_bool_array(self):
        import numpy as np
        return np.asarray([self.discharge, self.idle, self.charge], dtype=bool)

    def as_dict(self) -> dict[str, bool]:
        return {"discharge": self.discharge, "idle": self.idle, "charge": self.charge}

    def logits_mask_value(self, fill: float = -1e9):
        """返回加到 logits 上的 mask（非法模式为 fill）。"""
        import numpy as np
        values = np.zeros(3, dtype=np.float32)
        if not self.discharge:
            values[0] = fill
        if not self.idle:
            values[1] = fill
        if not self.charge:
            values[2] = fill
        return values
