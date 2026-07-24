"""用于约束压缩空气储能(CAES)充/放/待机可用性的模式掩码。"""

from __future__ import annotations

from dataclasses import dataclass

from .types import CaesMode


@dataclass(frozen=True)
class ModeMask:
    """模式掩码(ModeMask)：标记当前状态下哪些压空模式合法。"""

    discharge: bool = True
    idle: bool = True
    charge: bool = True

    def allows(self, mode: CaesMode) -> bool:
        """判断给定模式是否被允许。

        Args:
            mode: 压空模式(CaesMode)。

        Returns:
            允许则为真，否则为假。
        """
        if mode == CaesMode.DISCHARGE:
            return self.discharge
        if mode == CaesMode.IDLE:
            return self.idle
        if mode == CaesMode.CHARGE:
            return self.charge
        return False

    def as_bool_array(self):
        """转为长度为 3 的布尔数组。

        Returns:
            顺序为 [放电, 待机, 充电] 的布尔数组。
        """
        import numpy as np

        return np.asarray([self.discharge, self.idle, self.charge], dtype=bool)

    def as_dict(self) -> dict[str, bool]:
        """序列化为字典。

        Returns:
            含 discharge、idle、charge 的布尔字典。
        """
        return {"discharge": self.discharge, "idle": self.idle, "charge": self.charge}
