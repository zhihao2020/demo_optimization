"""将 HybridAction 解码为 FMU 原生物理指令。属于正式动作定义，不是环境裁剪。"""

from __future__ import annotations

from .types import CaesMode, HybridAction, PhysicalFmuAction

DISCHARGE_LO = -1.0
DISCHARGE_HI = -0.33
CHARGE_LO = 0.86
CHARGE_HI = 1.0


class HybridActionDecoder:
    """把 caes_mode+magnitude 线性映射到 CAES 合法闭区间，火电/电池原样透传。"""

    def decode(self, action: HybridAction) -> PhysicalFmuAction:
        mag = 0.0 if action.caes_mode == CaesMode.IDLE else float(action.caes_magnitude)
        if action.caes_mode == CaesMode.DISCHARGE:
            u_caes = DISCHARGE_LO + mag * (DISCHARGE_HI - DISCHARGE_LO)
        elif action.caes_mode == CaesMode.IDLE:
            u_caes = 0.0
        elif action.caes_mode == CaesMode.CHARGE:
            u_caes = CHARGE_LO + mag * (CHARGE_HI - CHARGE_LO)
        else:
            raise ValueError(f"未知 CaesMode: {action.caes_mode}")
        return PhysicalFmuAction(
            u_tp=float(action.u_tp),
            u_battery=float(action.u_battery),
            u_caes=float(u_caes),
        )

    def decode_dict(self, action: dict) -> PhysicalFmuAction:
        mode = action["caes_mode"]
        if not isinstance(mode, CaesMode):
            mode = CaesMode(int(mode))
        mag = float(action["caes_magnitude"])
        if hasattr(mag, "__len__"):
            mag = float(mag[0]) if len(mag) else 0.0
        u_tp = float(
            action["u_tp"][0] if hasattr(action["u_tp"], "__len__") else action["u_tp"]
        )
        u_bat = float(
            action["u_battery"][0]
            if hasattr(action["u_battery"], "__len__")
            else action["u_battery"]
        )
        return self.decode(
            HybridAction(u_tp=u_tp, u_battery=u_bat, caes_mode=mode, caes_magnitude=mag)
        )
