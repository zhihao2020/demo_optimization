"""将混合动作(HybridAction)解码为功能模型单元(FMU)原生物理指令；不做环境裁剪。"""

from __future__ import annotations

from .types import CaesMode, HybridAction, PhysicalFmuAction

"""
放电区两端 -1.0 ~ -0.33
充电区两端 0.86 ~ 1.0
"""
DISCHARGE_LO = -1.0  # 放电区下界
DISCHARGE_HI = -0.33  # 放电区上界
CHARGE_LO = 0.86  # 充电区下界
CHARGE_HI = 1.0  # 充电区上界


class HybridActionDecoder:
    """混合动作解码器(HybridActionDecoder)：模式+幅值映射到压空连续指令。"""

    def decode(self, action: HybridAction) -> PhysicalFmuAction:
        """解码混合动作为物理动作，只是将压空的指令改为连续量。
        主要就是解决CASE存在三个区间段，不连续的问题
        Args:
            action: 混合动作(HybridAction)。
        Returns:
            物理动作(PhysicalFmuAction)，都是连续量。
        Raises:
            ValueError: 压空模式未知时抛出。
        """
        # 压缩空气储能选定充/放模式之后的强度（0~1）
        mag = 0.0 if action.caes_mode == CaesMode.IDLE else float(action.caes_magnitude)
        # 放电模式
        if action.caes_mode == CaesMode.DISCHARGE:
            u_caes = DISCHARGE_LO + mag * (DISCHARGE_HI - DISCHARGE_LO)
        # 空闲模式 不做任何操作
        elif action.caes_mode == CaesMode.IDLE:
            u_caes = 0.0
        # 充电模式
        elif action.caes_mode == CaesMode.CHARGE:
            u_caes = CHARGE_LO + mag * (CHARGE_HI - CHARGE_LO)
        else:
            raise ValueError(f"未知 CaesMode: {action.caes_mode}")
        return PhysicalFmuAction(
            u_tp=float(action.u_tp),
            u_battery=float(action.u_battery),
            u_caes=float(u_caes),
        )
