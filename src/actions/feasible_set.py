"""用于约束火电、电池与压空模式的动态可行动作集。"""

from __future__ import annotations

from dataclasses import dataclass

from .mode_mask import ModeMask


@dataclass(frozen=True)
class DynamicFeasibleActionSet:
    """状态相关的动态可行动作集(DynamicFeasibleActionSet)。

    由可行性神谕(FeasibilityOracle)根据当前观测计算，给出火电/电池指令上下界、
    压空模式掩码(ModeMask)与压空各方向的安全幅值子区间。

    压空幅值子区间是必需的而非可选诊断：模式掩码只能表达「方向可否」，若不同时
    收窄幅值，智能体会在合法带内挑到越界的幅值而被逐动作校验拒绝；叠加最短运行
    锁后更会被committed进无合法动作的死角。
    """

    u_tp_low: float
    u_tp_high: float
    u_battery_low: float
    u_battery_high: float
    mode_mask: ModeMask
    # 放电区间 ⊆ [-1.0, -0.33]，充电区间 ⊆ [0.86, 1.0]；None 表示该方向不可行
    u_caes_discharge: tuple[float, float] | None = None
    u_caes_charge: tuple[float, float] | None = None
    grid_violation_predicted: bool = False
    metadata: dict | None = None

    def as_dict(self) -> dict:
        """序列化为日志/诊断用字典。

        Returns:
            含动态界、模式允许标志、压空幅值区间与元数据的字典。
        """
        dis = self.u_caes_discharge
        chg = self.u_caes_charge
        return {
            "u_tp_dynamic_low": self.u_tp_low,
            "u_tp_dynamic_high": self.u_tp_high,
            "u_battery_dynamic_low": self.u_battery_low,
            "u_battery_dynamic_high": self.u_battery_high,
            "caes_discharge_allowed": self.mode_mask.discharge,
            "caes_idle_allowed": self.mode_mask.idle,
            "caes_charge_allowed": self.mode_mask.charge,
            "u_caes_discharge_low": None if dis is None else dis[0],
            "u_caes_discharge_high": None if dis is None else dis[1],
            "u_caes_charge_low": None if chg is None else chg[0],
            "u_caes_charge_high": None if chg is None else chg[1],
            "grid_violation_predicted": self.grid_violation_predicted,
            **(self.metadata or {}),
        }
