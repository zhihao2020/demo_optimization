"""规则控制器：输出 HybridAction，与正式策略共用可行域接口。"""

from __future__ import annotations

from typing import Any

import numpy as np

from actions import CaesMode, FeasibilityOracle, HybridAction


class RuleBasedController:
    """规则基线控制器(RuleBasedController)：火电尽量维持高出力、储能 IDLE；动作始终落在动态可行域内。"""

    def __init__(self, env_or_space: Any = None, oracle: FeasibilityOracle | None = None) -> None:
        """初始化规则控制器。

        Args:
            env_or_space: 环境实例或动作空间(env_or_space)；若环境提供
                get_feasible_action_spec 则优先使用环境可行域。
            oracle: 可行域预言机(FeasibilityOracle)；未提供时从项目根加载默认实例。
        """
        self.action_space = getattr(env_or_space, "action_space", env_or_space)
        self.env = env_or_space if hasattr(env_or_space, "get_feasible_action_spec") else None
        self.oracle = oracle or FeasibilityOracle.from_root()

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> dict:
        """根据观测生成保守混合动作字典（SB3 风格数组包装）。

        Args:
            observation: 环境观测向量(observation)。
            deterministic: 是否确定性策略(deterministic)；本控制器忽略随机性，参数保留以兼容接口。

        Returns:
            含 u_tp、u_battery、caes_mode、caes_magnitude 键的字典，值为 NumPy 数组或整型。
        """
        if self.env is not None and self.env.last_outputs is not None:
            feasible = self.env.get_feasible_action_spec()
        else:
            # 无环境上下文时使用观测重建最小可行集
            outputs = self._obs_to_outputs(observation)
            feasible = self.oracle.compute(outputs, float(outputs.get("p_thermal", -1.5e8)))
        u_tp = float(feasible.u_tp_high)
        # 电池保持 0（若 0 不在动态区间则取区间中点）
        if feasible.u_battery_low <= 0.0 <= feasible.u_battery_high:
            u_bat = 0.0
        else:
            u_bat = 0.5 * (feasible.u_battery_low + feasible.u_battery_high)
        mode = CaesMode.IDLE
        if not feasible.mode_mask.idle:
            # 理论上 idle 应始终可用；若否则选任意可用模式幅值 0
            if feasible.mode_mask.discharge:
                mode = CaesMode.DISCHARGE
            elif feasible.mode_mask.charge:
                mode = CaesMode.CHARGE
        hybrid = HybridAction(u_tp=u_tp, u_battery=u_bat, caes_mode=mode, caes_magnitude=0.0)
        return {
            "u_tp": np.asarray([hybrid.u_tp], dtype=np.float32),
            "u_battery": np.asarray([hybrid.u_battery], dtype=np.float32),
            "caes_mode": int(hybrid.caes_mode),
            "caes_magnitude": np.asarray([hybrid.caes_magnitude], dtype=np.float32),
        }

    @staticmethod
    def _obs_to_outputs(observation: np.ndarray) -> dict[str, float]:
        """将扁平观测向量按固定顺序映射为 FMU 输出名字典。

        Args:
            observation: 环境观测向量(observation)。

        Returns:
            输出名到浮点值的字典；观测长度不足时仅映射已有前缀维度。
        """
        names = (
            "battery_soc", "caes_gas_soc", "caes_hot_soc", "caes_cold_soc",
            "caes_gas_pressure", "caes_gas_temperature", "caes_hot_temperature", "caes_cold_temperature",
            "p_thermal", "p_battery", "p_caes", "p_grid",
            "p_wind_available", "p_wind_actual", "p_pv_available", "p_pv_actual",
            "p_load_actual", "p_curtailment", "p_unserved",
        )
        obs = np.asarray(observation, dtype=np.float64).ravel()
        return {names[i]: float(obs[i]) for i in range(min(len(names), obs.size))}
