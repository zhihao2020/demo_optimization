"""规则控制器：输出物理三元组动作，与正式策略共用可行域接口。"""

from __future__ import annotations

from typing import Any

import numpy as np

from actions import FeasibilityOracle
from actions.caes_u import physical_dict


class RuleBasedController:
    """规则基线：火电尽量高出力、储能 IDLE；动作落在动态可行域内。"""

    def __init__(self, env_or_space: Any = None, oracle: FeasibilityOracle | None = None) -> None:
        self.action_space = getattr(env_or_space, "action_space", env_or_space)
        self.env = env_or_space if hasattr(env_or_space, "get_feasible_action_spec") else None
        self.oracle = oracle or FeasibilityOracle.from_root()

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> dict:
        _ = deterministic
        if self.env is not None and self.env.last_outputs is not None:
            feasible = self.env.get_feasible_action_spec()
        else:
            outputs = self._obs_to_outputs(observation)
            feasible = self.oracle.compute(outputs, float(outputs.get("p_thermal", -1.5e8)))
        u_tp = float(feasible.u_tp_high)
        if feasible.u_battery_low <= 0.0 <= feasible.u_battery_high:
            u_bat = 0.0
        else:
            u_bat = 0.5 * (feasible.u_battery_low + feasible.u_battery_high)
        u_caes = 0.0
        if not feasible.mode_mask.idle:
            # idle 应始终可用；否则取允许方向的最小合法端
            if feasible.mode_mask.discharge:
                u_caes = -0.33
            elif feasible.mode_mask.charge:
                u_caes = 0.86
        return physical_dict(u_tp, u_bat, u_caes)

    @staticmethod
    def _obs_to_outputs(observation: np.ndarray) -> dict[str, float]:
        names = (
            "battery_soc", "caes_gas_soc", "caes_hot_soc", "caes_cold_soc",
            "caes_gas_pressure", "caes_gas_temperature", "caes_hot_temperature", "caes_cold_temperature",
            "p_thermal", "p_battery", "p_caes", "p_grid",
            "p_wind_available", "p_wind_actual", "p_pv_available", "p_pv_actual",
            "p_load_actual", "p_curtailment", "p_unserved",
        )
        obs = np.asarray(observation, dtype=np.float64).ravel()
        return {names[i]: float(obs[i]) for i in range(min(len(names), obs.size))}
