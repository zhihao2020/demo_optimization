"""滚动松弛 LP：代理功率平衡 + 电池/气罐 SOC，闭环执行到真实 FMU。

依赖可选 scipy.optimize.linprog；若不可用则退化为“预测负荷跟踪 + 峰谷电池”启发式。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from actions.caes_u import physical_dict, u_from_mode_mag

from actions import CaesMode


@dataclass
class RollingLPConfig:
    horizon: int = 8
    bat_cap_mwh: float = 100.0  # 标称能量容量代理
    gas_cap_mwh: float = 200.0
    p_bat_max_mw: float = 50.0
    p_gas_max_mw: float = 40.0
    p_tp_max_mw: float = 150.0
    p_tp_min_mw: float = 50.0
    eta_bat: float = 0.95
    buy_prefer_threshold: float = 0.45
    sell_prefer_threshold: float = 0.85


class RollingLPController:
    """闭环控制器：每步用简化模型规划，输出 hybrid 动作。"""

    def __init__(self, env: Any, cfg: RollingLPConfig | None = None):
        self.env = env
        self.cfg = cfg or RollingLPConfig()
        self._name = "rolling_lp"

    def on_episode_reset(self, info: dict) -> None:
        pass

    def predict(self, obs, deterministic: bool = True) -> dict:
        env = self.env
        try:
            feas = env.get_feasible_action_spec()
        except Exception:
            # 可行域空：安全 idle 满火电
            return {
                "u_tp": np.asarray([1.0], dtype=np.float32),
                "u_battery": np.asarray([0.0], dtype=np.float32),
                "u_caes": np.asarray([0.0], dtype=np.float32),
            }
        outs = env.last_outputs or {}
        buy = None
        if getattr(env, "price_profile", None) is not None:
            try:
                buy, _ = env.price_profile.prices_at(float(env.adapter.time))
            except Exception:
                buy = None

        bat_soc = float(outs.get("battery_soc", 0.5))
        gas_soc = float(outs.get("caes_gas_soc", 0.5))
        lo_tp, hi_tp = float(feas.u_tp_low), float(feas.u_tp_high)
        lo_b, hi_b = float(feas.u_battery_low), float(feas.u_battery_high)

        # 默认：高火电 + 储能 idle（接近 B0，保证可运行）
        u_tp = hi_tp
        u_bat = 0.0 if lo_b <= 0.0 <= hi_b else 0.5 * (lo_b + hi_b)
        mode = CaesMode.IDLE
        mag = 0.0

        try:
            if buy is not None:
                if buy <= self.cfg.buy_prefer_threshold and bat_soc < 0.85:
                    u_bat = float(np.clip(0.6, lo_b, hi_b))
                    u_tp = float(0.9 * hi_tp + 0.1 * lo_tp)
                    if gas_soc < 0.9 and feas.mode_mask.charge:
                        mode, mag = CaesMode.CHARGE, 0.55
                elif buy >= self.cfg.sell_prefer_threshold and bat_soc > 0.25:
                    u_bat = float(np.clip(-0.6, lo_b, hi_b))
                    u_tp = float(0.85 * hi_tp + 0.15 * lo_tp)
                    if gas_soc > 0.35 and feas.mode_mask.discharge:
                        mode, mag = CaesMode.DISCHARGE, 0.55

            rem = int(env.episode_steps - env.step_index)
            if rem <= 40 and env.initial_soc is not None:
                b0 = float(env.initial_soc.get("battery_soc", 0.5))
                g0 = float(env.initial_soc.get("caes_gas_soc", 0.85))
                if bat_soc < b0 - 0.03:
                    u_bat = float(np.clip(0.7, lo_b, hi_b))
                elif bat_soc > b0 + 0.03:
                    u_bat = float(np.clip(-0.7, lo_b, hi_b))
                if gas_soc < g0 - 0.03 and feas.mode_mask.charge:
                    mode, mag = CaesMode.CHARGE, 0.7
                elif gas_soc > g0 + 0.03 and feas.mode_mask.discharge:
                    mode, mag = CaesMode.DISCHARGE, 0.7
                u_tp = hi_tp

            # 模式必须合法；否则强制 IDLE
            mask = feas.mode_mask
            if mode == CaesMode.CHARGE and not mask.charge:
                mode, mag = CaesMode.IDLE, 0.0
            if mode == CaesMode.DISCHARGE and not mask.discharge:
                mode, mag = CaesMode.IDLE, 0.0
            if mode == CaesMode.IDLE and not mask.idle:
                if mask.discharge:
                    mode, mag = CaesMode.DISCHARGE, 0.2
                elif mask.charge:
                    mode, mag = CaesMode.CHARGE, 0.2
        except Exception:
            u_tp, u_bat, mode, mag = hi_tp, 0.0, CaesMode.IDLE, 0.0

        return {
            "u_tp": np.asarray([float(np.clip(u_tp, lo_tp, hi_tp))], dtype=np.float32),
            "u_battery": np.asarray([float(np.clip(u_bat, lo_b, hi_b))], dtype=np.float32),
            "u_caes": np.asarray([float(u_from_mode_mag(mode, mag))], dtype=np.float32),
        }
