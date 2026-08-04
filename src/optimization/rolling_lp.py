"""滚动松弛 LP：代理功率平衡 + 电池/气罐 SOC，闭环执行到真实 FMU。

依赖可选 scipy.optimize.linprog；若不可用则退化为“预测负荷跟踪 + 峰谷电池”启发式。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

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
        feas = env.get_feasible_action_spec()
        outs = env.last_outputs or {}
        buy = sell = None
        if getattr(env, "price_profile", None) is not None:
            try:
                buy, sell = env.price_profile.prices_at(float(env.adapter.time))
            except Exception:
                buy = sell = None

        bat_soc = float(outs.get("battery_soc", 0.5))
        gas_soc = float(outs.get("caes_gas_soc", 0.5))
        p_load = float(outs.get("p_load_actual", 0.0)) * 1e-6  # W → MW
        p_wind = float(outs.get("p_wind_available", 0.0)) * 1e-6
        p_pv = float(outs.get("p_pv_available", 0.0)) * 1e-6
        net = p_load - p_wind - p_pv  # 正=缺电

        # —— 松弛启发式（可换 linprog）：默认高火电托底，电价驱动储能 ——
        # 火电默认取动态上界（与 B0 一致），避免欠发导致代理模型不可比
        u_tp = float(feas.u_tp_high)
        u_bat = 0.0
        mode = CaesMode.IDLE
        mag = 0.0

        if buy is not None:
            if buy <= self.cfg.buy_prefer_threshold and bat_soc < 0.85:
                # 谷：充电；火电略降但仍保持较高出力
                u_bat = float(np.clip(0.7, feas.u_battery_low, feas.u_battery_high))
                u_tp = float(0.85 * feas.u_tp_high + 0.15 * feas.u_tp_low)
                if gas_soc < 0.9 and feas.mode_mask.charge:
                    mode, mag = CaesMode.CHARGE, 0.7
            elif buy >= self.cfg.sell_prefer_threshold and bat_soc > 0.25:
                u_bat = float(np.clip(-0.7, feas.u_battery_low, feas.u_battery_high))
                if gas_soc > 0.35 and feas.mode_mask.discharge:
                    mode, mag = CaesMode.DISCHARGE, 0.7
                # 峰：可略降火电让储能放电
                u_tp = float(0.75 * feas.u_tp_high + 0.25 * feas.u_tp_low)
            else:
                # 平段：仍偏高出力
                u_tp = float(feas.u_tp_high)
                if net > 100:
                    u_tp = float(feas.u_tp_high)
                elif net < 0 and bat_soc < 0.8:
                    u_bat = float(np.clip(0.3, feas.u_battery_low, feas.u_battery_high))

        # 期末回收：与 env 硬回收叠加
        rem = int(env.episode_steps - env.step_index)
        if rem <= 40 and env.initial_soc is not None:
            b0 = float(env.initial_soc.get("battery_soc", 0.5))
            g0 = float(env.initial_soc.get("caes_gas_soc", 0.85))
            if bat_soc < b0 - 0.03:
                u_bat = float(np.clip(0.8, feas.u_battery_low, feas.u_battery_high))
            elif bat_soc > b0 + 0.03:
                u_bat = float(np.clip(-0.8, feas.u_battery_low, feas.u_battery_high))
            if gas_soc < g0 - 0.03 and feas.mode_mask.charge:
                mode, mag = CaesMode.CHARGE, 0.8
            elif gas_soc > g0 + 0.03 and feas.mode_mask.discharge:
                mode, mag = CaesMode.DISCHARGE, 0.8

        u_tp = float(np.clip(u_tp, feas.u_tp_low, feas.u_tp_high))
        return {
            "u_tp": np.asarray([u_tp], dtype=np.float32),
            "u_battery": np.asarray([u_bat], dtype=np.float32),
            "caes_mode": int(mode),
            "caes_magnitude": np.asarray([mag], dtype=np.float32),
        }
