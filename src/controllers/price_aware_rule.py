"""价格感知规则：谷段充电、峰/尖段放电，火电托底；动作投影到动态可行域。"""
from __future__ import annotations

from typing import Any

import numpy as np

from actions import CaesMode, FeasibilityOracle
from actions.caes_u import physical_dict, u_from_mode_mag
from controllers.rule_based_controller import RuleBasedController


class PriceAwareRuleController(RuleBasedController):
    """在可行域内做峰谷套利；无电价上下文时退化为父类保守策略。"""

    def __init__(
        self,
        env_or_space: Any = None,
        oracle: FeasibilityOracle | None = None,
        *,
        charge_threshold: float = 0.40,
        discharge_threshold: float = 0.90,
        battery_soc_low: float = 0.35,
        battery_soc_high: float = 0.75,
        charge_mag: float = 0.6,
        discharge_mag: float = 0.6,
    ) -> None:
        super().__init__(env_or_space, oracle)
        self.charge_threshold = charge_threshold
        self.discharge_threshold = discharge_threshold
        self.battery_soc_low = battery_soc_low
        self.battery_soc_high = battery_soc_high
        self.charge_mag = charge_mag
        self.discharge_mag = discharge_mag

    def _current_buy_price(self) -> float | None:
        env = self.env
        if env is None or getattr(env, "price_profile", None) is None:
            return None
        try:
            t = float(env.adapter.time)
            buy, _sell = env.price_profile.prices_at(t)
            return float(buy)
        except Exception:
            return None

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> dict:
        if self.env is None or self.env.last_outputs is None:
            return super().predict(observation, deterministic=deterministic)

        feasible = self.env.get_feasible_action_spec()
        outputs = self.env.last_outputs
        buy = self._current_buy_price()
        soc = float(outputs.get("battery_soc", 0.5))
        # 末段回收：靠近 episode 结束时停止套利（与 env.soc_recovery_horizon 对齐）
        step_idx = int(getattr(self.env, "step_index", 0) or 0)
        ep_len = int(getattr(self.env, "episode_steps", 168) or 168)
        rec_h = 48
        try:
            rec_h = int((self.env.config.get("market") or {}).get("soc_recovery_horizon", 48) or 48)
        except Exception:
            rec_h = 48
        recovery = step_idx >= max(ep_len - rec_h, 0)
        # 优先回到初始 SOC（若环境有），否则 0.5
        target_soc = 0.5
        if getattr(self.env, "initial_soc", None):
            target_soc = float(self.env.initial_soc.get("battery_soc", 0.5))

        # 火电：仍取可行上界（托底），谷时可略降给储能腾空间
        u_tp = float(feasible.u_tp_high)
        if buy is not None and buy <= self.charge_threshold and not recovery:
            u_tp = float(0.7 * feasible.u_tp_high + 0.3 * feasible.u_tp_low)

        # 电池：价低充、价高放；回收段朝 target_soc 修正
        u_bat = 0.0
        if feasible.u_battery_low <= 0.0 <= feasible.u_battery_high:
            u_bat = 0.0
        else:
            u_bat = 0.5 * (feasible.u_battery_low + feasible.u_battery_high)

        if recovery:
            if soc > target_soc + 0.05:
                u_bat = float(np.clip(-abs(self.discharge_mag), feasible.u_battery_low, feasible.u_battery_high))
            elif soc < target_soc - 0.05:
                u_bat = float(np.clip(abs(self.charge_mag), feasible.u_battery_low, feasible.u_battery_high))
        elif buy is not None:
            if buy <= self.charge_threshold and soc < self.battery_soc_high:
                target = abs(self.charge_mag)
                u_bat = float(np.clip(target, feasible.u_battery_low, feasible.u_battery_high))
            elif buy >= self.discharge_threshold and soc > self.battery_soc_low:
                target = -abs(self.discharge_mag)
                u_bat = float(np.clip(target, feasible.u_battery_low, feasible.u_battery_high))

        # CAES：峰放谷充；回收段尽量 IDLE
        mode = CaesMode.IDLE
        mag = 0.0
        if recovery:
            if not feasible.mode_mask.idle:
                if feasible.mode_mask.discharge:
                    mode = CaesMode.DISCHARGE
                    mag = 0.5
                elif feasible.mode_mask.charge:
                    mode = CaesMode.CHARGE
                    mag = 0.5
        elif buy is not None:
            if buy <= self.charge_threshold and feasible.mode_mask.charge:
                mode = CaesMode.CHARGE
                mag = 0.9
            elif buy >= self.discharge_threshold and feasible.mode_mask.discharge:
                mode = CaesMode.DISCHARGE
                mag = 0.9
            elif not feasible.mode_mask.idle:
                if feasible.mode_mask.discharge:
                    mode = CaesMode.DISCHARGE
                elif feasible.mode_mask.charge:
                    mode = CaesMode.CHARGE
        else:
            if not feasible.mode_mask.idle:
                if feasible.mode_mask.discharge:
                    mode = CaesMode.DISCHARGE
                elif feasible.mode_mask.charge:
                    mode = CaesMode.CHARGE

        return physical_dict(u_tp, u_bat, u_from_mode_mag(mode, mag))
