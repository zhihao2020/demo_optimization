"""PSO ParametricPricePolicy must map u_battery onto Oracle [lo, hi] every hour."""

from __future__ import annotations

import numpy as np

from actions.feasible_set import DynamicFeasibleActionSet
from actions.mode_mask import ModeMask
from optimization.pso_fmu import ParametricPricePolicy


def _env(*, bat_lo: float, bat_hi: float, buy: float, hour: int = 10):
    class _Profile:
        def prices_at(self, _t):
            return float(buy), float(buy)

    class _Env:
        episode_steps = 168
        step_index = hour
        last_outputs = {"battery_soc": 0.5, "caes_gas_soc": 0.85}
        initial_soc = {"battery_soc": 0.5, "caes_gas_soc": 0.85}
        price_profile = _Profile()
        adapter = type("A", (), {"time": 3600.0 * step_index})()
        _feas = DynamicFeasibleActionSet(
            u_tp_low=1.0 / 3.0,
            u_tp_high=1.0,
            u_battery_low=float(bat_lo),
            u_battery_high=float(bat_hi),
            mode_mask=ModeMask(discharge=True, idle=True, charge=True),
            u_caes_discharge=(-1.0, -0.33),
            u_caes_charge=(0.86, 1.0),
        )

        def get_feasible_action_spec(self):
            return self._feas

    return _Env()


def _policy(env, *, recovery: float = 0.0) -> ParametricPricePolicy:
    # charge_th=0.30, discharge_th=0.90 so buy=0.50 is the mid-price band.
    return ParametricPricePolicy(env, np.array([0.30, 0.90, 0.60, 0.50, 0.50, recovery]))


def _u_bat(action) -> float:
    return float(np.asarray(action["u_battery"]).reshape(-1)[0])


def test_mid_price_maps_zero_onto_excluded_battery_idle():
    """Week-25 hour-93: Oracle u_battery in [0.013, 1], policy used to send 0."""
    env = _env(bat_lo=0.013, bat_hi=1.0, buy=0.50)
    action = _policy(env).predict(None)
    u = _u_bat(action)
    assert 0.013 - 1e-9 <= u <= 1.0 + 1e-9
    assert abs(u - 0.013) < 1e-6


def test_mid_price_keeps_zero_when_idle_is_feasible():
    env = _env(bat_lo=-1.0, bat_hi=1.0, buy=0.50)
    assert abs(_u_bat(_policy(env).predict(None))) < 1e-9


def test_low_price_charge_still_clipped_to_dynamic_high():
    env = _env(bat_lo=-0.2, bat_hi=0.15, buy=0.20)
    u = _u_bat(_policy(env).predict(None))
    assert abs(u - 0.15) < 1e-6


def test_high_price_discharge_still_clipped_to_dynamic_low():
    env = _env(bat_lo=-0.15, bat_hi=0.2, buy=1.20)
    u = _u_bat(_policy(env).predict(None))
    assert abs(u - (-0.15)) < 1e-6
