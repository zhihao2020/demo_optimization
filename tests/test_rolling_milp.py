"""滚动 MILP：min-load 带、互斥、与 linprog 同接口。"""

from __future__ import annotations

import pytest

from optimization.rolling_milp import RollingMilpController


def _mock_env():
    class _Env:
        episode_steps = 168
        step_index = 10
        last_outputs = {
            "p_load_actual": 1.2e8,
            "p_wind_available": -3.0e7,
            "p_pv_available": -1.0e7,
            "p_thermal": -8.0e7,
            "battery_soc": 0.5,
            "caes_gas_soc": 0.85,
        }
        initial_soc = {"battery_soc": 0.5, "caes_gas_soc": 0.85}
        forecast_provider = None
        price_profile = None
        adapter = type("A", (), {"time": 3600.0 * 10})()
        config = {"fmu": {"decision_interval_seconds": 3600}}

        class _Feas:
            u_tp_low = 0.333
            u_tp_high = 1.0
            u_battery_low = -1.0
            u_battery_high = 1.0

            class _Mask:
                charge = True
                discharge = True
                idle = True

            mode_mask = _Mask()

        def get_feasible_action_spec(self):
            return self._Feas()

    return _Env()


def test_milp_import_or_skip():
    try:
        from scipy.optimize import milp  # noqa: F401
    except Exception as exc:
        pytest.skip(f"scipy.milp unavailable: {exc}")


def test_milp_solve_succeeds_on_mock_env():
    test_milp_import_or_skip()
    ctl = RollingMilpController(_mock_env())
    # Short horizon for unit speed
    ctl.cfg.horizon = 4
    ctl.cfg.time_limit_s = 30.0
    sol = ctl._solve()
    assert sol["ok"] == 1.0, sol
    assert 50.0 - 1e-3 <= sol["p_tp"] <= 150.0 + 1e-3
    # If CAES is on, power must sit in min-load bands (proxy MW)
    p_gas = float(sol["p_gas"])
    if abs(p_gas) > 1e-6:
        ratio = abs(p_gas) / 150.0
        if p_gas > 0:
            assert ratio + 1e-6 >= 0.86
        else:
            assert ratio + 1e-6 >= 0.33


def test_milp_predict_returns_action_dict(monkeypatch):
    ctl = RollingMilpController(_mock_env())
    monkeypatch.setattr(
        ctl,
        "_solve",
        lambda: {"ok": 1.0, "p_tp": 80.0, "p_bat": 0.0, "p_gas": 0.0, "z_chg": 0.0, "z_dis": 0.0},
    )
    out = ctl.predict(None)
    assert "u_tp" in out and "u_battery" in out and "u_caes" in out
    assert abs(float(out["u_caes"][0])) < 1e-6
