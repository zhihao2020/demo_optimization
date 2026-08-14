"""滚动线性规划：符号、风速代理、不再抬火电。"""

from optimization.rolling_linprog import demand_mw, gen_mw, wind_power_proxy


def test_gen_mw_flips_fmu_negative_generation():
    assert abs(gen_mw(-1.5e8) - 150.0) < 1e-9
    assert abs(gen_mw(0.0)) < 1e-12


def test_demand_mw_keeps_positive_load():
    assert abs(demand_mw(2.0e8) - 200.0) < 1e-9
    assert abs(demand_mw(-1e6)) < 1e-12


def test_wind_proxy_cubic_and_cutout():
    assert wind_power_proxy(1.0) == 0.0
    assert wind_power_proxy(12.0) == 1.0
    assert wind_power_proxy(30.0) == 0.0
    mid = wind_power_proxy(6.0)
    assert 0.0 < mid < 0.2


def test_predict_does_not_force_thermal_at_week_end(monkeypatch):
    from optimization.rolling_linprog import RollingLinprogController

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

    class _Env:
        episode_steps = 168
        step_index = 160
        last_outputs = {
            "p_load_actual": 1.0e8,
            "p_wind_available": -2.0e7,
            "p_pv_available": 0.0,
            "p_thermal": -6.0e7,
            "battery_soc": 0.5,
            "caes_gas_soc": 0.85,
        }
        initial_soc = {"battery_soc": 0.5, "caes_gas_soc": 0.85}
        forecast_provider = None
        price_profile = None
        adapter = type("A", (), {"time": 0.0})()
        config = {"fmu": {"decision_interval_seconds": 3600}}

        def get_feasible_action_spec(self):
            return _Feas()

    env = _Env()
    ctl = RollingLinprogController(env)
    monkeypatch.setattr(ctl, "_solve", lambda: {"ok": 1.0, "p_tp": 50.0, "p_bat": 0.0, "p_gas": 0.0})
    out = ctl.predict(None)
    u_tp = float(out["u_tp"][0])
    assert u_tp < 0.5


def test_solve_succeeds_on_mock_env():
    from optimization.rolling_linprog import RollingLinprogController

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

    ctl = RollingLinprogController(_Env())
    sol = ctl._solve()
    assert sol["ok"] == 1.0
    assert 50.0 - 1e-3 <= sol["p_tp"] <= 150.0 + 1e-3
