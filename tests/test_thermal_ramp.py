"""火电动态爬坡范围随上一时刻出力变化。"""

from actions import FeasibilityOracle


def _out(p_thermal: float):
    return {
        "battery_soc": 0.5,
        "caes_gas_soc": 0.85,
        "caes_hot_soc": 0.5,
        "caes_cold_soc": 0.5,
        "caes_gas_pressure": 8.5e6,
        "caes_gas_temperature": 300.0,
        "caes_hot_temperature": 400.0,
        "caes_cold_temperature": 290.0,
        "p_thermal": p_thermal,
        "p_battery": 0.0,
        "p_caes": 0.0,
        "p_grid": 0.0,
        "p_wind_available": 0.0,
        "p_wind_actual": 0.0,
        "p_pv_available": 0.0,
        "p_pv_actual": 0.0,
        "p_load_actual": 100e6,
        "p_curtailment": 0.0,
        "p_unserved": 0.0,
    }


def test_thermal_ramp_depends_on_previous_output():
    oracle = FeasibilityOracle.from_root()
    # u = -p / P_cap; p=-150e6 => u=1; p=-50e6 => u=1/3
    at_max = oracle.compute(_out(-150e6), previous_thermal_w=-150e6)
    at_min = oracle.compute(_out(-50e6), previous_thermal_w=-50e6)
    assert at_max.u_tp_high <= 1.0 + 1e-9
    assert at_max.u_tp_low >= 1.0 - 0.16  # rate*3600 = 0.15
    assert at_min.u_tp_low >= 1.0 / 3.0 - 1e-9
    assert at_min.u_tp_high <= 1.0 / 3.0 + 0.16
    assert at_max.u_tp_low > at_min.u_tp_low
