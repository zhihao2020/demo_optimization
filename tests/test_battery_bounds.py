"""电池动态边界随 SOC 收窄。"""

from actions import FeasibilityOracle


def _out(soc: float):
    return {
        "battery_soc": soc,
        "caes_gas_soc": 0.85,
        "caes_hot_soc": 0.5,
        "caes_cold_soc": 0.5,
        "caes_gas_pressure": 8.5e6,
        "caes_gas_temperature": 300.0,
        "caes_hot_temperature": 400.0,
        "caes_cold_temperature": 290.0,
        "p_thermal": -100e6,
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


def test_near_max_soc_shrinks_charge():
    oracle = FeasibilityOracle.from_root()
    mid = oracle.compute(_out(0.5))
    high = oracle.compute(_out(0.87))
    assert high.u_battery_high < mid.u_battery_high
    assert high.u_battery_high < 0.5


def test_near_min_soc_shrinks_discharge():
    oracle = FeasibilityOracle.from_root()
    mid = oracle.compute(_out(0.5))
    low = oracle.compute(_out(0.13))
    assert low.u_battery_low > mid.u_battery_low
    assert low.u_battery_low > -0.5
