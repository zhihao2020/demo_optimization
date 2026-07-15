"""CAES 模式 mask 边界场景。"""

from actions import FeasibilityOracle, CaesMode


def _base(**overrides):
    out = {
        "battery_soc": 0.5,
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
    out.update(overrides)
    return out


def test_near_full_forbids_charge():
    oracle = FeasibilityOracle.from_root()
    feasible = oracle.compute(_base(caes_gas_soc=0.99, caes_hot_soc=0.94, caes_cold_soc=0.94))
    assert feasible.mode_mask.charge is False
    assert feasible.mode_mask.idle is True


def test_near_empty_forbids_discharge():
    oracle = FeasibilityOracle.from_root()
    feasible = oracle.compute(_base(caes_gas_soc=0.61, caes_hot_soc=0.06, caes_cold_soc=0.06))
    assert feasible.mode_mask.discharge is False
    assert feasible.mode_mask.idle is True


def test_pressure_high_forbids_charge():
    oracle = FeasibilityOracle.from_root()
    feasible = oracle.compute(_base(caes_gas_pressure=1.05e7))
    assert feasible.mode_mask.charge is False


def test_pressure_low_forbids_discharge():
    oracle = FeasibilityOracle.from_root()
    feasible = oracle.compute(_base(caes_gas_pressure=5.5e6))
    assert feasible.mode_mask.discharge is False


def test_mask_allows_checks_mode():
    oracle = FeasibilityOracle.from_root()
    feasible = oracle.compute(_base(caes_gas_soc=0.99))
    assert not feasible.mode_mask.allows(CaesMode.CHARGE)
    assert feasible.mode_mask.allows(CaesMode.IDLE)
