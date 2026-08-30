"""FMU CAES/battery device cash is excluded from plant J by default."""

from envs.reward_calculator import RewardCalculator


def _base_cfg(**extra):
    return {
        "episode_steps": 168,
        "cost_reference": {"value": 1000.0},
        "terminal_soc": {"enabled": False},
        **extra,
    }


def _cash(**overrides):
    data = {
        "battery_soc": 0.5,
        "caes_gas_soc": 0.5,
        "caes_hot_soc": 0.5,
        "caes_cold_soc": 0.5,
        "economic_cashflow_total": 0.0,
        "economic_cashflow_wind": 0.0,
        "economic_cashflow_pv": 0.0,
        "economic_cashflow_thermal": 0.0,
        "economic_cashflow_battery": 0.0,
        "economic_cashflow_caes": 0.0,
        "economic_cashflow_load": 0.0,
        "economic_cashflow_grid": 0.0,
    }
    data.update(overrides)
    return data


def test_default_strips_caes_and_battery_from_j():
    calc = RewardCalculator(_base_cfg())
    calc.reset(_cash())
    _, terms = calc.calculate(
        _cash(
            economic_cashflow_total=100.0,
            economic_cashflow_battery=-40.0,
            economic_cashflow_caes=-50.0,
            economic_cashflow_thermal=190.0,
        ),
        is_final_step=False,
        episode_completed=False,
        no_failure=True,
    )
    # FMU total +100 includes battery -40 and caes -50; J keeps only the rest (+190 thermal).
    assert terms["storage_device_cash_excluded"] == 1.0
    assert abs(terms["fmu_battery_cashflow_delta"] - (-40.0)) < 1e-9
    assert abs(terms["fmu_caes_cashflow_delta"] - (-50.0)) < 1e-9
    assert abs(terms["storage_bookkeeping_removed_cny"] - (-90.0)) < 1e-9
    assert abs(terms["economic_cashflow_delta"] - 190.0) < 1e-9
    assert abs(terms["generalized_cashflow_delta"] - 190.0) < 1e-9
    assert abs(terms["economic_cashflow_delta_caes"] - (-50.0)) < 1e-9
    assert abs(terms["raw_caes_cost"] - 50.0) < 1e-9


def test_flag_off_keeps_legacy_storage_cash():
    calc = RewardCalculator(_base_cfg(exclude_storage_device_cash=False))
    calc.reset(_cash())
    _, terms = calc.calculate(
        _cash(
            economic_cashflow_total=100.0,
            economic_cashflow_battery=-40.0,
            economic_cashflow_caes=-50.0,
        ),
        is_final_step=False,
        episode_completed=False,
        no_failure=True,
    )
    assert terms["storage_device_cash_excluded"] == 0.0
    assert abs(terms["storage_bookkeeping_removed_cny"] - 0.0) < 1e-9
    assert abs(terms["economic_cashflow_delta"] - 100.0) < 1e-9
    assert abs(terms["generalized_cashflow_delta"] - 100.0) < 1e-9


def test_stripped_cash_still_subtracts_python_overlays():
    calc = RewardCalculator(
        _base_cfg(
            caes_startup={"enabled": True, "c_su_cny": 24.0, "p_cap_w": 1.5e8},
            grid_contract={"enabled": False},
        )
    )
    calc.reset(_cash())
    nxt = _cash(
        economic_cashflow_total=100.0,
        economic_cashflow_caes=-50.0,
        p_caes=1.5e8,
    )
    _, terms = calc.calculate(nxt, is_final_step=False, episode_completed=False, no_failure=True)
    # cash' = 100 - (-50) = 150; J_gen = 150 - 24 startup
    assert abs(terms["economic_cashflow_delta"] - 150.0) < 1e-9
    assert abs(terms["caes_startup_cost_cny"] - 24.0) < 1e-9
    assert abs(terms["generalized_cashflow_delta"] - 126.0) < 1e-9
