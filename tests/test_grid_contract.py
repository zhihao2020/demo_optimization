"""Story A grid-contract term in J_gen."""
from __future__ import annotations

from envs.reward_calculator import RewardCalculator


def _cfg(**over):
    cfg = {
        "cost_reference": {"value": 1000.0},
        "episode_steps": 168,
        "terminal_soc": {"enabled": False},
        "carbon": {"enabled": False},
        "curtailment": {"enabled": False},
        "battery_degradation": {"enabled": False},
        "caes_startup": {"enabled": False},
        "grid_contract": {
            "enabled": True,
            "p_lim_w": 2.0e8,
            "nu_cny_per_mwh": 600.0,
            "grid_power_key": "p_grid",
        },
    }
    cfg.update(over)
    return cfg


def _outs(p_grid: float) -> dict:
    return {
        "economic_cashflow_total": 0.0,
        "economic_cashflow_wind": 0.0,
        "economic_cashflow_pv": 0.0,
        "economic_cashflow_thermal": 0.0,
        "economic_cashflow_battery": 0.0,
        "economic_cashflow_caes": 0.0,
        "economic_cashflow_load": 0.0,
        "economic_cashflow_grid": 0.0,
        "p_grid": p_grid,
        "p_thermal": 0.0,
        "p_battery": 0.0,
        "p_curtailment": 0.0,
        "p_unserved": 0.0,
        "battery_soc": 0.5,
        "caes_gas_soc": 0.5,
        "caes_hot_soc": 0.5,
        "caes_cold_soc": 0.5,
    }


def test_within_contract_is_free():
    rc = RewardCalculator(_cfg(), require_complete=False)
    rc.reset(_outs(0.0))
    _, terms = rc.calculate(
        _outs(1.5e8),
        is_final_step=False,
        episode_completed=False,
        no_failure=True,
        decision_interval_hours=1.0,
    )
    assert terms["grid_contract_enabled"] == 1.0
    assert terms["grid_contract_excess_mwh"] == 0.0
    assert terms["grid_contract_cost_cny"] == 0.0


def test_export_over_limit_costs_nu_times_excess():
    rc = RewardCalculator(_cfg(), require_complete=False)
    rc.reset(_outs(0.0))
    # 350 MW export, limit 200 MW → 150 MWh * 600 = 90000
    _, terms = rc.calculate(
        _outs(-3.5e8),
        is_final_step=False,
        episode_completed=False,
        no_failure=True,
        decision_interval_hours=1.0,
    )
    assert abs(terms["grid_contract_excess_mwh"] - 150.0) < 1e-6
    assert abs(terms["grid_contract_cost_cny"] - 90000.0) < 1e-6
    assert abs(terms["generalized_cashflow_delta"] + 90000.0) < 1e-6


def test_import_over_limit_same_as_export():
    rc = RewardCalculator(_cfg(), require_complete=False)
    rc.reset(_outs(0.0))
    _, terms = rc.calculate(
        _outs(3.5e8),
        is_final_step=False,
        episode_completed=False,
        no_failure=True,
        decision_interval_hours=1.0,
    )
    assert abs(terms["grid_contract_cost_cny"] - 90000.0) < 1e-6


def test_disabled_contract_is_zero():
    rc = RewardCalculator(_cfg(grid_contract={"enabled": False}), require_complete=False)
    rc.reset(_outs(0.0))
    _, terms = rc.calculate(
        _outs(-5.0e8),
        is_final_step=False,
        episode_completed=False,
        no_failure=True,
        decision_interval_hours=1.0,
    )
    assert terms["grid_contract_enabled"] == 0.0
    assert terms["grid_contract_cost_cny"] == 0.0
