"""Scheme B comprehensive costs: carbon + CUT + convex battery deg."""
from __future__ import annotations

from envs.reward_calculator import RewardCalculator


def _base_cfg(**over):
    cfg = {
        "cost_reference": {"value": 1000.0},
        "episode_steps": 168,
        "terminal_soc": {"enabled": False},
        "carbon": {
            "enabled": True,
            "price_cny_per_t": 80.0,
            "eta_thermal_t_per_mwh": 0.85,
            "eta_grid_t_per_mwh": 0.5703,
            "count_grid_import_only": True,
        },
        "curtailment": {
            "enabled": True,
            "nu_curt_cny_per_mwh": 300.0,
            "nu_uns_cny_per_mwh": 1000.0,
        },
        "battery_degradation": {
            "enabled": True,
            "mode": "convex_cumulative",
            "power_exp": 2.03,
            "e_cap_mwh": 400.0,
            "capex_cny_per_kwh": 1000.0,
            "n_cycles": 5000.0,
            "dod_eq": 0.8,
            "offset_frac_of_life": 0.25,
            "battery_power_key": "p_battery",
            "discharge_negative_power": True,
        },
    }
    for k, v in over.items():
        if k in cfg and isinstance(cfg[k], dict) and isinstance(v, dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    return cfg


def _cash_outputs(**kw):
    base = {
        "economic_cashflow_total": 0.0,
        "economic_cashflow_wind": 0.0,
        "economic_cashflow_pv": 0.0,
        "economic_cashflow_thermal": 0.0,
        "economic_cashflow_battery": 0.0,
        "economic_cashflow_caes": 0.0,
        "economic_cashflow_load": 0.0,
        "economic_cashflow_grid": 0.0,
        "p_thermal": 0.0,
        "p_grid": 0.0,
        "p_battery": 0.0,
        "p_curtailment": 0.0,
        "p_unserved": 0.0,
        "battery_soc": 0.5,
        "caes_gas_soc": 0.5,
        "caes_hot_soc": 0.5,
        "caes_cold_soc": 0.5,
    }
    base.update(kw)
    return base


def _step(rc, o1):
    return rc.calculate(
        o1,
        is_final_step=False,
        episode_completed=False,
        no_failure=True,
        decision_interval_hours=1.0,
    )


def test_zero_external_when_no_power():
    rc = RewardCalculator(_base_cfg(), require_complete=False)
    rc.reset(_cash_outputs())
    r, terms = _step(rc, _cash_outputs(economic_cashflow_total=100.0))
    assert terms["carbon_cost_cny"] == 0.0
    assert terms["cut_total_cost_cny"] == 0.0
    assert terms["battery_deg_cost_cny"] == 0.0
    assert abs(terms["generalized_cashflow_delta"] - 100.0) < 1e-6


def test_carbon_and_cut():
    rc = RewardCalculator(_base_cfg(), require_complete=False)
    rc.reset(_cash_outputs())
    r, terms = _step(
        rc,
        _cash_outputs(
            economic_cashflow_total=0.0,
            p_thermal=1.0e6,
            p_curtailment=1.0e6,
        ),
    )
    carbon = 80.0 * 0.85
    cut = 300.0
    assert abs(terms["carbon_cost_cny"] - carbon) < 1e-6
    assert abs(terms["cut_total_cost_cny"] - cut) < 1e-6


def test_convex_deg_calibration_life_total():
    """a0 from Capex/E_life^p; full-life ψ equals Capex_total."""
    rc = RewardCalculator(_base_cfg(), require_complete=False)
    cfg = _base_cfg()["battery_degradation"]
    par = rc._battery_deg_params(cfg)
    e_life = par["e_life_mwh"]
    capex = par["capex_total_cny"]
    a0 = par["a0"]
    p = par["power_exp"]
    assert abs(a0 * (e_life**p) - capex) / capex < 1e-9
    # E_life = 5000*0.8*400 = 1.6e6 MWh; Capex = 1000*400*1000 = 4e8
    assert abs(e_life - 1.6e6) < 1e-6
    assert abs(capex - 4.0e8) < 1e-3


def test_convex_deg_positive_and_increasing_marginal():
    rc = RewardCalculator(_base_cfg(), require_complete=False)
    rc.reset(_cash_outputs())
    # two equal discharge steps: second should cost more (convex + offset)
    costs = []
    for _ in range(2):
        _, terms = _step(rc, _cash_outputs(p_battery=-1.0e6))  # 1 MWh discharge
        costs.append(terms["battery_deg_cost_cny"])
        assert terms["battery_deg_mode"] == 2.0
        assert terms["battery_discharge_mwh"] == 1.0
    assert costs[0] > 0.0
    assert costs[1] >= costs[0] - 1e-9
    # mid-life marginal should be order ~100 CNY/MWh (not ~0, not millions)
    assert 20.0 < terms["battery_deg_marginal_cny_per_mwh"] < 2000.0


def test_charge_does_not_increase_delta():
    rc = RewardCalculator(_base_cfg(), require_complete=False)
    rc.reset(_cash_outputs())
    _, t1 = _step(rc, _cash_outputs(p_battery=1.0e6))  # charge
    assert t1["battery_discharge_mwh"] == 0.0
    assert t1["battery_deg_cost_cny"] == 0.0
    assert t1["battery_deg_delta_mwh"] == 0.0


def test_linear_mode_still_works():
    rc = RewardCalculator(
        _base_cfg(battery_degradation={"mode": "linear_throughput", "discharge_only": True}),
        require_complete=False,
    )
    rc.reset(_cash_outputs())
    _, terms = _step(rc, _cash_outputs(p_battery=-1.0e6))
    # c_lin = Capex_total/E_life = 4e8/1.6e6 = 250
    assert abs(terms["c_deg_cny_per_mwh"] - 250.0) < 1e-6
    assert abs(terms["battery_deg_cost_cny"] - 250.0) < 1e-6
