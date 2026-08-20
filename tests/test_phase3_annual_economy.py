"""Phase 3: intensity carbon quota, weekend shaping, aux obs, carbon goal dim."""
from __future__ import annotations

import math

import numpy as np

from envs.forecast_provider import AUX_FEATURE_DIM, DEFAULT_OBSERVATION_DIM
from envs.reward_calculator import RewardCalculator
from training.ghtd3.goals import (
    default_goal_boxes,
    enforce_budget_on_action,
    goal_budget_layout,
    goal_transition_intent,
)


def _base_cfg(**over):
    cfg = {
        "cost_reference": {"value": 1000.0},
        "episode_steps": 168,
        "terminal_soc": {"enabled": False},
        "carbon": {
            "enabled": True,
            "mode": "flat_tax",
            "price_cny_per_t": 80.0,
            "eta_thermal_t_per_mwh": 0.85,
            "eta_grid_t_per_mwh": 0.5703,
            "count_grid_import_only": True,
        },
        "curtailment": {"enabled": False},
        "battery_degradation": {"enabled": False},
        "caes_startup": {"enabled": False},
        "grid_contract": {"enabled": False},
    }
    for k, v in over.items():
        if k in cfg and isinstance(cfg[k], dict) and isinstance(v, dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    return cfg


def _cash(**kw):
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
        "p_caes": 0.0,
        "p_curtailment": 0.0,
        "p_unserved": 0.0,
        "battery_soc": 0.5,
        "caes_gas_soc": 0.5,
        "caes_hot_soc": 0.5,
        "caes_cold_soc": 0.5,
    }
    base.update(kw)
    return base


def test_aux_feature_dim_constant():
    assert AUX_FEATURE_DIM == 3
    assert DEFAULT_OBSERVATION_DIM == 19 + 96 + 48 + 3


def test_intensity_accumulates_and_settles_at_end():
    rc = RewardCalculator(
        _base_cfg(
            episode_steps=2,
            carbon={
                "mode": "intensity_benchmark",
                "beta_t_per_mwh": 0.82,
                "settle_at_episode_end": True,
                "grid_in_quota": False,
            },
        ),
        require_complete=False,
    )
    rc.reset(_cash())
    # 1 MWh thermal → allowance 0.82 t, emissions 0.85 t
    _, t1 = rc.calculate(
        _cash(p_thermal=1.0e6),
        is_final_step=False,
        episode_completed=False,
        no_failure=True,
        decision_interval_hours=1.0,
    )
    assert abs(t1["carbon_cost_cny"] - 0.0) < 1e-9  # no grid, no settlement yet
    assert abs(t1["carbon_allowance_t"] - 0.82) < 1e-9
    assert abs(t1["carbon_emissions_t"] - 0.85) < 1e-9
    assert abs(t1["carbon_position_t"] - (-0.03)) < 1e-9

    _, t2 = rc.calculate(
        _cash(p_thermal=0.0),
        is_final_step=True,
        episode_completed=True,
        no_failure=True,
        decision_interval_hours=1.0,
    )
    # Q = -0.03 → buy 0.03 t × 80 = 2.4 CNY cost
    assert abs(t2["carbon_settlement_cny"] - 2.4) < 1e-6
    assert abs(t2["carbon_cost_cny"] - 2.4) < 1e-6


def test_flat_tax_unchanged_default_mode():
    rc = RewardCalculator(_base_cfg(), require_complete=False)
    rc.reset(_cash())
    _, terms = rc.calculate(
        _cash(p_thermal=1.0e6),
        is_final_step=False,
        episode_completed=False,
        no_failure=True,
        decision_interval_hours=1.0,
    )
    assert abs(terms["carbon_cost_cny"] - 80.0 * 0.85) < 1e-6


def test_weekend_soft_anchor_raises_coef():
    rc = RewardCalculator(
        _base_cfg(
            episode_steps=336,
            terminal_soc={
                "enabled": True,
                "period_hours": 168,
                "mode": "binary_bonus",
                "bonus": 0.0,
                "tolerance": 0.06,
                "weights": {
                    "battery_soc": 1.0,
                    "caes_gas_soc": 1.0,
                    "caes_hot_soc": 0.0,
                    "caes_cold_soc": 0.0,
                },
                "shaping": {
                    "enabled": True,
                    "mode": "potential",
                    "coef": 5.0,
                    "absolute_coef": 0.0,
                    "recovery_horizon_steps": 0,
                    "weekend_soft_anchor": True,
                    "weekend_horizon_steps": 24,
                    "weekend_coef_scale": 3.0,
                },
            },
        ),
        require_complete=False,
    )
    outs0 = _cash(battery_soc=0.5, caes_gas_soc=0.5)
    rc.reset(outs0)
    # Mid-week (step 10): scale ~1
    _, mid = rc._soc_shaping(0.1, steps_done=10)
    # Last 24h of first week (step 160): into=160, week_rem=8 → in window
    _, wee = rc._soc_shaping(0.1, steps_done=160)
    assert mid["soc_recovery_scale"] == 1.0
    assert wee["soc_recovery_scale"] > 1.0


def test_aux_obs_tracks_progress_and_quota():
    rc = RewardCalculator(
        _base_cfg(
            episode_steps=100,
            carbon={
                "mode": "intensity_benchmark",
                "beta_t_per_mwh": 0.82,
                "q_norm_scale_t": 1.0,
            },
            battery_degradation={
                "enabled": True,
                "mode": "convex_cumulative",
                "e_cap_mwh": 400.0,
                "capex_cny_per_kwh": 1000.0,
                "n_cycles": 5000.0,
                "dod_eq": 0.8,
                "offset_frac_of_life": 0.0,
                "power_exp": 2.03,
            },
        ),
        require_complete=False,
    )
    rc.reset(_cash())
    aux0 = rc.aux_observation_features()
    assert aux0.shape == (3,)
    assert abs(aux0[2]) < 1e-9
    rc.step_in_episode = 50
    rc._carbon_allowance_t = 10.0
    rc._carbon_emissions_t = 0.0
    aux = rc.aux_observation_features()
    assert abs(aux[2] - 0.5) < 1e-6
    assert aux[0] > 0.9  # tanh(10/1)


def test_carbon_goal_layout_and_enforce():
    lay = goal_budget_layout(carbon_budget=True)
    assert lay == {"carbon": 2}
    low, high = default_goal_boxes(3, carbon_budget=True)
    assert low.shape == (3,)
    assert high[2] == 1.0
    out = enforce_budget_on_action(
        {"u_tp": 1.0, "u_battery": 0.0, "u_caes": 0.0},
        np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        carbon_budget=True,
        carbon_enforce=True,
        layout=lay,
        c=8,
    )
    # remain_frac=0 → u_tp clipped to u_min = 1/3
    assert abs(float(out["u_tp"]) - 1.0 / 3.0) < 1e-5
    g_next = goal_transition_intent(
        np.zeros(2, dtype=np.float32),
        np.asarray([0.1, 0.0, 0.5], dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        low,
        high,
        carbon_used=0.125,
        carbon_budget=True,
        layout=lay,
    )
    assert abs(float(g_next[2]) - 0.375) < 1e-5


def test_offset_default_zero_near_zero_marginal():
    rc = RewardCalculator(
        _base_cfg(
            battery_degradation={
                "enabled": True,
                "mode": "convex_cumulative",
                "e_cap_mwh": 400.0,
                "capex_cny_per_kwh": 1000.0,
                "n_cycles": 5000.0,
                "dod_eq": 0.8,
                "offset_frac_of_life": 0.0,
                "power_exp": 2.03,
            },
        ),
        require_complete=False,
    )
    par = rc._battery_deg_params(rc.config["battery_degradation"])
    assert par["delta_offset_mwh"] == 0.0
    rc.reset(_cash())
    _, terms = rc.calculate(
        _cash(p_battery=-1.0e6),
        is_final_step=False,
        episode_completed=False,
        no_failure=True,
        decision_interval_hours=1.0,
    )
    # Near δ=0 with p=2.03, marginal is small but cost of first MWh is positive
    assert terms["battery_deg_cost_cny"] > 0.0
    assert terms["battery_deg_marginal_cny_per_mwh"] < 1.0
    assert math.isfinite(terms["battery_deg_marginal_cny_per_mwh"])
