"""Aligned HMSD: hybrid CAES (mode, mag), startup in J, CAES on-budget."""

from __future__ import annotations

import numpy as np
import torch
import yaml
from pathlib import Path

from actions.caes_u import (
    CHARGE_LO,
    DISCHARGE_HI,
    mode_from_u,
    startup_events,
    u_from_mode_mag,
)
from actions.types import CaesMode
from envs.reward_calculator import RewardCalculator
from training.ghtd3.goals import (
    caes_on_frac,
    default_goal_boxes,
    enforce_budget_on_action,
    goal_budget_layout,
)
from training.ghtd3.networks import LowLevelActor


def test_startup_events_match_cui_bits():
    assert startup_events(CaesMode.IDLE, CaesMode.CHARGE) == 1
    assert startup_events(CaesMode.CHARGE, CaesMode.IDLE) == 1
    assert startup_events(CaesMode.CHARGE, CaesMode.DISCHARGE) == 2
    assert startup_events(CaesMode.IDLE, CaesMode.IDLE) == 0


def test_u_from_mode_mag_lands_in_legal_band():
    u_ch = u_from_mode_mag(CaesMode.CHARGE, 0.0)
    u_dis = u_from_mode_mag(CaesMode.DISCHARGE, 1.0)
    assert abs(u_ch - CHARGE_LO) < 1e-6
    assert abs(u_dis - DISCHARGE_HI) < 1e-6
    assert u_from_mode_mag(CaesMode.IDLE, 0.9) == 0.0
    assert mode_from_u(u_ch) == CaesMode.CHARGE
    assert mode_from_u(u_dis) == CaesMode.DISCHARGE


def test_hybrid_actor_outputs_legal_u(monkeypatch=None):
    actor = LowLevelActor(8, goal_dim=4, hybrid_caes=True)
    actor.eval()
    obs = torch.zeros(2, 8)
    goal = torch.zeros(2, 4)
    mask = torch.ones(2, 3, dtype=torch.bool)
    out = actor.act(
        obs, goal,
        torch.full((2,), 1.0 / 3.0),
        torch.ones(2),
        -torch.ones(2),
        torch.ones(2),
        mask,
        deterministic=True,
    )
    u = out["u_caes"].detach().numpy()
    for ui in u:
        m = mode_from_u(float(ui))
        assert m in (CaesMode.IDLE, CaesMode.CHARGE, CaesMode.DISCHARGE)
        if m != CaesMode.IDLE:
            assert abs(float(ui)) >= 0.33 - 1e-5


def test_caes_budget_forces_idle():
    g = np.asarray([0.1, 0.0, 0.2, 0.0], dtype=np.float32)
    out = enforce_budget_on_action(
        {"u_tp": 0.8, "u_battery": -0.2, "u_caes": 0.9},
        g,
        wear_budget=True,
        caes_budget=True,
        caes_enforce=True,
        layout=goal_budget_layout(wear_budget=True, caes_budget=True),
    )
    assert float(out["u_caes"]) == 0.0
    assert caes_on_frac(0.9, c=8) == 0.125
    assert caes_on_frac(0.0, c=8) == 0.0


def test_aligned_goal_box_is_4d():
    low, high = default_goal_boxes(4, wear_budget=True, caes_budget=True)
    assert low.shape == (4,)
    assert high[2] == 0.50
    assert high[3] == 1.0
    lay = goal_budget_layout(wear_budget=True, caes_budget=True)
    assert lay == {"wear": 2, "caes": 3}


def test_cui2024_startup_scales_from_table2():
    from envs.reward_calculator import RewardCalculator

    base = {
        "c_su_usd_ref": 3.42,
        "p_ref_w": 8.0e5,
        "p_cap_w": 1.5e8,
        "usd_cny": 7.2,
    }
    none = RewardCalculator.caes_startup_unit_cny({**base, "scale_mode": "none"})
    linear = RewardCalculator.caes_startup_unit_cny({**base, "scale_mode": "linear_capacity"})
    assert abs(none - 3.42 * 7.2) < 1e-6
    assert abs(linear - 3.42 * (1.5e8 / 8.0e5) * 7.2) < 1e-6
    assert abs(linear - 4617.0) < 1e-6


def test_startup_enters_j_gen():
    cfg = {
        "episode_steps": 168,
        "cost_reference": {"value": 1000.0},
        "terminal_soc": {"enabled": False},
        "caes_startup": {"enabled": True, "c_su_cny": 8000.0, "p_cap_w": 1.5e8},
    }
    calc = RewardCalculator(cfg)
    base = {
        "battery_soc": 0.5, "caes_gas_soc": 0.8, "caes_hot_soc": 0.5, "caes_cold_soc": 0.5,
        "economic_cashflow_total": 0.0,
        "economic_cashflow_wind": 0.0, "economic_cashflow_pv": 0.0,
        "economic_cashflow_thermal": 0.0, "economic_cashflow_battery": 0.0,
        "economic_cashflow_caes": 0.0, "economic_cashflow_load": 0.0,
        "economic_cashflow_grid": 0.0,
        "p_caes": 0.0,
    }
    calc.reset(base)
    nxt = dict(base)
    nxt["p_caes"] = 0.9 * 1.5e8
    nxt["economic_cashflow_total"] = 100.0
    _, terms = calc.calculate(nxt, is_final_step=False, episode_completed=False, no_failure=True)
    assert terms["caes_startup_events"] == 1.0
    assert abs(terms["caes_startup_cost_cny"] - 8000.0) < 1e-6
    assert abs(terms["generalized_cashflow_delta"] - (100.0 - 8000.0)) < 1e-6


def test_aligned_yaml_loads():
    path = Path("src/config/ablation/ghtd3_aligned.yaml")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    g = cfg["ghtd3"]
    assert g["hybrid_caes"] is True
    assert g["caes_budget"] is True
    assert g["goal_dim"] == 4
