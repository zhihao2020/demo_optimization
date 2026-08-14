"""Wear / thermal block quotas: countdown, HER dims, hard enforcement."""

from __future__ import annotations

import numpy as np

from training.ghtd3.goals import (
    G_THB,
    G_WEAR,
    achieved_goal_from_cycle,
    clip_discharge_to_budget,
    clip_thermal_to_budget,
    default_goal_boxes,
    enforce_budget_on_action,
    goal_transition_intent,
    max_discharge_u,
)


def test_discharge_cap_zero_remain_forbids_discharge():
    assert clip_discharge_to_budget(-0.8, 0.0) == 0.0
    assert clip_discharge_to_budget(0.4, 0.0) == 0.4


def test_discharge_cap_matches_oracle_soc_step():
    # Full-power discharge drops ~0.235 SoC in one hour (device_params).
    full = max_discharge_u(1.0)
    assert full > 1.0
    one_hour = max_discharge_u(0.24)
    assert 0.9 < one_hour < 1.2
    tight = clip_discharge_to_budget(-1.0, 0.05)
    assert -0.25 < tight < -0.15


def test_thermal_cap_zero_remain_pins_floor():
    assert abs(clip_thermal_to_budget(1.0, 0.0) - (1.0 / 3.0)) < 1e-6
    assert clip_thermal_to_budget(1.0, 1.0) == 1.0


def test_goal_transition_counts_down_and_floors():
    low, high = default_goal_boxes(3, wear_budget=True)
    g = np.asarray([0.10, 0.02, 0.08], dtype=np.float32)
    nxt = goal_transition_intent(
        np.asarray([0.50, 0.80]),
        g,
        np.asarray([0.46, 0.81]),
        low,
        high,
        wear_used=0.10,
        wear_budget=True,
    )
    assert nxt[2] == 0.0
    np.testing.assert_allclose(nxt[:2], [0.14, 0.01], atol=1e-5)


def test_enforce_rewrites_action_dict():
    g = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    out = enforce_budget_on_action(
        {"u_tp": 0.8, "u_battery": -0.9, "u_caes": 0.0},
        g,
        wear_budget=True,
        wear_enforce=True,
    )
    assert float(out["u_battery"]) == 0.0
    assert float(out["u_tp"]) == 0.8


def test_achieved_goal_wear_is_used_not_thermal_inventory():
    low, high = default_goal_boxes(3, wear_budget=True)
    g = achieved_goal_from_cycle(
        np.asarray([0.70, 0.80, 0.50]),
        np.asarray([0.55, 0.82, 0.40]),
        goal_low=low,
        goal_high=high,
        wear_used=0.15,
    )
    assert g.size == 3
    np.testing.assert_allclose(g[0], -0.15, atol=1e-5)
    assert abs(float(g[G_WEAR]) - 0.15) < 1e-6
    assert g[G_WEAR] != -0.10
