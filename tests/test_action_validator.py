"""Physical action validator + u_caes legal bands."""
from __future__ import annotations

import pytest

from actions import CaesMode, PhysicalActionValidator, PhysicalFmuAction
from actions.caes_u import is_legal_u_caes, project_u_caes, u_from_mode_mag
from actions.feasible_set import DynamicFeasibleActionSet
from actions.mode_mask import ModeMask
from envs.failures import StaticActionViolation


def test_project_gap_to_idle():
    assert project_u_caes(0.5) == 0.0
    assert is_legal_u_caes(project_u_caes(0.5))


def test_u_from_mode_mag_bounds():
    assert abs(u_from_mode_mag(CaesMode.DISCHARGE, 0.0) - (-1.0)) < 1e-9
    assert abs(u_from_mode_mag(CaesMode.DISCHARGE, 1.0) - (-0.33)) < 1e-9
    assert abs(u_from_mode_mag(CaesMode.IDLE, 0.7)) < 1e-9
    assert abs(u_from_mode_mag(CaesMode.CHARGE, 0.0) - 0.86) < 1e-9
    assert abs(u_from_mode_mag(CaesMode.CHARGE, 1.0) - 1.0) < 1e-9


def test_validator_rejects_illegal_band():
    v = PhysicalActionValidator()
    feas = DynamicFeasibleActionSet(
        u_tp_low=1/3, u_tp_high=1.0,
        u_battery_low=-1.0, u_battery_high=1.0,
        mode_mask=ModeMask(True, True, True),
    )
    # force illegal by constructing without project
    bad = PhysicalFmuAction(1.0, 0.0, 0.5)
    with pytest.raises(StaticActionViolation):
        v.validate(bad, feas)
