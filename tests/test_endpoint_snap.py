"""Dynamic CAES interval endpoint contract (float32 actor vs float64 oracle)."""
from __future__ import annotations

import numpy as np
import pytest

from actions.caes_u import ENDPOINT_SNAP_HARD_FAIL, snap_to_interval_endpoint
from actions.feasibility_oracle import FeasibilityOracle
from actions.feasible_set import DynamicFeasibleActionSet
from actions.mode_mask import ModeMask
from actions.types import PhysicalFmuAction
from test_mode_mask import _base


def test_float32_hi_endpoint_snaps():
    hi = 0.9299999999999999
    u = float(np.float32(0.93))
    assert u == pytest.approx(0.93000000715)
    snapped, flag = snap_to_interval_endpoint(u, 0.86, hi)
    assert flag
    assert snapped == hi


def test_true_overshoot_does_not_snap():
    hi = 0.9299999999999999
    u = hi + 1e-4
    snapped, flag = snap_to_interval_endpoint(u, 0.86, hi)
    assert not flag
    assert snapped == u
    oracle = FeasibilityOracle.from_root()
    outputs = _base()
    base = oracle.compute(outputs)
    feasible = DynamicFeasibleActionSet(
        u_tp_low=base.u_tp_low,
        u_tp_high=base.u_tp_high,
        u_battery_low=base.u_battery_low,
        u_battery_high=base.u_battery_high,
        mode_mask=ModeMask(discharge=False, idle=True, charge=True),
        u_caes_discharge=None,
        u_caes_charge=(0.86, hi),
        metadata=base.metadata,
    )
    ok, reason = oracle.check_action_executable(
        PhysicalFmuAction(1.0, 0.0, u), outputs, feasible=feasible
    )
    assert ok is False
    assert reason is not None


def test_low_endpoint_and_discharge_negative():
    lo, hi = -1.0, -0.33000000000000007
    u = float(np.float32(-0.33))
    snapped, flag = snap_to_interval_endpoint(u, lo, hi)
    # float32(-0.33) may sit slightly off the float64 end; if so, snap.
    if flag:
        assert snapped in (lo, hi)
    u_low = lo - 1e-9
    s2, f2 = snap_to_interval_endpoint(u_low, lo, hi)
    assert f2
    assert s2 == lo
    u_far = lo - 1e-4
    s3, f3 = snap_to_interval_endpoint(u_far, lo, hi)
    assert not f3
    assert s3 == u_far


def test_discharge_mag1_float32_not_idled():
    """Stage C eval death: mag=1 on hi=-0.33 was projected to idle."""
    import torch
    from actions.caes_u import (
        DISCHARGE_HI,
        apply_mode_mask_to_u_torch,
        project_u_caes,
        project_u_caes_torch,
        u_from_mode_onehot_dynamic,
    )

    d_lo = torch.tensor([-0.9371875000000001])
    d_hi = torch.tensor([-0.33])
    mag = torch.tensor([1.0])
    onehot = torch.tensor([[1.0, 0.0, 0.0]])
    u = u_from_mode_onehot_dynamic(
        onehot, mag, d_lo, d_hi, torch.tensor([0.86]), torch.tensor([1.0])
    )
    masked = apply_mode_mask_to_u_torch(u, torch.tensor([[True, False, True]]))
    assert float(masked) < 0.0
    assert float(masked) <= DISCHARGE_HI + 1e-6
    interp = -0.32999998331069946
    assert project_u_caes(interp) < 0.0
    assert abs(project_u_caes(interp) - DISCHARGE_HI) <= 1e-7
    assert float(project_u_caes_torch(torch.tensor([interp]))) < 0.0


def test_true_gap_still_idles():
    from actions.caes_u import project_u_caes

    assert project_u_caes(0.2) == 0.0
    assert project_u_caes(-0.1) == 0.0


def test_charge_lo_float32_not_idled():
    import torch
    from actions.caes_u import CHARGE_LO, apply_mode_mask_to_u_torch, u_from_mode_onehot_dynamic

    u = u_from_mode_onehot_dynamic(
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([0.0]),
        torch.tensor([-1.0]),
        torch.tensor([-0.33]),
        torch.tensor([0.86]),
        torch.tensor([1.0]),
    )
    masked = apply_mode_mask_to_u_torch(u, torch.tensor([[True, False, True]]))
    assert float(masked) >= CHARGE_LO - 1e-6


def test_hard_fail_threshold_is_above_ulp():
    assert ENDPOINT_SNAP_HARD_FAIL > 1.0e-6
    snapped, flag = snap_to_interval_endpoint(0.93 + 1e-3, 0.86, 0.93)
    assert not flag


def test_oracle_accepts_float32_endpoint_after_snap():
    oracle = FeasibilityOracle.from_root()
    outputs = _base()
    feasible = oracle.compute(outputs)
    if feasible.u_caes_charge is None:
        pytest.skip("charge interval empty on mid state")
    lo, hi = feasible.u_caes_charge
    u = float(np.float32(hi))
    ok, reason = oracle.check_action_executable(
        PhysicalFmuAction(1.0, 0.0, u), outputs, feasible=feasible
    )
    assert ok, reason
