"""Replay-only tests for FS-HSAC (alias of algorithm rejection guards)."""

from __future__ import annotations

import numpy as np

from actions.feasible_set import DynamicFeasibleActionSet
from actions.mode_mask import ModeMask
from replay.fs_hsac_replay import FSHSACReplayBuffer


def _feas():
    return DynamicFeasibleActionSet(
        u_tp_low=1.0 / 3.0,
        u_tp_high=1.0,
        u_battery_low=-1.0,
        u_battery_high=1.0,
        mode_mask=ModeMask(True, True, True),
        u_caes_discharge=(-1.0, -0.33),
        u_caes_charge=(0.86, 1.0),
    )


def test_bellman_batch_has_caes_intervals():
    buf = FSHSACReplayBuffer()
    feas = _feas()
    buf.add_physical(
        obs=np.zeros(5, dtype=np.float32),
        next_obs=np.ones(5, dtype=np.float32) * 0.01,
        action={"u_tp": [0.7], "u_battery": [-0.1], "u_caes": [0.95]},
        reward=0.5,
        terminated=False,
        truncated=False,
        feasible=feas,
        next_feasible=feas,
    )
    b = buf.sample_bellman(1)
    assert "dis_lo" in b and "chg_hi" in b and "next_dis_lo" in b
    assert abs(float(b["chg_lo"][0]) - 0.86) < 1e-5


def test_post_step_failure_not_in_bellman():
    buf = FSHSACReplayBuffer()
    feas = _feas()
    buf.add_post_step_failure(
        obs=np.zeros(3, dtype=np.float32),
        next_obs=np.zeros(3, dtype=np.float32),
        action={"u_tp": [0.5], "u_battery": [0.0], "u_caes": [-0.7]},
        feasible=feas,
        next_feasible=feas,
        failure_type="cold_tank",
    )
    assert buf.bellman_size == 0
    assert buf.feasibility_size == 1
