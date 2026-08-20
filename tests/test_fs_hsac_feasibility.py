"""FS-HSAC replay + feasibility trainer tests (Phase 3 gate)."""

from __future__ import annotations

import numpy as np
import torch

from actions.feasible_set import DynamicFeasibleActionSet
from actions.mode_mask import ModeMask
from replay.fs_hsac_replay import FSHSACReplayBuffer
from training.fs_hsac.feasibility import FeasibilityTrainer, ResidualFeasibilityNet
from training.fs_hsac.algorithm import FSHSAC
from training.fs_hsac.action_support import one_hot_modes


def _feas():
    return DynamicFeasibleActionSet(
        u_tp_low=0.4,
        u_tp_high=1.0,
        u_battery_low=-1.0,
        u_battery_high=1.0,
        mode_mask=ModeMask(True, True, True),
        u_caes_discharge=(-1.0, -0.4),
        u_caes_charge=(0.9, 1.0),
    )


def test_missing_caes_bounds_raises():
    buf = FSHSACReplayBuffer()
    feas = _feas()
    buf.add_physical(
        obs=np.zeros(2, dtype=np.float32),
        next_obs=np.zeros(2, dtype=np.float32),
        action={"u_tp": [0.5], "u_battery": [0.0], "u_caes": [0.0]},
        reward=0.0,
        terminated=False,
        truncated=False,
        feasible=feas,
        next_feasible=feas,
    )
    # corrupt stored support
    buf.bellman._storage[0].support.pop("u_caes_discharge_low")
    try:
        buf.sample_bellman(1)
        assert False
    except KeyError:
        pass


def test_feasibility_labels_routed():
    buf = FSHSACReplayBuffer()
    feas = _feas()
    for _ in range(20):
        buf.add_physical(
            obs=np.zeros(4, dtype=np.float32),
            next_obs=np.zeros(4, dtype=np.float32),
            action={"u_tp": [0.6], "u_battery": [0.1], "u_caes": [0.0]},
            reward=0.2,
            terminated=False,
            truncated=False,
            feasible=feas,
            next_feasible=feas,
        )
    for _ in range(20):
        buf.add_rejection(
            obs=np.zeros(4, dtype=np.float32),
            action={"u_tp": [0.6], "u_battery": [0.1], "u_caes": [-1.0]},
            feasible=feas,
            failure_type="oracle_pressure",
        )
    fb = buf.sample_feasibility(32)
    labels = fb["feasibility_label"]
    assert set(np.unique(labels).tolist()) <= {0.0, 1.0}
    assert (labels == 0).any() and (labels == 1).any()


def test_classifier_penalty_pushes_away_from_unsafe():
    torch.manual_seed(0)
    net = ResidualFeasibilityNet(obs_dim=4)
    # craft net that scores charge mag high as unsafe-ish by training briefly
    trainer = FeasibilityTrainer(net, device="cpu", min_safe=5, min_unsafe=5)
    buf = FSHSACReplayBuffer()
    feas = _feas()
    for _ in range(40):
        buf.add_physical(
            obs=np.zeros(4, dtype=np.float32),
            next_obs=np.zeros(4, dtype=np.float32),
            action={"u_tp": [0.5], "u_battery": [0.0], "u_caes": [0.0]},
            reward=0.0,
            terminated=False,
            truncated=False,
            feasible=feas,
            next_feasible=feas,
        )
        buf.add_rejection(
            obs=np.zeros(4, dtype=np.float32),
            action={"u_tp": [0.5], "u_battery": [0.0], "u_caes": [-1.0]},
            feasible=feas,
            failure_type="oracle",
        )
    for _ in range(30):
        trainer.update(buf.sample_feasibility(32))

    agent = FSHSAC(obs_dim=4, use_feasibility_penalty=True, feasibility_beta=1.0)
    agent.feasibility_net = net
    obs = torch.zeros(8, 4)
    oh_safe = one_hot_modes(torch.ones(8, dtype=torch.long))
    oh_bad = one_hot_modes(torch.zeros(8, dtype=torch.long))
    mag = torch.ones(8) * 0.9
    u_tp = torch.ones(8) * 0.5
    u_bat = torch.zeros(8)
    pen_safe = agent._feasibility_penalty(obs, u_tp, u_bat, oh_safe, torch.zeros(8))
    pen_bad = agent._feasibility_penalty(obs, u_tp, u_bat, oh_bad, mag)
    # unsafe actions should have higher -log C (lower C)
    assert float(pen_bad.mean()) >= float(pen_safe.mean()) - 1e-5
