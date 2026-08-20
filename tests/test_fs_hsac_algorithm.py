"""FS-HSAC algorithm unit tests (Phase 2 gate)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from actions.feasible_set import DynamicFeasibleActionSet
from actions.mode_mask import ModeMask
from replay.fs_hsac_replay import FSHSACReplayBuffer
from training.fs_hsac.action_support import MODE_CHARGE, feasible_to_support_dict
from training.fs_hsac.algorithm import ALGORITHM_VERSION, FSHSAC


def _feas(**kw):
    return DynamicFeasibleActionSet(
        u_tp_low=0.35,
        u_tp_high=1.0,
        u_battery_low=-1.0,
        u_battery_high=1.0,
        mode_mask=ModeMask(
            discharge=kw.get("discharge", True),
            idle=kw.get("idle", True),
            charge=kw.get("charge", True),
        ),
        u_caes_discharge=kw.get("dis", (-1.0, -0.4)),
        u_caes_charge=kw.get("chg", (0.88, 1.0)),
    )


def _fill_buffer(n: int = 64) -> FSHSACReplayBuffer:
    buf = FSHSACReplayBuffer(capacity=1000)
    feas = _feas()
    for i in range(n):
        obs = np.zeros(6, dtype=np.float32)
        nxt = np.zeros(6, dtype=np.float32)
        nxt[0] = float(i % 3) * 0.01
        action = {"u_tp": [0.6], "u_battery": [0.0], "u_caes": [0.0]}
        buf.add_physical(
            obs=obs,
            next_obs=nxt,
            action=action,
            reward=0.1,
            terminated=False,
            truncated=False,
            feasible=feas,
            next_feasible=feas,
        )
    return buf


def test_single_batch_update_finite():
    agent = FSHSAC(obs_dim=6, skip_nonfinite_update=True)
    buf = _fill_buffer(64)
    metrics = agent.update(buf, batch_size=16)
    assert metrics
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["actor_loss"])
    assert np.isfinite(metrics["alpha_d"])
    assert np.isfinite(metrics["alpha_c"])


def test_dual_alpha_independent():
    agent = FSHSAC(obs_dim=6)
    a0 = float(agent.alpha_d.item()), float(agent.alpha_c.item())
    buf = _fill_buffer(80)
    for _ in range(5):
        agent.update(buf, batch_size=16)
    a1 = float(agent.alpha_d.item()), float(agent.alpha_c.item())
    # at least one should move (very likely both)
    assert a0 != a1


def test_single_legal_mode_only_idle():
    agent = FSHSAC(obs_dim=4)
    buf = FSHSACReplayBuffer()
    feas = _feas(discharge=False, charge=False, idle=True)
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
    metrics = agent.update(buf, batch_size=8)
    assert np.isfinite(metrics["actor_loss"])


def test_save_load_roundtrip(tmp_path: Path):
    agent = FSHSAC(obs_dim=5)
    buf = _fill_buffer(40)
    # change obs dim mismatch avoided
    agent = FSHSAC(obs_dim=6)
    agent.update(buf, batch_size=8)
    path = tmp_path / "fs.pt"
    agent.save(path)
    agent2 = FSHSAC(obs_dim=6)
    agent2.load(path)
    assert agent2.total_it == agent.total_it


def test_load_rejects_old_version(tmp_path: Path):
    path = tmp_path / "bad.pt"
    torch.save({"algorithm_version": "hybrid_sac", "actor": {}}, path)
    agent = FSHSAC(obs_dim=3)
    try:
        agent.load(path)
        assert False, "should raise"
    except RuntimeError as exc:
        assert "Incompatible" in str(exc)


def test_rejection_never_in_bellman_sample():
    buf = FSHSACReplayBuffer()
    feas = _feas()
    buf.add_physical(
        obs=np.zeros(3, dtype=np.float32),
        next_obs=np.zeros(3, dtype=np.float32),
        action={"u_tp": [0.5], "u_battery": [0.0], "u_caes": [0.0]},
        reward=1.0,
        terminated=False,
        truncated=False,
        feasible=feas,
        next_feasible=feas,
    )
    buf.add_rejection(
        obs=np.zeros(3, dtype=np.float32),
        action={"u_tp": [0.5], "u_battery": [0.0], "u_caes": [-1.0]},
        feasible=feas,
        failure_type="oracle",
    )
    assert buf.bellman_size == 1
    assert buf.feasibility_size == 2
    batch = buf.sample_bellman(1)
    assert float(batch["u_caes"][0]) == 0.0
