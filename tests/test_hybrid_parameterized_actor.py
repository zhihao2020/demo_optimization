"""Parameterized CAES (mode, mag) heads never emit a masked-illegal band."""
from __future__ import annotations

import torch

from actions.caes_u import (
    CHARGE_LO,
    is_legal_u_caes,
    mag_from_u_torch,
    mode_from_u,
    mode_index_from_u_torch,
    perturb_u_caes_keep_mode,
)
from actions.types import CaesMode
from training.hybrid_common.stochastic_actor import HybridStochasticActor
from training.hybrid_td3.actor import HybridActor


def _bounds(n: int = 8):
    ones = torch.ones(n)
    return (
        ones * (1.0 / 3.0),
        ones,
        -ones,
        ones,
    )


def test_sac_actor_masked_charge_never_charges():
    actor = HybridStochasticActor(6, parameterized_caes=True)
    actor.eval()
    obs = torch.zeros(32, 6)
    mask = torch.ones(32, 3, dtype=torch.bool)
    mask[:, 2] = False  # no charge
    tp_lo, tp_hi, bat_lo, bat_hi = _bounds(32)
    with torch.no_grad():
        out = actor.act(obs, tp_lo, tp_hi, bat_lo, bat_hi, mask, deterministic=True)
        out_stoch = actor.act(obs, tp_lo, tp_hi, bat_lo, bat_hi, mask, deterministic=False)
    for u in torch.cat([out["u_caes"], out_stoch["u_caes"]]).tolist():
        assert is_legal_u_caes(float(u))
        assert mode_from_u(float(u)) != CaesMode.CHARGE
        assert float(u) < CHARGE_LO


def test_sac_actor_masked_discharge_never_discharges():
    actor = HybridStochasticActor(6, parameterized_caes=True)
    actor.eval()
    obs = torch.zeros(32, 6)
    mask = torch.ones(32, 3, dtype=torch.bool)
    mask[:, 0] = False
    tp_lo, tp_hi, bat_lo, bat_hi = _bounds(32)
    with torch.no_grad():
        out = actor.act(obs, tp_lo, tp_hi, bat_lo, bat_hi, mask, deterministic=False)
    for u in out["u_caes"].tolist():
        assert is_legal_u_caes(float(u))
        assert mode_from_u(float(u)) != CaesMode.DISCHARGE


def test_td3_actor_parameterized_legal():
    actor = HybridActor(4, parameterized_caes=True)
    actor.eval()
    obs = torch.randn(16, 4)
    mask = torch.ones(16, 3, dtype=torch.bool)
    tp_lo, tp_hi, bat_lo, bat_hi = _bounds(16)
    with torch.no_grad():
        out = actor.act(obs, tp_lo, tp_hi, bat_lo, bat_hi, mask, deterministic=True)
    for u in out["u_caes"].tolist():
        assert is_legal_u_caes(float(u))


def test_parameterized_actor_has_grad_through_u_caes():
    actor = HybridStochasticActor(5, parameterized_caes=True)
    obs = torch.randn(4, 5, requires_grad=False)
    tp_lo, tp_hi, bat_lo, bat_hi = _bounds(4)
    mask = torch.ones(4, 3, dtype=torch.bool)
    out = actor.act(obs, tp_lo, tp_hi, bat_lo, bat_hi, mask, deterministic=False)
    loss = out["u_caes"].sum() + out["log_prob"].sum()
    loss.backward()
    grads = [p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters() if p.requires_grad]
    assert any(grads)


def test_perturb_keeps_mode_band():
    u = torch.tensor([-0.7, 0.0, 0.95])
    noise = torch.tensor([0.5, 0.5, -0.5])
    out = perturb_u_caes_keep_mode(u, noise)
    assert mode_from_u(float(out[0])) == CaesMode.DISCHARGE
    assert abs(float(out[1])) <= 1e-6
    assert mode_from_u(float(out[2])) == CaesMode.CHARGE


def test_hybrid_sac_update_one_batch():
    from training.hybrid_sac.algorithm import HybridSAC

    class _Buf:
        def __len__(self):
            return 64

        def sample(self, batch_size: int):
            b = batch_size
            return {
                "obs": torch.zeros(b, 6).numpy(),
                "next_obs": torch.zeros(b, 6).numpy(),
                "u_tp": torch.full((b,), 0.5).numpy(),
                "u_battery": torch.zeros(b).numpy(),
                "u_caes": torch.zeros(b).numpy(),
                "reward": torch.zeros(b).numpy(),
                "done": torch.zeros(b).numpy(),
                "u_tp_low": torch.full((b,), 1.0 / 3.0).numpy(),
                "u_tp_high": torch.ones(b).numpy(),
                "u_bat_low": -torch.ones(b).numpy(),
                "u_bat_high": torch.ones(b).numpy(),
                "mode_mask": torch.ones(b, 3, dtype=torch.bool).numpy(),
                "next_u_tp_low": torch.full((b,), 1.0 / 3.0).numpy(),
                "next_u_tp_high": torch.ones(b).numpy(),
                "next_u_bat_low": -torch.ones(b).numpy(),
                "next_u_bat_high": torch.ones(b).numpy(),
                "next_mode_mask": torch.ones(b, 3, dtype=torch.bool).numpy(),
            }

    agent = HybridSAC(obs_dim=6, parameterized_caes=True, skip_nonfinite_update=True)
    metrics = agent.update(_Buf(), batch_size=8)
    assert metrics
    assert "critic_loss" in metrics


def test_mode_index_roundtrip():
    u = torch.tensor([-1.0, -0.33, 0.0, 0.86, 1.0])
    idx = mode_index_from_u_torch(u)
    assert idx.tolist() == [0, 0, 1, 2, 2]
    mag = mag_from_u_torch(u)
    assert float(mag[2]) == 0.0
    assert float(mag[0]) == 0.0 or float(mag[0]) >= 0.0
