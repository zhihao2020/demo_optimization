"""Unit tests for FS-HSAC feasible support actor/critic (Phase 1 gate)."""

from __future__ import annotations

import torch

from actions.mode_mask import ModeMask
from actions.feasible_set import DynamicFeasibleActionSet
from training.fs_hsac.action_support import (
    MODE_CHARGE,
    MODE_DISCHARGE,
    MODE_IDLE,
    feasible_to_support_dict,
    interval_log_jacobian,
    sigmoid_log_jacobian,
    support_from_feasible_batch,
)
from training.fs_hsac.actor import FSHSACActor
from training.fs_hsac.critic import FSHSACCritic


def _feas(
    *,
    discharge=True,
    idle=True,
    charge=True,
    dis=(-1.0, -0.5),
    chg=(0.9, 1.0),
):
    return DynamicFeasibleActionSet(
        u_tp_low=0.4,
        u_tp_high=1.0,
        u_battery_low=-0.8,
        u_battery_high=0.8,
        mode_mask=ModeMask(discharge=discharge, idle=idle, charge=charge),
        u_caes_discharge=dis if discharge else None,
        u_caes_charge=chg if charge else None,
    )


def test_support_drops_narrow_band():
    feas = _feas(dis=(-0.4, -0.40001))  # span << MIN_SPAN
    d = feasible_to_support_dict(feas)
    # still discharge True from mask, but stack_supports will drop it
    support = support_from_feasible_batch([feas])
    assert bool(support["mode_mask"][0, MODE_DISCHARGE].item()) is False
    assert bool(support["mode_mask"][0, MODE_IDLE].item()) is True


def test_actor_samples_inside_dynamic_bands():
    actor = FSHSACActor(8)
    actor.eval()
    obs = torch.zeros(64, 8)
    feas = _feas(dis=(-0.9, -0.55), chg=(0.92, 0.99))
    support = support_from_feasible_batch([feas] * 64)
    with torch.no_grad():
        out = actor.act(obs, support, deterministic=False)
    for i in range(64):
        mode = int(out["mode_idx"][i])
        u = float(out["u_caes"][i])
        assert float(out["mode_probs"][i, mode]) > 0
        if mode == MODE_DISCHARGE:
            assert -0.9 - 1e-5 <= u <= -0.55 + 1e-5
        elif mode == MODE_CHARGE:
            assert 0.92 - 1e-5 <= u <= 0.99 + 1e-5
        else:
            assert abs(u) < 1e-5


def test_illegal_mode_probability_zero():
    actor = FSHSACActor(6)
    obs = torch.zeros(16, 6)
    feas = _feas(charge=False, discharge=True, idle=True)
    support = support_from_feasible_batch([feas] * 16)
    with torch.no_grad():
        out = actor.act(obs, support, deterministic=False)
    assert torch.all(out["mode_probs"][:, MODE_CHARGE] < 1e-6)
    assert torch.all(out["mode_idx"] != MODE_CHARGE)


def test_idle_has_zero_mag_entropy_contribution():
    actor = FSHSACActor(4)
    obs = torch.zeros(8, 4)
    feas = _feas(discharge=False, charge=False, idle=True)
    support = support_from_feasible_batch([feas] * 8)
    idle = actor.sample_mode_action(obs, support, MODE_IDLE, deterministic=False)
    # cont entropy includes tp/bat only; mag term is zeroed
    assert torch.all(idle["u_caes"].abs() < 1e-8)
    assert torch.all(idle["cont_dim"] == 2.0)


def test_log_jacobian_matches_finite_difference():
    z = torch.tensor([0.0, 1.0, -1.5], requires_grad=False)
    y = torch.sigmoid(z)
    # dy/dz = y(1-y)
    analytic = torch.exp(sigmoid_log_jacobian(z))
    expected = y * (1.0 - y)
    assert torch.allclose(analytic, expected, atol=1e-5)
    lo = torch.tensor([-1.0, -0.8, 0.86])
    hi = torch.tensor([-0.33, -0.5, 1.0])
    jac = torch.exp(interval_log_jacobian(lo, hi))
    assert torch.allclose(jac, (hi - lo), atol=1e-6)


def test_critic_distinguishes_same_mag_different_modes():
    critic = FSHSACCritic(5)
    obs = torch.zeros(2, 5)
    u_tp = torch.tensor([0.7, 0.7])
    u_bat = torch.tensor([0.0, 0.0])
    mag = torch.tensor([0.5, 0.5])
    # force different Q by setting different mode onehots; after random init they differ almost surely
    oh = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    q1, _ = critic(obs, u_tp, u_bat, oh, mag)
    assert not torch.allclose(q1[0], q1[1])


def test_actor_grad_flows():
    actor = FSHSACActor(5)
    obs = torch.randn(4, 5)
    support = support_from_feasible_batch([_feas()] * 4)
    out = actor.act(obs, support, deterministic=False)
    loss = out["log_prob"].mean() + out["u_caes"].mean()
    loss.backward()
    grads = [p.grad is not None and float(p.grad.abs().sum()) > 0 for p in actor.parameters()]
    assert any(grads)
