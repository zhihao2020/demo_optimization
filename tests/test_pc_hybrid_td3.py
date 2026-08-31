"""PC-HybridTD3: actor in A_f(s), critic 6-D, target keeps mode."""
from __future__ import annotations

import torch

from actions.caes_u import is_legal_u_caes, mode_from_u, u_from_mode_mag_feasible
from actions.feasible_set import DynamicFeasibleActionSet
from actions.mode_mask import ModeMask
from actions.types import CaesMode
from training.hybrid_td3.actor import HybridActor
from training.hybrid_td3.algorithm import HybridTD3
from training.hybrid_td3.critic import HybridCritic


def test_actor_decodes_inside_dynamic_discharge_band():
    actor = HybridActor(4, parameterized_caes=True)
    actor.eval()
    n = 32
    obs = torch.zeros(n, 4)
    mask = torch.ones(n, 3, dtype=torch.bool)
    dis_lo = torch.full((n,), -0.8)
    dis_hi = torch.full((n,), -0.5)
    chg_lo = torch.full((n,), 0.9)
    chg_hi = torch.full((n,), 0.98)
    with torch.no_grad():
        out = actor.act(
            obs,
            torch.full((n,), 0.4),
            torch.ones(n),
            -torch.ones(n),
            torch.ones(n),
            mask,
            deterministic=True,
            dis_lo=dis_lo,
            dis_hi=dis_hi,
            chg_lo=chg_lo,
            chg_hi=chg_hi,
        )
    for u in out["u_caes"].tolist():
        assert is_legal_u_caes(float(u))
        m = mode_from_u(float(u))
        if m == CaesMode.DISCHARGE:
            assert -0.8 - 1e-5 <= float(u) <= -0.5 + 1e-5
        elif m == CaesMode.CHARGE:
            assert 0.9 - 1e-5 <= float(u) <= 0.98 + 1e-5
        else:
            assert abs(float(u)) < 1e-5


def test_train_random_feasible_accepts_feasible_kw():
    from inspect import signature
    from training.hybrid_td3.train import RandomFeasiblePolicy

    assert "feasible" in signature(RandomFeasiblePolicy.predict).parameters


def test_stage_a_module_reports_zero_violations():
    from training.hybrid_td3.stage_a import run_stage_a_support

    out = run_stage_a_support(n=1024, seed=0)
    assert out["status"] == "completed"
    assert out["illegal_caes_mode"] == 0
    assert out["dynamic_bound_violation"] == 0
    assert out["nan"] == 0


def test_stage_a_ten_thousand_actions_stay_in_dynamic_support():
    actor = HybridActor(8, parameterized_caes=True)
    actor.eval()
    illegal = 0
    bound_viol = 0
    n = 256
    rounds = 40
    for i in range(rounds):
        torch.manual_seed(i)
        obs = torch.randn(n, 8)
        mask = torch.ones(n, 3, dtype=torch.bool)
        dis_lo = torch.full((n,), -0.9)
        dis_hi = torch.full((n,), -0.4)
        chg_lo = torch.full((n,), 0.88)
        chg_hi = torch.full((n,), 0.99)
        with torch.no_grad():
            out = actor.act(
                obs,
                torch.full((n,), 0.3),
                torch.ones(n),
                -torch.ones(n),
                torch.ones(n),
                mask,
                deterministic=False,
                dis_lo=dis_lo,
                dis_hi=dis_hi,
                chg_lo=chg_lo,
                chg_hi=chg_hi,
            )
        u = out["u_caes"]
        assert torch.isfinite(u).all()
        for j, val in enumerate(u.tolist()):
            if not is_legal_u_caes(float(val)):
                illegal += 1
            m = mode_from_u(float(val))
            if m == CaesMode.DISCHARGE and not (
                float(dis_lo[j]) - 1e-4 <= float(val) <= float(dis_hi[j]) + 1e-4
            ):
                bound_viol += 1
            if m == CaesMode.CHARGE and not (
                float(chg_lo[j]) - 1e-4 <= float(val) <= float(chg_hi[j]) + 1e-4
            ):
                bound_viol += 1
    assert illegal == 0
    assert bound_viol == 0


def test_static_support_ablation_uses_device_envelope():
    actor = HybridActor(4, parameterized_caes=True, use_dynamic_support=False)
    actor.eval()
    n = 16
    obs = torch.zeros(n, 4)
    mask = torch.ones(n, 3, dtype=torch.bool)
    with torch.no_grad():
        out = actor.act(
            obs,
            torch.full((n,), 0.4),
            torch.ones(n),
            -torch.ones(n),
            torch.ones(n),
            mask,
            deterministic=True,
            dis_lo=torch.full((n,), -0.8),
            dis_hi=torch.full((n,), -0.5),
            chg_lo=torch.full((n,), 0.9),
            chg_hi=torch.full((n,), 0.98),
        )
    for u in out["u_caes"].tolist():
        assert is_legal_u_caes(float(u))


def test_critic_hybrid_pack_dim():
    c = HybridCritic(6, parameterized_caes=True)
    obs = torch.zeros(4, 6)
    onehot = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=torch.float32)
    mag = torch.tensor([0.0, 0.3, 0.8, 0.0])
    q1, q2 = c(obs, torch.ones(4) * 0.5, torch.zeros(4), torch.zeros(4), mode_onehot=onehot, mag=mag)
    assert q1.shape == (4,)
    assert torch.isfinite(q1).all() and torch.isfinite(q2).all()


def test_target_update_finite_one_batch():
    agent = HybridTD3(obs_dim=6, parameterized_caes=True, device="cpu")

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
                "caes_mode": torch.ones(b, dtype=torch.long).numpy(),
                "caes_magnitude": torch.zeros(b).numpy(),
                "reward": torch.zeros(b).numpy(),
                "done": torch.zeros(b).numpy(),
                "u_tp_low": torch.full((b,), 1.0 / 3.0).numpy(),
                "u_tp_high": torch.ones(b).numpy(),
                "u_bat_low": -torch.ones(b).numpy(),
                "u_bat_high": torch.ones(b).numpy(),
                "mode_mask": torch.ones(b, 3, dtype=torch.bool).numpy(),
                "dis_lo": torch.full((b,), -1.0).numpy(),
                "dis_hi": torch.full((b,), -0.33).numpy(),
                "chg_lo": torch.full((b,), 0.86).numpy(),
                "chg_hi": torch.ones(b).numpy(),
                "next_u_tp_low": torch.full((b,), 1.0 / 3.0).numpy(),
                "next_u_tp_high": torch.ones(b).numpy(),
                "next_u_bat_low": -torch.ones(b).numpy(),
                "next_u_bat_high": torch.ones(b).numpy(),
                "next_mode_mask": torch.ones(b, 3, dtype=torch.bool).numpy(),
                "next_dis_lo": torch.full((b,), -1.0).numpy(),
                "next_dis_hi": torch.full((b,), -0.33).numpy(),
                "next_chg_lo": torch.full((b,), 0.86).numpy(),
                "next_chg_hi": torch.ones(b).numpy(),
            }

    metrics = agent.update(_Buf(), batch_size=16)
    assert metrics
    assert torch.isfinite(torch.tensor(metrics["critic_loss"]))


def test_random_feasible_uses_oracle_interval():
    feas = DynamicFeasibleActionSet(
        u_tp_low=0.4,
        u_tp_high=1.0,
        u_battery_low=-1.0,
        u_battery_high=1.0,
        mode_mask=ModeMask(discharge=True, idle=True, charge=True),
        u_caes_discharge=(-0.7, -0.4),
        u_caes_charge=(0.9, 0.97),
    )
    u = u_from_mode_mag_feasible(feas, CaesMode.CHARGE, 0.0)
    assert 0.9 - 1e-6 <= u <= 0.9 + 1e-6
    u2 = u_from_mode_mag_feasible(feas, CaesMode.DISCHARGE, 1.0)
    assert -0.7 - 1e-6 <= u2 <= -0.4 + 1e-6
