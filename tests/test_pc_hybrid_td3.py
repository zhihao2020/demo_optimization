"""PC-HybridTD3: actor in A_f(s), critic 3-D physical, target keeps mode."""
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
    assert out.get("grid_violation", 0) == 0
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


def test_critic_scores_physical_triple():
    c = HybridCritic(6, parameterized_caes=True)
    obs = torch.zeros(4, 6)
    q1, q2 = c(obs, torch.ones(4) * 0.5, torch.zeros(4), torch.zeros(4))
    assert q1.shape == (4,)
    assert torch.isfinite(q1).all() and torch.isfinite(q2).all()
    # First layer: obs 6 + (u_tp, u_bat, u_caes).
    assert c.q1[0].in_features == 9


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


def test_actor_returns_mode_and_magnitude_keys():
    actor = HybridActor(4, parameterized_caes=True)
    actor.eval()
    n = 8
    with torch.no_grad():
        out = actor.act(
            torch.zeros(n, 4),
            torch.full((n,), 0.4),
            torch.ones(n),
            -torch.ones(n),
            torch.ones(n),
            torch.ones(n, 3, dtype=torch.bool),
            deterministic=True,
        )
    assert "caes_mode_onehot" in out and "caes_magnitude" in out
    assert out["caes_mode_onehot"].shape == (n, 3)
    assert out["caes_magnitude"].shape == (n,)


def test_paper_yaml_disables_storage_use_and_givesafe_replay():
    from pathlib import Path
    import yaml

    root = Path(__file__).resolve().parents[1]
    paper = yaml.safe_load((root / "src/config/paper_pc_hybrid_td3.yaml").read_text(encoding="utf-8"))
    reward = yaml.safe_load((root / "src/config/reward_config.yaml").read_text(encoding="utf-8"))
    gs = yaml.safe_load((root / "src/config/givesafe_config.yaml").read_text(encoding="utf-8"))
    assert paper["reward"]["storage_use"] is False
    assert reward["storage_use"]["enabled"] is False
    assert abs(float(gs["replay_sampling"]["physical_fraction"]) - 1.0) < 1e-9
    assert abs(float(gs["replay_sampling"]["givesafe_fraction"])) < 1e-9
    proto = yaml.safe_load((root / "src/config/experiment_protocol.yaml").read_text(encoding="utf-8"))
    assert proto["weekly_split"]["train"] == 36
    assert proto["weekly_split"]["test"] == 8


def test_formal_eval_start_requires_explicit_week(monkeypatch):
    import os
    from training.episode_starts import eval_start_seconds

    monkeypatch.delenv("OPTIMAL_DEMO_EVAL_EPISODE_START", raising=False)
    monkeypatch.setenv("OPTIMAL_DEMO_TRAIN_WEEK_STARTS", "0,168")
    try:
        eval_start_seconds(formal=True)
        raise AssertionError("expected configuration error")
    except ValueError as exc:
        assert "TEST week" in str(exc)
    monkeypatch.setenv("OPTIMAL_DEMO_EVAL_EPISODE_START", "1848000")
    assert eval_start_seconds(formal=True) == 1848000.0


def test_economic_replay_never_samples_rejections_when_fraction_zero():
    import numpy as np
    from replay.hybrid_replay_buffer import HybridGiveSafeReplayBuffer
    from training.hybrid_td3.buffer import Transition

    buf = HybridGiveSafeReplayBuffer(capacity=32, physical_fraction=1.0, givesafe_fraction=0.0)
    z = np.zeros(4, dtype=np.float32)
    mask = np.ones(3, dtype=bool)
    bounds = {
        "u_tp_low": 0.33,
        "u_tp_high": 1.0,
        "u_battery_low": -1.0,
        "u_battery_high": 1.0,
    }

    def _tr(kind: str, reward: float, valid: bool) -> Transition:
        return Transition(
            observation=z,
            hybrid_action={"u_tp": 0.5, "u_battery": 0.0, "u_caes": 0.0},
            decoded_fmu_action={"u_tp": 0.5, "u_battery": 0.0, "u_caes": 0.0},
            reward=reward,
            next_observation=z,
            terminated=False,
            valid_mode_mask=mask,
            dynamic_action_bounds=bounds,
            reward_terms={},
            physically_valid=valid,
            transition_type=kind,
        )

    buf.add_givesafe_rejection(_tr("givesafe_rejection", -99.0, False))
    buf.add_physical(_tr("physical", 1.0, True))
    batch = buf.sample(8)
    assert np.allclose(batch["reward"], 1.0)
    assert np.all(batch["transition_type"] == 0)


def test_target_smoothing_keeps_caes_mode():
    from actions.caes_u import mode_from_u, u_from_mode_onehot_dynamic

    n = 16
    onehot = torch.tensor([[1, 0, 0]] * n, dtype=torch.float32)
    mag = torch.full((n,), 0.4)
    mag_n = (mag + 0.2 * torch.randn(n)).clamp(0.0, 1.0)
    u = u_from_mode_onehot_dynamic(
        onehot,
        mag_n,
        torch.full((n,), -1.0),
        torch.full((n,), -0.33),
        torch.full((n,), 0.86),
        torch.ones(n),
    )
    for val in u.tolist():
        assert mode_from_u(float(val)).name == "DISCHARGE"


def test_physical_from_dict_allows_diagnostic_magnitude_keys():
    from actions.validator import physical_from_dict

    phys = physical_from_dict(
        {
            "u_tp": 0.5,
            "u_battery": 0.0,
            "u_caes": 0.0,
            "caes_mode_onehot": [0.0, 1.0, 0.0],
            "caes_magnitude": 0.4,
        }
    )
    assert abs(float(phys.u_caes)) < 1e-8


def test_stage_c_gates_require_cost_and_unserved():
    from training.hybrid_td3.train import compute_stage_c_gates

    ok_eval = {
        "eval_failed": False,
        "eval_status": "ok",
        "weekly_raw_total_cost": 100.0,
        "fmu_failure_count": 0,
        "valid_steps": 168,
        "steps": 168,
        "metrics": {"unserved_energy_mwh": 0.0},
    }
    ok_random = {
        "eval_failed": False,
        "eval_status": "ok",
        "weekly_raw_total_cost": 150.0,
        "valid_steps": 168,
        "steps": 168,
    }
    passed = compute_stage_c_gates(
        last_metrics={"critic_loss": 0.1, "actor_loss": 0.2, "q1_mean": -3.0},
        step_log=[{"critic_loss": 0.2}],
        collector_stats={
            "post_step_hard_constraint_violation_count": 0,
            "main_fmu_unsafe_execution_count": 0,
        },
        eval_result=ok_eval,
        random_eval=ok_random,
    )
    assert passed["passed"] is True
    assert passed["c5_cost_better_than_random"] is True

    worse = dict(ok_eval)
    worse["weekly_raw_total_cost"] = 200.0
    fail_cost = compute_stage_c_gates(
        last_metrics={"critic_loss": 0.1},
        step_log=[],
        collector_stats={},
        eval_result=worse,
        random_eval=ok_random,
    )
    assert fail_cost["passed"] is False
    assert fail_cost["c5_cost_better_than_random"] is False

    nosafe = dict(ok_eval)
    nosafe["eval_failed"] = True
    nosafe["eval_status"] = "failed_no_safe_action"
    fail_safe = compute_stage_c_gates(
        last_metrics={"critic_loss": 0.1},
        step_log=[],
        collector_stats={},
        eval_result=nosafe,
        random_eval=ok_random,
    )
    assert fail_safe["c3_heldout_nosafe_zero"] is False
    assert fail_safe["passed"] is False

    unserved = dict(ok_eval)
    unserved["metrics"] = {"unserved_energy_mwh": 1.0}
    fail_uns = compute_stage_c_gates(
        last_metrics={"critic_loss": 0.1},
        step_log=[],
        collector_stats={},
        eval_result=unserved,
        random_eval=ok_random,
    )
    assert fail_uns["c4_unserved_approx_zero"] is False
    assert fail_uns["passed"] is False

    nan_m = compute_stage_c_gates(
        last_metrics={"critic_loss": float("nan")},
        step_log=[],
        collector_stats={},
        eval_result=ok_eval,
        random_eval=ok_random,
    )
    assert nan_m["c1_nan_inf_zero"] is False
    assert nan_m["passed"] is False

    short = dict(ok_eval)
    short["valid_steps"] = 48
    short["fmu_failure_count"] = 1
    fail_short = compute_stage_c_gates(
        last_metrics={"critic_loss": 0.1},
        step_log=[],
        collector_stats={},
        eval_result=short,
        random_eval=ok_random,
    )
    assert fail_short["c2_fmu_hard_zero"] is False
    assert fail_short["complete_week"] is False
    assert fail_short["c5_cost_better_than_random"] is False
    assert fail_short["passed"] is False
