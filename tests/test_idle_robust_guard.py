"""Idle one-step robust envelope: pressure / SOC / temperature, no predict(u=0)."""
from __future__ import annotations

import numpy as np

from actions.feasibility_oracle import FeasibilityOracle
from actions.types import PhysicalFmuAction
from test_env_reset import FakeAdapter
from test_mode_mask import _base
from training.hybrid_td3.actor import HybridActor
from envs.power_system_env import PowerSystemEnv


def test_idle_pressure_lower_and_upper():
    oracle = FeasibilityOracle.from_root()
    g = oracle.idle_robust_guards()
    small = 0.02 * g["pressure"]
    mid = _base()
    high_enough = g["pressure_min"] + g["pressure"] + small
    low = g["pressure_min"] + g["pressure"] - small
    assert high_enough > g["pressure_min"]
    assert low > g["pressure_min"]  # still above the physical floor
    feas_ok = oracle.compute(_base(caes_gas_pressure=high_enough))
    assert feas_ok.mode_mask.idle is True
    feas_bad = oracle.compute(_base(caes_gas_pressure=low))
    assert feas_bad.mode_mask.idle is False
    assert (feas_bad.metadata or {}).get("idle_forbidden_reason") == "idle_pressure_low"

    # Charge remains available near the lower idle bound.
    assert feas_bad.mode_mask.charge is True

    upper_ok = g["pressure_max"] - g["pressure"] - small
    upper_bad = g["pressure_max"] - g["pressure"] + small
    assert oracle.compute(_base(caes_gas_pressure=upper_ok)).mode_mask.idle is True
    feas_hi = oracle.compute(_base(caes_gas_pressure=upper_bad))
    assert feas_hi.mode_mask.idle is False
    assert (feas_hi.metadata or {}).get("idle_forbidden_reason") == "idle_pressure_high"


def test_pressure_below_idle_guard_but_above_physical_min():
    oracle = FeasibilityOracle.from_root()
    g = oracle.idle_robust_guards()
    p = 6.60e6
    assert g["pressure_min"] < p < g["pressure_min"] + g["pressure"]
    feas = oracle.compute(_base(caes_gas_pressure=p))
    assert feas.mode_mask.idle is False


def test_all_modes_infeasible_marks_empty():
    oracle = FeasibilityOracle.from_root()
    # Near-full gas forbids charge; near-min pressure forbids discharge and idle.
    feas = oracle.compute(
        _base(caes_gas_soc=0.99, caes_hot_soc=0.94, caes_gas_pressure=6.51e6)
    )
    assert feas.mode_mask.idle is False
    assert feas.mode_mask.discharge is False
    assert feas.mode_mask.charge is False
    assert oracle.is_feasible_set_empty(feas) is True
    assert bool((feas.metadata or {}).get("feasible_set_empty")) is True


def test_idle_soc_and_temperature_sides():
    oracle = FeasibilityOracle.from_root()
    g = oracle.idle_robust_guards()
    small_soc = 0.002
    gas_low = g["gas_min"] + g["gas"] - small_soc
    gas_ok = g["gas_min"] + g["gas"] + small_soc
    assert oracle.compute(_base(caes_gas_soc=gas_ok)).mode_mask.idle is True
    bad_gas = oracle.compute(_base(caes_gas_soc=gas_low))
    assert bad_gas.mode_mask.idle is False
    assert str((bad_gas.metadata or {}).get("idle_forbidden_reason")).startswith("idle_gas")

    hot_high = g["hot_max"] - g["hot"] + small_soc
    bad_hot = oracle.compute(_base(caes_hot_soc=hot_high))
    assert bad_hot.mode_mask.idle is False

    cold_low = g["cold_min"] + g["cold"] - small_soc
    bad_cold = oracle.compute(_base(caes_cold_soc=cold_low))
    assert bad_cold.mode_mask.idle is False

    # Finite temperature sides only (gas lower bound is -inf clamp).
    hot_t_hi = 550.0 - g["temp"] + 0.5
    bad_t = oracle.compute(_base(caes_hot_temperature=hot_t_hi))
    assert bad_t.mode_mask.idle is False
    assert "temperature" in str((bad_t.metadata or {}).get("idle_forbidden_reason"))


def test_masked_actor_does_not_emit_forbidden_idle():
    oracle = FeasibilityOracle.from_root()
    feas = oracle.compute(_base(caes_gas_pressure=6.60e6))
    assert feas.mode_mask.idle is False
    assert feas.mode_mask.charge or feas.mode_mask.discharge
    actor = HybridActor(8, parameterized_caes=True)
    actor.eval()
    obs = np.zeros(8, dtype=np.float32)
    packed = actor.act_numpy(obs, feas, deterministic=True)
    u = float(np.asarray(packed["u_caes"]).reshape(-1)[0])
    assert abs(u) > 1e-6


def test_oracle_rejects_idle_even_if_mask_is_stale_true():
    """GiveSafe checker path: do not admit idle via predict_next_state(u=0)."""
    oracle = FeasibilityOracle.from_root()
    outputs = _base(caes_gas_pressure=6.60e6)
    stale = oracle.compute(_base())  # mid pressure, idle allowed
    assert stale.mode_mask.idle is True
    ok, reason = oracle.check_action_executable(
        PhysicalFmuAction(1.0, 0.0, 0.0), outputs, feasible=stale
    )
    assert ok is False
    assert reason is not None
    assert "idle" in reason.lower() or "IDLE" in reason


def test_env_does_not_step_fmu_on_forbidden_idle():
    adapter = FakeAdapter()
    env = PowerSystemEnv(adapter=adapter)
    env.reset(seed=0)
    env.last_outputs["caes_gas_pressure"] = 6.60e6
    before = adapter.step_calls
    action = {
        "u_tp": np.asarray([1.0], dtype=np.float32),
        "u_battery": np.asarray([0.0], dtype=np.float32),
        "u_caes": np.asarray([0.0], dtype=np.float32),
    }
    _obs, _r, _term, truncated, info = env.step(action)
    assert adapter.step_calls == before
    assert info.get("transition_valid") is False
    assert info.get("fmu_status") == "not_called"
    assert truncated is True
    env.close()


def test_charge_then_weak_discharge_near_pmax_is_blocked():
    """Stage C 两条 false-safe：9.48 MPa 充电后切 u=-0.33，实测压力升过 9.50。"""
    oracle = FeasibilityOracle.from_root()
    out = _base(
        caes_gas_soc=0.9481939540520397,
        caes_gas_pressure=9481939.540520396,
        caes_hot_soc=0.5644694895075674,
        caes_cold_soc=0.43553051049243346,
        caes_gas_temperature=282.6202567294596,
        p_caes=150e6,
    )
    feas = oracle.compute(out)
    assert feas.mode_mask.charge is False
    assert feas.mode_mask.discharge is False
    ok, reason = oracle.check_action_executable(
        PhysicalFmuAction(0.3333333333333333, 0.0, -0.33), out
    )
    assert ok is False
    assert reason is not None


def test_charge_stops_before_idle_pressure_dead_zone():
    """停充点必须低于 idle 高压 envelope，避免冲进 9.11–9.50 只许放电的死区。"""
    oracle = FeasibilityOracle.from_root()
    g = oracle.idle_robust_guards()
    p_dead = g["pressure_max"] - g["pressure"] + 1.0e4  # just inside idle-forbidden band
    feas = oracle.compute(_base(caes_gas_pressure=p_dead, caes_gas_soc=p_dead / 1e7))
    assert feas.mode_mask.charge is False
    p_ok = 8.80e6
    feas_ok = oracle.compute(_base(caes_gas_pressure=p_ok, caes_gas_soc=0.88))
    assert feas_ok.mode_mask.charge is True
    assert feas_ok.mode_mask.idle is True


def test_empty_set_is_not_rescued_by_idle():
    oracle = FeasibilityOracle.from_root()
    feas = oracle.compute(
        _base(caes_gas_soc=0.99, caes_hot_soc=0.94, caes_gas_pressure=6.51e6)
    )
    assert oracle.is_feasible_set_empty(feas)
    # Explicitly keep idle false — never re-open idle as a last resort.
    assert feas.mode_mask.idle is False
