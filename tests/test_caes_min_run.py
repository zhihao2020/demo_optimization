from actions import CaesMinimumRunController, CaesMode, ModeMask
import numpy as np

from envs.power_system_env import PowerSystemEnv
from test_env_reset import FakeAdapter


def test_charge_locks_direction_for_four_successful_steps_with_variable_magnitude():
    ctrl = CaesMinimumRunController()
    mask, state = ctrl.constrain(ModeMask(), steps_remaining=8, step=0)
    assert mask.charge and state["caes_locked_mode"] is None
    for step in range(1, 5):
        completed = ctrl.record_success(CaesMode.CHARGE, step=step)
        if step < 4:
            mask, state = ctrl.constrain(ModeMask(), steps_remaining=8 - step, step=step)
            assert mask.charge and not mask.idle and not mask.discharge
            assert state["caes_locked_steps_completed"] == step
        else:
            assert completed and completed["completed"] is True and completed["steps"] == 4


def test_discharge_tail_cannot_start_and_unsafe_lock_is_interrupted():
    ctrl = CaesMinimumRunController()
    tail, _ = ctrl.constrain(ModeMask(), steps_remaining=3, step=0)
    assert tail.idle and not tail.charge and not tail.discharge
    ctrl.record_success(CaesMode.DISCHARGE, step=1)
    mask, state = ctrl.constrain(ModeMask(discharge=False, idle=True, charge=True), steps_remaining=7, step=1)
    assert state["caes_min_run_event"]["reason"] == "locked_mode_no_longer_safe"
    assert mask.idle and mask.charge and not mask.discharge
    assert ctrl.summary()["caes_min_run_interruption_count"] == 1


def _action(mode, magnitude=0.0):
    return {
        "u_tp": np.asarray([1.0], dtype=np.float32),
        "u_battery": np.asarray([0.0], dtype=np.float32),
        "caes_mode": int(mode),
        "caes_magnitude": np.asarray([magnitude], dtype=np.float32),
    }


def test_environment_rejects_idle_during_locked_run_without_fmu_step():
    adapter = FakeAdapter()
    env = PowerSystemEnv(adapter=adapter)
    env.episode_steps = 4
    env.reset()
    _, _, _, _, first = env.step(_action(CaesMode.DISCHARGE, 0.2))
    assert first["transition_valid"] is True
    before = adapter.step_calls
    _, _, _, truncated, rejected = env.step(_action(CaesMode.IDLE))
    assert truncated is True
    assert rejected["fmu_status"] == "not_called"
    assert adapter.step_calls == before
    env.close()


def test_environment_forbids_new_start_in_tail():
    adapter = FakeAdapter()
    env = PowerSystemEnv(adapter=adapter)
    env.episode_steps = 3
    env.reset()
    _, _, _, truncated, rejected = env.step(_action(CaesMode.DISCHARGE, 0.2))
    assert truncated is True
    assert rejected["fmu_status"] == "not_called"
    assert adapter.step_calls == 0
    env.close()
