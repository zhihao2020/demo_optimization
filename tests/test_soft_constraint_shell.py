"""软约束外壳单测：预检可恢复、后验不重试、GiveSafe use_fallback 仍禁止。"""

from __future__ import annotations

import numpy as np
import pytest

from envs.power_system_env import PowerSystemEnv
from replay import HybridGiveSafeReplayBuffer
from safety import GiveSafeController, load_givesafe_config
from safety.soft_constraint_shell import (
    SoftConstraintEnv,
    SoftConstraintShell,
    conservative_recover_action,
    is_no_retry_failure,
    is_precheck_failure,
)
from test_env_reset import FakeAdapter
from training.hybrid_td3.givesafe_collector import GiveSafeTransitionCollector


def _idle():
    return {
        "u_tp": np.asarray([1.0], dtype=np.float32),
        "u_battery": np.asarray([0.0], dtype=np.float32),
        "u_caes": np.asarray([0.0], dtype=np.float32),
    }


def _charge_near_full():
    return {
        "u_tp": np.asarray([1.0], dtype=np.float32),
        "u_battery": np.asarray([0.0], dtype=np.float32),
        "u_caes": np.asarray([1.0], dtype=np.float32),
    }


def test_conservative_recover_prefers_idle():
    adapter = FakeAdapter()
    env = PowerSystemEnv(adapter=adapter)
    env.reset(seed=0)
    act = conservative_recover_action(env)
    assert float(act["u_tp"][0]) >= float(env.get_feasible_action_spec().u_tp_low)
    assert abs(float(act["u_caes"][0])) < 1e-6
    env.close()


def test_soft_env_recovers_precheck_and_advances_clock():
    """非法充电预检失败后，外壳用 idle 二次步进，时钟前进 1 h。"""
    adapter = FakeAdapter()
    env = PowerSystemEnv(adapter=adapter)
    env.reset(seed=0)
    env.last_outputs["caes_gas_soc"] = 0.99
    env.last_outputs["caes_hot_soc"] = 0.94
    env.last_outputs["caes_cold_soc"] = 0.94
    shell = SoftConstraintShell()
    wrapped = SoftConstraintEnv(env, shell)
    t0 = adapter.time
    steps0 = adapter.step_calls
    obs, reward, term, trunc, info = wrapped.step(_charge_near_full())
    assert info.get("soft_shell_applied") is True
    assert info.get("transition_valid") is True
    assert adapter.step_calls == steps0 + 1
    assert adapter.time == t0 + 3600
    assert shell.recovery_count == 1
    assert float(info.get("constraint_reward", 0)) < 0
    env.close()


def test_soft_env_no_retry_on_post_step():
    """后验硬约束失败不得二次步进。"""

    class FakeEnv:
        def __init__(self):
            self.calls = 0
            self.last_outputs = {"x": 1.0}
            self.previous_thermal = 0.0

        def get_feasible_action_spec(self):
            class F:
                u_tp_low, u_tp_high = 0.5, 1.0
                u_battery_low, u_battery_high = -1.0, 1.0

                class M:
                    idle = True
                    discharge = False
                    charge = False

                mode_mask = M()

            return F()

        def step(self, action):
            self.calls += 1
            info = {
                "transition_valid": False,
                "physically_valid": False,
                "failure_type": "PostStepHardConstraintViolation",
                "fmu_status": "failure",
                "action_executed_by_main_fmu": True,
            }
            return np.zeros(4), 0.0, False, True, info

        def reset(self, *a, **k):
            return np.zeros(4), {}

    fake = FakeEnv()
    wrapped = SoftConstraintEnv(fake)
    _, _, _, trunc, info = wrapped.step(_idle())
    assert trunc is True
    assert fake.calls == 1
    assert info.get("soft_shell_applied") is not True
    assert is_no_retry_failure(info)
    assert not is_precheck_failure(info)


def test_collector_soft_shell_recovers_no_safe_action():
    adapter = FakeAdapter()
    env = PowerSystemEnv(adapter=adapter)
    env.reset(seed=0)
    env.last_outputs["caes_gas_soc"] = 0.99
    env.last_outputs["caes_hot_soc"] = 0.94
    env.last_outputs["caes_cold_soc"] = 0.94
    buf = HybridGiveSafeReplayBuffer()
    ctrl = GiveSafeController(
        oracle=env.oracle,
        shadow=None,
        config={
            "use_fallback": False,
            "max_attempts_per_env_step": 3,
            "constraint_reward": {"base_rejection_cost": 1.0},
        },
    )
    collector = GiveSafeTransitionCollector(buf, ctrl, soft_shell=True)
    t0 = adapter.time

    def always_bad():
        return _charge_near_full()

    obs, reward, term, trunc, info = collector.step_with_givesafe(env, always_bad)
    assert collector.stats["soft_shell_recovery_count"] == 1
    assert info.get("soft_shell_applied") is True
    assert info.get("transition_type") == "physical"
    assert adapter.time == t0 + 3600
    assert buf.physical_size == 1
    env.close()


def test_use_fallback_still_forbidden():
    cfg = load_givesafe_config()
    assert cfg["givesafe"]["use_fallback"] is False
    with pytest.raises(RuntimeError):
        GiveSafeController(config={"use_fallback": True})
