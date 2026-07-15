"""GiveSafe 单元测试：拒绝不调主 FMU、自环样本、重采样、禁止 fallback、最大尝试。"""

from __future__ import annotations

import numpy as np
import pytest

from actions import CaesMode, FeasibilityOracle, HybridAction
from envs.power_system_env import PowerSystemEnv
from replay import HybridGiveSafeReplayBuffer
from safety import (
    ConstraintRewardCalculator,
    GiveSafeConstraintChecker,
    GiveSafeController,
    NoSafeActionFoundError,
    SafetyCheckResult,
    ShadowFmuValidator,
)
from safety.givesafe_controller import load_givesafe_config
from test_env_reset import FakeAdapter
from training.hybrid_td3.buffer import Transition
from training.hybrid_td3.givesafe_collector import GiveSafeTransitionCollector


class CountingAdapter(FakeAdapter):
    """独立计数；不调用会双重计数的 super.step。"""

    def step(self, action):
        self.set_calls += 1
        self.step_calls += 1
        self.time += 3600
        return self._out()


def _idle():
    return {
        "u_tp": np.asarray([1.0], dtype=np.float32),
        "u_battery": np.asarray([0.0], dtype=np.float32),
        "caes_mode": int(CaesMode.IDLE),
        "caes_magnitude": np.asarray([0.0], dtype=np.float32),
    }


def _charge_near_full():
    return {
        "u_tp": np.asarray([1.0], dtype=np.float32),
        "u_battery": np.asarray([0.0], dtype=np.float32),
        "caes_mode": int(CaesMode.CHARGE),
        "caes_magnitude": np.asarray([1.0], dtype=np.float32),
    }


def test_rejection_does_not_call_main_fmu():
    adapter = CountingAdapter()
    env = PowerSystemEnv(adapter=adapter)
    env.reset(seed=0)
    env.last_outputs["caes_gas_soc"] = 0.99
    env.last_outputs["caes_hot_soc"] = 0.94
    env.last_outputs["caes_cold_soc"] = 0.94
    before = adapter.step_calls
    t0 = adapter.time
    checker = GiveSafeConstraintChecker(env.oracle)
    safety = checker.check(env.last_outputs, _charge_near_full(), env.previous_thermal)
    assert safety.safe is False
    assert adapter.step_calls == before
    assert adapter.time == t0
    env.close()


def test_givesafe_self_loop_sample():
    buf = HybridGiveSafeReplayBuffer()
    obs = np.zeros(19, dtype=np.float32)
    tr = Transition(
        observation=obs,
        hybrid_action={"u_tp": 1.0, "u_battery": 0.0, "caes_mode": 2, "caes_magnitude": 1.0},
        decoded_fmu_action={"u_tp": 1.0, "u_battery": 0.0, "u_caes": 1.0},
        reward=-1.5,
        next_observation=obs.copy(),
        terminated=False,
        truncated=False,
        valid_mode_mask=np.ones(3, dtype=bool),
        dynamic_action_bounds={"u_tp_low": 1 / 3, "u_tp_high": 1.0, "u_battery_low": -1.0, "u_battery_high": 1.0},
        reward_terms={
            "constraint_reward": -1.5,
            "economic_reward": 0.0,
            "terminal_soc_bonus": 0.0,
            "total_training_reward": -1.5,
        },
        physically_valid=False,
        transition_type="givesafe_rejection",
    )
    assert buf.add_givesafe_rejection(tr)
    assert buf.givesafe_size == 1
    assert buf.physical_size == 0
    assert np.allclose(tr.next_observation, tr.observation)
    assert tr.reward_terms["economic_reward"] == 0.0
    assert tr.reward_terms["terminal_soc_bonus"] == 0.0
    assert tr.reward_terms["constraint_reward"] < 0


def test_same_state_resample_then_one_fmu():
    adapter = CountingAdapter()
    env = PowerSystemEnv(adapter=adapter)
    env.reset(seed=0)
    # 近满储使 CHARGE 非法
    env.last_outputs["caes_gas_soc"] = 0.99
    env.last_outputs["caes_hot_soc"] = 0.94
    env.last_outputs["caes_cold_soc"] = 0.94
    buf = HybridGiveSafeReplayBuffer()
    ctrl = GiveSafeController(oracle=env.oracle, shadow=None, config={"use_fallback": False, "max_attempts_per_env_step": 8})
    collector = GiveSafeTransitionCollector(buf, ctrl, shadow=None)
    calls = {"n": 0}

    def propose():
        calls["n"] += 1
        if calls["n"] < 3:
            return _charge_near_full()
        return _idle()

    t0 = adapter.time
    obs, reward, term, trunc, info = collector.step_with_givesafe(env, propose)
    assert calls["n"] == 3
    assert adapter.step_calls == 1
    assert buf.givesafe_size == 2
    assert buf.physical_size == 1
    assert adapter.time == t0 + 3600
    assert info.get("transition_type") == "physical"
    env.close()


def test_no_fallback_on_unsafe():
    cfg = load_givesafe_config()
    assert cfg["givesafe"]["use_fallback"] is False
    with pytest.raises(RuntimeError):
        GiveSafeController(config={"use_fallback": True})


def test_max_attempts_no_safe_action():
    adapter = CountingAdapter()
    env = PowerSystemEnv(adapter=adapter)
    env.reset(seed=0)
    env.last_outputs["caes_gas_soc"] = 0.99
    env.last_outputs["caes_hot_soc"] = 0.94
    env.last_outputs["caes_cold_soc"] = 0.94
    buf = HybridGiveSafeReplayBuffer()
    ctrl = GiveSafeController(
        oracle=env.oracle,
        shadow=None,
        config={"use_fallback": False, "max_attempts_per_env_step": 5, "constraint_reward": {"base_rejection_cost": 1.0}},
    )
    collector = GiveSafeTransitionCollector(buf, ctrl)
    t0 = adapter.time

    def always_bad():
        return _charge_near_full()

    obs, reward, term, trunc, info = collector.step_with_givesafe(env, always_bad)
    assert adapter.step_calls == 0
    assert adapter.time == t0
    assert info["failure_type"] == "NoSafeActionFound"
    assert trunc is True
    assert buf.physical_size == 0
    assert buf.givesafe_size == 5
    assert collector.stats["no_safe_action_found_count"] == 1
    env.close()


def test_shadow_rejection_no_main_fmu():
    adapter = CountingAdapter()
    env = PowerSystemEnv(adapter=adapter)
    env.reset(seed=0)

    class FailShadow:
        def __init__(self):
            self.time = 0
            self.closed = False

        def reset(self, start):
            self.time = start
            return {}

        def step(self, action):
            raise RuntimeError("shadow nonlinear solver failure")

        def close(self):
            self.closed = True

    shadow = ShadowFmuValidator(factory=lambda: FailShadow(), oracle=env.oracle, enabled=True, mode="always")
    shadow.on_episode_reset(0.0)
    ctrl = GiveSafeController(oracle=env.oracle, shadow=shadow, config={"use_fallback": False, "max_attempts_per_env_step": 3})
    buf = HybridGiveSafeReplayBuffer()
    collector = GiveSafeTransitionCollector(buf, ctrl, shadow=shadow)
    t0 = adapter.time
    obs, reward, term, trunc, info = collector.step_with_givesafe(env, _idle)
    # idle 可能一级通过后被 shadow 拒绝 → NoSafe 或若继续采样仍只 idle
    assert adapter.step_calls == 0
    assert adapter.time == t0
    assert buf.physical_size == 0
    assert collector.stats["shadow_fmu_rejection_count"] >= 1 or info.get("failure_type") == "NoSafeActionFound"
    env.close()


def test_mixed_replay_sampling_fractions():
    buf = HybridGiveSafeReplayBuffer(physical_fraction=0.7, givesafe_fraction=0.3)
    obs = np.zeros(19, dtype=np.float32)
    bounds = {"u_tp_low": 1 / 3, "u_tp_high": 1.0, "u_battery_low": -1.0, "u_battery_high": 1.0}
    mask = np.ones(3, dtype=bool)
    for i in range(70):
        buf.add_physical(
            Transition(
                observation=obs,
                hybrid_action={"u_tp": 1.0, "u_battery": 0.0, "caes_mode": 1, "caes_magnitude": 0.0},
                decoded_fmu_action={"u_tp": 1.0, "u_battery": 0.0, "u_caes": 0.0},
                reward=-0.1,
                next_observation=obs + 0.01,
                terminated=False,
                valid_mode_mask=mask,
                dynamic_action_bounds=bounds,
                reward_terms={},
                physically_valid=True,
                transition_type="physical",
            )
        )
    for i in range(30):
        buf.add_givesafe_rejection(
            Transition(
                observation=obs,
                hybrid_action={"u_tp": 1.0, "u_battery": 0.0, "caes_mode": 2, "caes_magnitude": 1.0},
                decoded_fmu_action={"u_tp": 1.0, "u_battery": 0.0, "u_caes": 1.0},
                reward=-1.0,
                next_observation=obs,
                terminated=False,
                valid_mode_mask=mask,
                dynamic_action_bounds=bounds,
                reward_terms={"constraint_reward": -1.0},
                physically_valid=False,
                transition_type="givesafe_rejection",
            )
        )
    batch = buf.sample(100)
    assert batch["transition_type"].sum() == 30  # givesafe=1
    assert (batch["transition_type"] == 0).sum() == 70


def test_constraint_reward_not_1e9():
    calc = ConstraintRewardCalculator({"base_rejection_cost": 1.0, "weights": {"forbidden_mode": 3.0}})
    terms = calc.calculate(
        SafetyCheckResult(safe=False, violation_type="forbidden_mode", violation_severity=1.0, normalized_violations={"forbidden_mode": 1.0})
    )
    assert terms["constraint_reward"] > -1e6
    assert terms["economic_reward"] == 0.0
    assert abs(terms["constraint_reward"] - terms["total_training_reward"]) < 1e-9


def test_terminal_soc_not_advanced_by_rejections():
    adapter = CountingAdapter()
    env = PowerSystemEnv(adapter=adapter)
    env.episode_steps = 168
    env.reset(seed=0)
    env.last_outputs["caes_gas_soc"] = 0.99
    env.last_outputs["caes_hot_soc"] = 0.94
    env.last_outputs["caes_cold_soc"] = 0.94
    before = env.valid_episode_steps
    buf = HybridGiveSafeReplayBuffer()
    ctrl = GiveSafeController(
        oracle=env.oracle,
        shadow=None,
        config={"use_fallback": False, "max_attempts_per_env_step": 3},
    )
    collector = GiveSafeTransitionCollector(buf, ctrl)
    collector.step_with_givesafe(env, _charge_near_full)
    assert env.valid_episode_steps == before  # NoSafe truncates without physical step
    env.close()
