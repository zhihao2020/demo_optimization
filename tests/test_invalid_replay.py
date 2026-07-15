"""无效 transition 不得进入 replay。"""

import numpy as np

from envs.forecast_provider import DEFAULT_OBSERVATION_DIM

from actions import CaesMode
from envs.failures import (
    FmiLifecycleFailure,
    FmuNumericalFailure,
    NonFiniteOutputFailure,
    PostStepHardConstraintViolation,
)
from fmu.exceptions import FmuSolverError
from training.hybrid_td3.buffer import FilteredReplayBuffer, Transition
from training.hybrid_td3.collector import ValidTransitionCollector
from envs.power_system_env import PowerSystemEnv
from test_env_reset import FakeAdapter


def _valid_transition(**kwargs):
    base = Transition(
        observation=np.zeros(DEFAULT_OBSERVATION_DIM, dtype=np.float32),
        hybrid_action={"u_tp": 1.0, "u_battery": 0.0, "caes_mode": 1, "caes_magnitude": 0.0},
        decoded_fmu_action={"u_tp": 1.0, "u_battery": 0.0, "u_caes": 0.0},
        reward=-0.1,
        next_observation=np.zeros(DEFAULT_OBSERVATION_DIM, dtype=np.float32),
        terminated=False,
        valid_mode_mask=np.array([True, True, True]),
        dynamic_action_bounds={"u_tp_low": 1 / 3, "u_tp_high": 1.0, "u_battery_low": -1.0, "u_battery_high": 1.0},
        reward_terms={},
        physically_valid=True,
    )
    for k, v in kwargs.items():
        setattr(base, k, v)
    return base


def test_forbidden_and_failures_do_not_grow_buffer():
    buf = FilteredReplayBuffer(capacity=10)
    assert buf.add(_valid_transition(physically_valid=False)) is False
    assert len(buf) == 0
    assert buf.add(_valid_transition()) is True
    assert len(buf) == 1
    # 模拟各类无效
    for _ in range(5):
        buf.add(_valid_transition(physically_valid=False))
    assert len(buf) == 1
    assert buf.rejected_count == 6


class FailingAdapter(FakeAdapter):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode

    def step(self, action):
        self.step_calls += 1
        if self.mode == "solver":
            raise FmuSolverError("nonlinear solver failure")
        if self.mode == "nan":
            out = self._out()
            out["battery_soc"] = float("nan")
            self.time += 3600
            return out
        if self.mode == "soc":
            out = self._out()
            out["battery_soc"] = 1.05
            self.time += 3600
            return out
        return super().step(action)


def test_collector_rejects_post_step_and_fmu_failures():
    for mode in ("solver", "nan", "soc"):
        buf = FilteredReplayBuffer()
        collector = ValidTransitionCollector(buf)
        env = PowerSystemEnv(adapter=FailingAdapter(mode))
        env.reset(seed=0)
        action = {
            "u_tp": np.asarray([1.0], dtype=np.float32),
            "u_battery": np.asarray([0.0], dtype=np.float32),
            "caes_mode": int(CaesMode.IDLE),
            "caes_magnitude": np.asarray([0.0], dtype=np.float32),
        }
        collector.step_and_store(env, action)
        assert len(buf) == 0
        env.close()
