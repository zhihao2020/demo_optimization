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
    """构造默认合法 Transition，可按字段覆盖。

    Args:
        **kwargs: 覆盖 Transition 任意字段。

    Returns:
        填充默认值的 Transition 实例。
    """
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
    """验证 physically_valid=False 的转移不会进入 replay 缓冲区。"""
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
    """按模式注入 FMU 求解失败、NaN 或 SOC 越界的假适配器(FailingAdapter)。"""

    def __init__(self, mode: str):
        """初始化失败模式适配器。

        Args:
            mode: 失败模式(mode)："solver" | "nan" | "soc"。
        """
        super().__init__()
        self.mode = mode

    def step(self, action):
        """模拟一步并按 mode 注入失败或返回默认输出。

        Args:
            action: FMU 调度输入（部分模式未使用）。

        Returns:
            正常时为 FakeAdapter 默认输出；nan/soc 模式修改 SOC。

        Raises:
            FmuSolverError: mode 为 "solver" 时。
        """
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
    """验证 collector 对求解失败、NaN 输出与后验 SOC 越界均不入库。"""
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
