"""全年评估测试：验证 IdlePolicy 在 FakeAdapter 下精确覆盖 8760 小时。"""

import numpy as np

from actions import CaesMode
from envs.power_system_env import PowerSystemEnv
from test_env_reset import FakeAdapter
from training.evaluate_td3 import evaluate_annual_policy


class IdlePolicy:
    """始终 IDLE 的占位策略(IdlePolicy)，用于全年步数覆盖测试。"""

    def predict(self, _observation, deterministic=True):
        """返回 IDLE 混合动作。

        Args:
            _observation: 未使用的观测。
            deterministic: 未使用的确定性标志。

        Returns:
            IDLE 混合动作字典。
        """
        return {
            "u_tp": np.asarray([1.0], dtype=np.float32),
            "u_battery": np.asarray([0.0], dtype=np.float32),
            "caes_mode": int(CaesMode.IDLE),
            "caes_magnitude": np.asarray([0.0], dtype=np.float32),
        }


def test_annual_evaluation_covers_exactly_8760_hours_without_overrun():
    """验证全年评估恰好 8760 步、53 个窗口且无无效转移。"""
    env = PowerSystemEnv(adapter=FakeAdapter())
    try:
        result = evaluate_annual_policy(env, IdlePolicy(), annual_horizon_hours=8760)
    finally:
        env.close()
    assert result["windows"] == 53
    assert result["steps"] == 8760
    assert result["valid_steps"] == 8760
    assert result["invalid_transition_count"] == 0
