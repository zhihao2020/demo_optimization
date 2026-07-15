import numpy as np

from actions import CaesMode
from envs.power_system_env import PowerSystemEnv
from test_env_reset import FakeAdapter
from training.evaluate_td3 import evaluate_annual_policy


class IdlePolicy:
    def predict(self, _observation, deterministic=True):
        return {
            "u_tp": np.asarray([1.0], dtype=np.float32),
            "u_battery": np.asarray([0.0], dtype=np.float32),
            "caes_mode": int(CaesMode.IDLE),
            "caes_magnitude": np.asarray([0.0], dtype=np.float32),
        }


def test_annual_evaluation_covers_exactly_8760_hours_without_overrun():
    env = PowerSystemEnv(adapter=FakeAdapter())
    try:
        result = evaluate_annual_policy(env, IdlePolicy(), annual_horizon_hours=8760)
    finally:
        env.close()
    assert result["windows"] == 53
    assert result["steps"] == 8760
    assert result["valid_steps"] == 8760
    assert result["invalid_transition_count"] == 0
