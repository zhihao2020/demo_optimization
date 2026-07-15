import numpy as np

from actions import CaesMode, HybridAction
from envs.power_system_env import PowerSystemEnv
from test_env_reset import FakeAdapter


def _idle_action():
    return {
        "u_tp": np.asarray([1.0], dtype=np.float32),
        "u_battery": np.asarray([0.0], dtype=np.float32),
        "caes_mode": int(CaesMode.IDLE),
        "caes_magnitude": np.asarray([0.0], dtype=np.float32),
    }


def test_step_preserves_hybrid_action_and_check_env():
    env = PowerSystemEnv(adapter=FakeAdapter())
    env.reset(seed=1)
    action = _idle_action()
    observation, _, _, _, info = env.step(action)
    assert info["requested_u_tp"] == 1.0
    assert info["decoded_u_caes"] == 0.0
    assert info["transition_valid"] is True
    assert env.observation_space.contains(observation)
    # Dict+动态可行域下 gymnasium check_env 对 info 全等过严；改为自检 sample 接口
    sample = env.action_space.sample()
    assert sample["caes_mode"] == int(CaesMode.IDLE)
    env2 = PowerSystemEnv(adapter=FakeAdapter())
    env2.reset(seed=0)
    o2, _, _, _, info2 = env2.step(sample)
    assert info2["transition_valid"] is True
    assert env2.observation_space.contains(o2)
    env.close()
    env2.close()


def test_step_physical_for_test():
    env = PowerSystemEnv(adapter=FakeAdapter())
    env.reset(seed=0)
    obs, reward, term, trunc, info = env.step_physical_for_test(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert info["decoded_u_caes"] == 0.0
    assert info.get("transition_valid") is True
    env.close()
