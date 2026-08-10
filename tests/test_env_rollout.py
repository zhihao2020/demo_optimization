"""环境 rollout 测试：step 保留混合动作、观测合法与 step_physical_for_test。"""

import numpy as np

from actions import CaesMode, PhysicalFmuAction
from envs.power_system_env import PowerSystemEnv
from test_env_reset import FakeAdapter


def _idle_action():
    """构造 IDLE 混合动作。

    Returns:
        env.step 接受的 IDLE 动作字典。
    """
    return {
        "u_tp": np.asarray([1.0], dtype=np.float32),
        "u_battery": np.asarray([0.0], dtype=np.float32),
        "u_caes": np.asarray([0.0], dtype=np.float32),
    }


def test_step_preserves_hybrid_action_and_check_env():
    """验证 step 回写 requested/decoded 字段，action_space.sample 与观测均在空间内。"""
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
    assert float(sample["u_caes"][0]) == 0.0
    env2 = PowerSystemEnv(adapter=FakeAdapter())
    env2.reset(seed=0)
    o2, _, _, _, info2 = env2.step(sample)
    assert info2["transition_valid"] is True
    assert env2.observation_space.contains(o2)
    env.close()
    env2.close()


def test_step_physical_for_test():
    """验证 step_physical_for_test 接受连续向量并解码 u_caes=0（IDLE）。"""
    env = PowerSystemEnv(adapter=FakeAdapter())
    env.reset(seed=0)
    obs, reward, term, trunc, info = env.step_physical_for_test(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert info["decoded_u_caes"] == 0.0
    assert info.get("transition_valid") is True
    env.close()
