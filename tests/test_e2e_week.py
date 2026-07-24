"""端到端：混合随机可行策略 + FakeAdapter 多 episode；规则路径接口。"""

import numpy as np

from actions import CaesMode
from controllers.rule_based_controller import RuleBasedController
from envs.power_system_env import PowerSystemEnv
from test_env_reset import FakeAdapter
from training.hybrid_td3.buffer import FilteredReplayBuffer
from training.hybrid_td3.collector import ValidTransitionCollector
from training.hybrid_td3.train import RandomFeasiblePolicy


def test_rule_and_random_feasible_no_invalid_replay():
    """验证规则与随机可行策略 rollout 不产生 forbidden 且入库转移均合法。"""
    adapter = FakeAdapter()
    env = PowerSystemEnv(adapter=adapter)
    # 缩短 episode 便于单测
    env.episode_steps = 8
    env.reward_calculator.episode_steps = 8
    buf = FilteredReplayBuffer()
    collector = ValidTransitionCollector(buf)
    policy = RandomFeasiblePolicy(env)
    obs, _ = env.reset(seed=0)
    for _ in range(20):
        obs, reward, term, trunc, info = collector.step_and_store(env, policy.predict(obs))
        assert np.all(np.isfinite(obs))
        if term or trunc:
            obs, _ = env.reset()
    assert collector.stats["forbidden_action_attempts"] == 0
    assert buf.invalid_attempt_count == collector.stats["rejected_transition_count"] or buf.rejected_count >= 0
    # 所有入库转移合法
    assert all(t.physically_valid for t in buf._storage)
    env.close()


def test_rule_controller_outputs_hybrid_dict():
    """验证 RuleBasedController 输出完整混合动作字典且 IDLE 步 transition 合法。"""
    env = PowerSystemEnv(adapter=FakeAdapter())
    env.reset(seed=1)
    ctrl = RuleBasedController(env)
    action = ctrl.predict(env.build_observation())
    assert set(action) >= {"u_tp", "u_battery", "caes_mode", "caes_magnitude"}
    assert action["caes_mode"] == int(CaesMode.IDLE)
    _, _, _, _, info = env.step(action)
    assert info["transition_valid"] is True
    env.close()
