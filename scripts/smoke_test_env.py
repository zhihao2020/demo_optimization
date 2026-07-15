from envs.power_system_env import PowerSystemEnv
import numpy as np

with PowerSystemEnv() as env:
    obs, info = env.reset(seed=0)
    print("reset", obs.shape, info["time"])
    # 使用可行混合动作，而非原始 Dict.sample（可能落入动态禁区）
    from controllers.rule_based_controller import RuleBasedController
    action = RuleBasedController(env).predict(obs)
    print(env.step(action))
