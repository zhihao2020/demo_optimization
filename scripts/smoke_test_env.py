"""环境 smoke：reset + 规则动作一步，快速确认 FMU/Python 路径可用。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from envs.power_system_env import PowerSystemEnv
import numpy as np

with PowerSystemEnv() as env:
    obs, info = env.reset(seed=0)
    print("reset", obs.shape, info["time"])
    # 使用可行混合动作，而非原始 Dict.sample（可能落入动态禁区）
    from controllers.rule_based_controller import RuleBasedController
    action = RuleBasedController(env).predict(obs)
    print(env.step(action))
