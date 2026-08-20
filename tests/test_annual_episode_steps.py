"""episode_steps 贯通：周窗口与连续年共用同一 env 路径。"""

from __future__ import annotations

from test_env_reset import FakeAdapter

from src.envs.power_system_env import PowerSystemEnv


def test_episode_steps_override_weekly_default() -> None:
    env = PowerSystemEnv(adapter=FakeAdapter(), episode_steps=8760, forecast_enabled=False)
    assert env.episode_steps == 8760
    assert env.reward_calculator.episode_steps == 8760
    assert int(env.reward_calculator.config["episode_steps"]) == 8760
    env.close()


def test_episode_steps_default_is_weekly() -> None:
    env = PowerSystemEnv(adapter=FakeAdapter(), forecast_enabled=False)
    assert env.episode_steps == 168
    env.close()
