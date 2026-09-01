"""环境 reset 测试：FakeAdapter 可复现性、观测维度与 forecast 开关。"""

import numpy as np

from actions import CaesMode
from envs.forecast_provider import BASE_OBSERVATION_DIM, DEFAULT_OBSERVATION_DIM
from envs.power_system_env import PowerSystemEnv


class FakeAdapter:
    """不加载真实 FMU 的假适配器(FakeAdapter)，返回固定物理输出。"""

    def __init__(self):
        """初始化计数器与默认仿真时刻。"""
        self.time = 0
        self.closed = False
        self.set_calls = 0
        self.step_calls = 0
        self.last_input_readback = {}

    def _out(self):
        """构造默认 FMU 输出快照。

        Returns:
            含 SOC、功率与经济现金流字段的字典。
        """
        return {
            "battery_soc": .5, "caes_gas_soc": .85, "caes_hot_soc": .5, "caes_cold_soc": .5,
            "caes_gas_pressure": 8.5e6, "caes_gas_temperature": 300., "caes_hot_temperature": 400., "caes_cold_temperature": 290.,
            "p_thermal": -150e6, "p_battery": 0., "p_caes": 0., "p_grid": 1e6,
            "p_wind_available": -2e6, "p_wind_actual": -2e6, "p_pv_available": 0., "p_pv_actual": 0.,
            "p_load_actual": 151e6, "p_curtailment": 0., "p_unserved": 0.,
            "economic_cashflow_total": 0., "economic_cashflow_wind": 0., "economic_cashflow_pv": 0.,
            "economic_cashflow_thermal": 0., "economic_cashflow_battery": 0., "economic_cashflow_caes": 0.,
            "economic_cashflow_load": 0., "economic_cashflow_grid": 0.,
        }

    def reset(self, start, boundaries=None):
        """重置仿真时刻并返回初始输出。

        Args:
            start: 起始时刻(秒)。
            boundaries: 可选边界字典（真实 FMU 路径使用；此处忽略）。

        Returns:
            初始 FMU 输出字典。
        """
        _ = boundaries
        self.time = start
        return self._out()

    def step(self, action, boundaries=None):
        """模拟一步 FMU 并递增时刻。

        Args:
            action: FMU 调度输入（未修改输出）。
            boundaries: 可选边界字典（真实 FMU 路径使用；此处忽略）。

        Returns:
            步进后 FMU 输出字典。
        """
        _ = boundaries
        self.set_calls += 1
        self.step_calls += 1
        self.time += 3600
        if isinstance(action, dict):
            u = action.get("u_caes", 0.0)
            try:
                u = float(np.asarray(u).reshape(-1)[0])
            except Exception:
                u = 0.0
            self.last_input_readback = {
                "u_tp": float(np.asarray(action.get("u_tp", 1.0)).reshape(-1)[0]),
                "u_battery": float(np.asarray(action.get("u_battery", 0.0)).reshape(-1)[0]),
                "u_caes": u,
            }
        return self._out()

    def close(self):
        """标记适配器已关闭。"""
        self.closed = True


def _action(mode=CaesMode.IDLE, u_tp=1.0, u_bat=0.0, mag=0.0):
    """构造混合动作字典。

    Args:
        mode: CAES 模式(CaesMode)。
        u_tp: 火电指令。
        u_bat: 电池指令。
        mag: CAES 幅值。

    Returns:
        env.step 接受的混合动作字典。
    """
    return {
        "u_tp": np.asarray([u_tp], dtype=np.float32),
        "u_battery": np.asarray([u_bat], dtype=np.float32),
        "u_caes": np.asarray([float((__import__("actions.caes_u", fromlist=["u_from_mode_mag"]).u_from_mode_mag(mode, mag)))], dtype=np.float32),
    }


def test_reset_is_repeatable_and_releases_adapter():
    """验证同 seed reset 观测可复现，且 close 后 adapter 标记 closed。"""
    adapter = FakeAdapter()
    env = PowerSystemEnv(adapter=adapter)
    first, _ = env.reset(seed=7)
    assert first.shape == (DEFAULT_OBSERVATION_DIM,)
    assert np.allclose(first[BASE_OBSERVATION_DIM:BASE_OBSERVATION_DIM + 4], [1.33 / 15.0, 0.0, (262.45 - 273.15) / 40.0, 203661340.2 / 3.0e8])
    env.step(_action())
    second, _ = env.reset(seed=7)
    assert np.array_equal(first, second)
    env.close()
    assert adapter.closed


def test_forecast_can_be_disabled_for_same_seed_baseline():
    """验证 forecast_enabled=False 时观测维度为 BASE 且无 forecast 后缀。"""
    env = PowerSystemEnv(adapter=FakeAdapter(), forecast_enabled=False)
    observation, _ = env.reset(seed=7)
    assert observation.shape == (BASE_OBSERVATION_DIM,)
    assert env.observation_space.contains(observation)
    env.close()
