import numpy as np

from actions import CaesMode
from envs.forecast_provider import BASE_OBSERVATION_DIM, DEFAULT_OBSERVATION_DIM
from envs.power_system_env import PowerSystemEnv


class FakeAdapter:
    def __init__(self):
        self.time = 0
        self.closed = False
        self.set_calls = 0
        self.step_calls = 0

    def _out(self):
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

    def reset(self, start):
        self.time = start
        return self._out()

    def step(self, action):
        self.set_calls += 1
        self.step_calls += 1
        self.time += 3600
        return self._out()

    def close(self):
        self.closed = True


def _action(mode=CaesMode.IDLE, u_tp=1.0, u_bat=0.0, mag=0.0):
    return {
        "u_tp": np.asarray([u_tp], dtype=np.float32),
        "u_battery": np.asarray([u_bat], dtype=np.float32),
        "caes_mode": int(mode),
        "caes_magnitude": np.asarray([mag], dtype=np.float32),
    }


def test_reset_is_repeatable_and_releases_adapter():
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
    env = PowerSystemEnv(adapter=FakeAdapter(), forecast_enabled=False)
    observation, _ = env.reset(seed=7)
    assert observation.shape == (BASE_OBSERVATION_DIM,)
    assert env.observation_space.contains(observation)
    env.close()
