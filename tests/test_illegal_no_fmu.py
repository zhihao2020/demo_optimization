"""非法动作不得触发 FMU set/do_step。"""

import numpy as np

from actions import CaesMode
from envs.power_system_env import PowerSystemEnv
from test_env_reset import FakeAdapter


def test_illegal_mode_does_not_call_fmu():
    adapter = FakeAdapter()
    env = PowerSystemEnv(adapter=adapter)
    env.reset(seed=0)
    # 强制近满储使 charge 非法
    env.last_outputs["caes_gas_soc"] = 0.99
    env.last_outputs["caes_hot_soc"] = 0.94
    env.last_outputs["caes_cold_soc"] = 0.94
    before = adapter.step_calls
    action = {
        "u_tp": np.asarray([1.0], dtype=np.float32),
        "u_battery": np.asarray([0.0], dtype=np.float32),
        "caes_mode": int(CaesMode.CHARGE),
        "caes_magnitude": np.asarray([1.0], dtype=np.float32),
    }
    _, _, _, truncated, info = env.step(action)
    assert adapter.step_calls == before
    assert info["transition_valid"] is False
    assert info["stored_in_replay"] is False
    assert info["fmu_status"] == "not_called"
    assert truncated is True
    env.close()
