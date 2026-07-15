import numpy as np

from envs.observation_builder import ObservationBuilder
from fmu.variable_registry import VariableRegistry, VariableSpec


def test_observation_is_physical_float32_and_not_normalized():
    registry = VariableRegistry((), {"power": VariableSpec("power", "p_grid", "W", "output"),
                                     "soc": VariableSpec("soc", "battery_soc", "1", "output", 0, 1)})
    builder = ObservationBuilder(registry)
    observation = builder.build({"power": 12_345_678.0, "soc": .5})
    assert observation.dtype == np.float32
    assert observation.shape == (2,)
    assert observation[0] == np.float32(12_345_678.0)
    assert np.isneginf(builder.low[0]) and np.isposinf(builder.high[0])
