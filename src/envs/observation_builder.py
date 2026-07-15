from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from fmu.variable_registry import VariableRegistry


@dataclass(frozen=True)
class ObservationSpec:
    name: str
    source: str
    unit: str
    low: float
    high: float


class ObservationBuilder:
    def __init__(self, registry: VariableRegistry):
        self.specs = tuple(
            ObservationSpec(item.name, item.fmu_name, item.unit,
                            -np.inf if item.low is None else float(item.low),
                            np.inf if item.high is None else float(item.high))
            for item in registry.outputs.values()
        )

    def build(self, outputs: dict[str, float]) -> np.ndarray:
        missing = [item.name for item in self.specs if item.name not in outputs]
        if missing:
            raise KeyError(f"缺少观测输出: {missing}")
        values = np.asarray([outputs[item.name] for item in self.specs], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"观测含 NaN/Inf: {values!r}")
        return values.astype(np.float32)

    @property
    def low(self) -> np.ndarray:
        return np.asarray([item.low for item in self.specs], dtype=np.float32)

    @property
    def high(self) -> np.ndarray:
        return np.asarray([item.high for item in self.specs], dtype=np.float32)
