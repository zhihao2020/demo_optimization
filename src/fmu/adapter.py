"""FMU 生命周期适配器：只委托既有 FmuSession 管理 FMI 读写。"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .exceptions import FmuSolverError
from .session import FmuSession
from .variable_registry import VariableRegistry


class FmuAdapter:
    def __init__(self, fmu_path: Path, communication_step_seconds: float, registry: VariableRegistry):
        self.registry = registry
        self._session = FmuSession(
            fmu_path, step_size=communication_step_seconds,
            outputs=tuple(spec.fmu_name for spec in registry.read_outputs.values()),
        )

    @property
    def time(self) -> float:
        return self._session.time

    def reset(self, start_time: float) -> dict[str, float]:
        try:
            raw = self._session.reset(start_time)
            return self._map_outputs(raw)
        except Exception as exc:  # FMI Python wrapper gives heterogeneous exception types
            raise FmuSolverError(f"FMU reset 失败: {exc}") from exc

    def step(self, action: Mapping[str, float]) -> dict[str, float]:
        try:
            raw = self._session.step({spec.fmu_name: float(action[spec.name]) for spec in self.registry.actions})
            return self._map_outputs(raw)
        except Exception as exc:
            raise FmuSolverError(f"FMU doStep/read 失败 (t={self.time}): {exc}") from exc

    def _map_outputs(self, raw: Mapping[str, float]) -> dict[str, float]:
        return {name: float(raw[spec.fmu_name]) for name, spec in self.registry.read_outputs.items()}

    def close(self) -> None:
        self._session.close()
