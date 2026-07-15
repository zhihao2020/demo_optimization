"""FMU 生命周期适配器：只委托既有 FmuSession 管理 FMI 读写。

环境层通过本类使用「逻辑变量名」(env_config)，由 VariableRegistry 映射到 FMU 名。
异常统一为 FmuSolverError，便于上层区分硬失败路径。
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .exceptions import FmuSolverError
from .session import FmuSession
from .variable_registry import VariableRegistry


class FmuAdapter:
    """Gymnasium / 训练代码面对的薄封装：reset/step/close + 逻辑名输出。"""

    def __init__(self, fmu_path: Path, communication_step_seconds: float, registry: VariableRegistry):
        self.registry = registry
        self._session = FmuSession(
            fmu_path, step_size=communication_step_seconds,
            outputs=tuple(spec.fmu_name for spec in registry.outputs.values()),
        )

    @property
    def time(self) -> float:
        """当前通信点时间（秒）。"""
        return self._session.time

    def reset(self, start_time: float) -> dict[str, float]:
        """重新实例化并返回逻辑名输出字典。"""
        try:
            raw = self._session.reset(start_time)
            return self._map_outputs(raw)
        except Exception as exc:  # FMI Python wrapper gives heterogeneous exception types
            raise FmuSolverError(f"FMU reset 失败: {exc}") from exc

    def step(self, action: Mapping[str, float]) -> dict[str, float]:
        """一步通信：逻辑名动作 → FMU 输入名 → doStep → 逻辑名输出。"""
        try:
            raw = self._session.step({spec.fmu_name: float(action[spec.name]) for spec in self.registry.actions})
            return self._map_outputs(raw)
        except Exception as exc:
            raise FmuSolverError(f"FMU doStep/read 失败 (t={self.time}): {exc}") from exc

    def _map_outputs(self, raw: Mapping[str, float]) -> dict[str, float]:
        return {name: float(raw[spec.fmu_name]) for name, spec in self.registry.outputs.items()}

    def close(self) -> None:
        self._session.close()
