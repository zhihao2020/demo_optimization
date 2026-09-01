"""FMU 生命周期适配器：只委托既有 FmuSession 管理 FMI 读写。"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .exceptions import FmuSolverError
from .session import FmuSession
from .variable_registry import VariableRegistry


class FmuAdapter:
    """FMU 适配器(FmuAdapter)。

    在 Python 变量名与 FMU 变量名之间映射，供 Gymnasium 环境逐步调用。
    动作与边界严格分离：``step(action, boundaries=...)``。
    """

    def __init__(
        self,
        fmu_path: Path,
        communication_step_seconds: float,
        registry: VariableRegistry,
        *,
        require_boundaries: bool = True,
    ) -> None:
        """构造适配器并创建底层 FmuSession。

        Args:
            fmu_path: ``.fmu`` 文件路径。
            communication_step_seconds: FMI 通信步长（秒）。
            registry: 变量注册表(VariableRegistry)。
            require_boundaries: 新 FMU 须为 True。
        """
        self.registry = registry
        self._session = FmuSession(
            fmu_path,
            step_size=communication_step_seconds,
            outputs=tuple(spec.fmu_name for spec in registry.read_outputs.values()),
            require_boundaries=require_boundaries,
        )

    @property
    def time(self) -> float:
        """当前仿真时刻（秒）。

        Returns:
            FmuSession 内部时钟。
        """
        return self._session.time

    @property
    def last_input_readback(self) -> dict[str, float]:
        """Last ``setFloat64`` then ``getFloat64`` of u_tp/u_battery/u_caes."""
        return dict(getattr(self._session, "last_input_readback", {}) or {})

    def reset(
        self,
        start_time: float,
        boundaries: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """重置 FMU 并返回初始输出（Python 变量名）。

        Args:
            start_time: 起始仿真时刻（秒）。
            boundaries: 可选，边界 FMU 名 -> 物理值。

        Returns:
            Python 名 -> 标量值的输出字典。

        Raises:
            FmuSolverError: FMU reset 或读输出失败。
        """
        try:
            raw = self._session.reset(
                start_time,
                boundaries=dict(boundaries) if boundaries is not None else None,
            )
            return self._map_outputs(raw)
        except Exception as exc:  # FMI Python wrapper gives heterogeneous exception types
            raise FmuSolverError(f"FMU reset 失败: {exc}") from exc

    def step(
        self,
        action: Mapping[str, float],
        boundaries: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """写入 Python 动作与边界、推进一步并读输出。

        Args:
            action: Python 动作名 -> 值（如 ``u_tp``）。
            boundaries: 边界 FMU 名 -> 物理值；新 FMU 必填。

        Returns:
            映射后的 FMU 输出字典。

        Raises:
            FmuSolverError: doStep 或读输出失败。
        """
        try:
            raw = self._session.step(
                {spec.fmu_name: float(action[spec.name]) for spec in self.registry.actions},
                boundaries=dict(boundaries) if boundaries is not None else None,
            )
            return self._map_outputs(raw)
        except Exception as exc:
            raise FmuSolverError(f"FMU doStep/read 失败 (t={self.time}): {exc}") from exc

    def _map_outputs(self, raw: Mapping[str, float]) -> dict[str, float]:
        """将 FMU 变量名输出映射为 Python 名。

        Args:
            raw: FMU 变量名 -> 标量值。

        Returns:
            Python 名 -> 标量值。
        """
        return {
            name: float(raw[spec.fmu_name])
            for name, spec in self.registry.read_outputs.items()
        }

    def close(self) -> None:
        """释放底层 FmuSession 资源。"""
        self._session.close()
