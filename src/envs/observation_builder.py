"""从 FMU 输出字典组装 Gymnasium 观测向量。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fmu.variable_registry import VariableRegistry


@dataclass(frozen=True)
class ObservationSpec:
    """单维观测规格(ObservationSpec)。

    Attributes:
        name: Python 侧观测名。
        source: 对应 FMU 变量名。
        unit: 物理单位。
        low: 观测下界（``None`` 映射为 ``-inf``）。
        high: 观测上界（``None`` 映射为 ``inf``）。
    """

    name: str
    source: str
    unit: str
    low: float
    high: float


class ObservationBuilder:
    """观测构造器(ObservationBuilder)。

    按 ``VariableRegistry`` 中输出顺序，将 FMU 字典转为 ``float32`` 向量。
    """

    def __init__(self, registry: VariableRegistry) -> None:
        """从变量注册表构建观测规格列表。

        Args:
            registry: FMU 变量注册表(VariableRegistry)。
        """
        self.specs = tuple(
            ObservationSpec(
                item.name,
                item.fmu_name,
                item.unit,
                -np.inf if item.low is None else float(item.low),
                np.inf if item.high is None else float(item.high),
            )
            for item in registry.outputs.values()
        )

    def build(self, outputs: dict[str, float]) -> np.ndarray:
        """组装观测向量。

        Args:
            outputs: Python 名 -> 标量值的 FMU 输出字典。

        Returns:
            与 ``specs`` 顺序一致的 ``float32`` 一维数组。

        Raises:
            KeyError: 缺少必需观测键。
            ValueError: 观测含 NaN/Inf。
        """
        missing = [item.name for item in self.specs if item.name not in outputs]
        if missing:
            raise KeyError(f"缺少观测输出: {missing}")
        values = np.asarray([outputs[item.name] for item in self.specs], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"观测含 NaN/Inf: {values!r}")
        return values.astype(np.float32)

    @property
    def low(self) -> np.ndarray:
        """各观测维度的 ``Box`` 下界。

        Returns:
            ``float32`` 下界数组。
        """
        return np.asarray([item.low for item in self.specs], dtype=np.float32)

    @property
    def high(self) -> np.ndarray:
        """各观测维度的 ``Box`` 上界。

        Returns:
            ``float32`` 上界数组。
        """
        return np.asarray([item.high for item in self.specs], dtype=np.float32)
