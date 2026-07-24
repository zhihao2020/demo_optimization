"""从 FMU 的 modelDescription 读取输入/输出清单（只读，不仿真）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fmpy


@dataclass
class ModelVariable:
    """模型变量(ModelVariable)：modelDescription 中的单个变量元数据。

    Attributes:
        name: FMU 变量名。
        causality: 因果性（``input`` / ``output`` 等）。
        description: 变量描述文本。
        start: 初始值（若有）。
        unit: 物理单位（若有）。
    """

    name: str
    causality: str
    description: str
    start: float | None = None
    unit: str | None = None


@dataclass
class ModelInfo:
    """模型信息(ModelInfo)：FMU modelDescription 的结构化摘要。

    Attributes:
        path: FMU 文件路径。
        model_name: Modelica 模型名。
        fmi_version: FMI 版本字符串。
        step_size: 默认实验步长（若有）。
        stop_time: 默认实验停止时间（若有）。
        variables: 全部变量列表。
    """

    path: Path
    model_name: str
    fmi_version: str
    step_size: float | None
    stop_time: float | None
    variables: list[ModelVariable]

    @property
    def outputs(self) -> list[ModelVariable]:
        """输出变量列表。

        Returns:
            因果性为 ``output`` 的变量。
        """
        return [v for v in self.variables if v.causality == "output"]


def read_model_info(fmu_path: Path) -> ModelInfo:
    """只读解析 FMU 的 modelDescription。

    Args:
        fmu_path: ``.fmu`` 文件路径。

    Returns:
        模型信息(ModelInfo) 实例。
    """
    md = fmpy.read_model_description(str(fmu_path))

    step_size = None
    stop_time = None
    if md.defaultExperiment is not None:
        step_size = md.defaultExperiment.stepSize
        stop_time = md.defaultExperiment.stopTime

    variables = [
        ModelVariable(
            name=v.name,
            causality=v.causality or "unknown",
            description=v.description or "",
            start=getattr(v, "start", None),
            unit=getattr(v, "unit", None),
        )
        for v in md.modelVariables
    ]

    return ModelInfo(
        path=fmu_path,
        model_name=md.modelName,
        fmi_version=md.fmiVersion,
        step_size=step_size,
        stop_time=stop_time,
        variables=variables,
    )
