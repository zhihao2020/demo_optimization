"""从 FMU 的 modelDescription 读取输入/输出清单（只读，不仿真）。

用于脚本检查与 registry 构建前的元数据审计，不启动 solver。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fmpy


@dataclass
class ModelVariable:
    """单个 modelDescription 变量的精简视图。"""

    name: str
    causality: str
    description: str
    start: float | None = None
    unit: str | None = None


@dataclass
class ModelInfo:
    """FMU 静态元数据：名称、FMI 版本、默认实验、变量列表。"""

    path: Path
    model_name: str
    fmi_version: str
    step_size: float | None
    stop_time: float | None
    variables: list[ModelVariable]

    @property
    def inputs(self) -> list[ModelVariable]:
        return [v for v in self.variables if v.causality == "input"]

    @property
    def outputs(self) -> list[ModelVariable]:
        return [v for v in self.variables if v.causality == "output"]

    def summary_dict(self) -> dict[str, Any]:
        """便于 JSON 打印的摘要（不含全部变量详情）。"""
        return {
            "path": str(self.path),
            "model_name": self.model_name,
            "fmi_version": self.fmi_version,
            "step_size": self.step_size,
            "stop_time": self.stop_time,
            "n_variables": len(self.variables),
            "inputs": [v.name for v in self.inputs],
            "outputs": [v.name for v in self.outputs],
        }


def read_model_info(fmu_path: Path) -> ModelInfo:
    """解析 FMU ZIP 内 modelDescription，不 instantiate。"""
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
