"""集中保存 FMU 名称、单位和可审计边界来源，环境不硬编码 FMU 名。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fmpy import read_model_description


@dataclass(frozen=True)
class VariableSpec:
    name: str
    fmu_name: str
    unit: str
    causality: str
    low: float | None = None
    high: float | None = None
    bound_source: str = "FMU metadata unavailable"
    description: str = ""


@dataclass(frozen=True)
class VariableRegistry:
    actions: tuple[VariableSpec, ...]
    outputs: dict[str, VariableSpec]

    @property
    def action_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.actions)

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(self.outputs)


def build_registry(fmu_path: Path, env_config: dict[str, Any]) -> VariableRegistry:
    """以 modelDescription 为名称/单位主来源、以显式 YAML 为缺失边界来源。"""
    md = read_model_description(str(fmu_path))
    variables = {item.name: item for item in md.modelVariables}
    actions: list[VariableSpec] = []
    for item in env_config["actions"]:
        fmu_name = item.get("fmu_variable", item["name"])
        model_variable = variables.get(fmu_name)
        if model_variable is None:
            raise KeyError(f"FMU 中不存在动作变量 {fmu_name!r}")
        if model_variable.causality != "input":
            raise ValueError(f"{fmu_name!r} 不是 FMU input")
        if "min" not in item or "max" not in item:
            raise ValueError(f"动作 {item['name']} 缺少显式边界，禁止使用默认值")
        actions.append(
            VariableSpec(
                name=item["name"], fmu_name=fmu_name,
                unit=item.get("unit") or getattr(model_variable, "unit", None) or "1",
                causality="input", low=float(item["min"]), high=float(item["max"]),
                bound_source=item.get("bound_source", "env_config.yaml"),
                description=getattr(model_variable, "description", "") or "",
            )
        )

    outputs: dict[str, VariableSpec] = {}
    for item in env_config["observations"]:
        fmu_name = item.get("fmu_variable", item["name"])
        model_variable = variables.get(fmu_name)
        if model_variable is None:
            raise KeyError(f"FMU 中不存在观测变量 {fmu_name!r}")
        if model_variable.causality != "output":
            raise ValueError(f"{fmu_name!r} 不是 FMU output")
        outputs[item["name"]] = VariableSpec(
            name=item["name"], fmu_name=fmu_name,
            unit=item.get("unit") or getattr(model_variable, "unit", None) or "1",
            causality="output", low=item.get("low"), high=item.get("high"),
            description=getattr(model_variable, "description", "") or "",
        )
    return VariableRegistry(tuple(actions), outputs)
