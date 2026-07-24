"""集中保存 FMU 名称、单位和可审计边界来源，环境不硬编码 FMU 名。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fmpy import read_model_description


@dataclass(frozen=True)
class VariableSpec:
    """变量规格(VariableSpec)：Python 名与 FMU 名的映射及边界元数据。

    Attributes:
        name: Python/环境侧变量名。
        fmu_name: FMU modelDescription 中的变量名。
        unit: 物理单位。
        causality: ``input`` 或 ``output``。
        low: 下界（输出可为 ``None`` 表示无界）。
        high: 上界（输出可为 ``None`` 表示无界）。
        bound_source: 边界来源说明（审计用）。
        description: FMU 变量描述。
    """

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
    """变量注册表(VariableRegistry)：动作、观测与经济输出的统一索引。

    Attributes:
        actions: 动作变量规格元组（有序）。
        outputs: 物理观测名 -> VariableSpec。
        economics: 经济审计输出名 -> VariableSpec。
    """

    actions: tuple[VariableSpec, ...]
    outputs: dict[str, VariableSpec]
    economics: dict[str, VariableSpec] = field(default_factory=dict)

    @property
    def action_names(self) -> tuple[str, ...]:
        """Python 侧动作名元组。

        Returns:
            与 ``actions`` 顺序一致的动作名。
        """
        return tuple(item.name for item in self.actions)

    @property
    def read_outputs(self) -> dict[str, VariableSpec]:
        """FMU 每步读取的物理观测与经济审计输出。

        Returns:
            ``outputs`` 与 ``economics`` 的合并字典。
        """
        return {**self.outputs, **self.economics}


def build_registry(
    fmu_path: Path,
    env_config: dict[str, Any],
    *,
    verify_metadata: bool = True,
) -> VariableRegistry:
    """以 modelDescription 为名称/单位主来源、以显式 YAML 为缺失边界来源。

    注入测试 adapter 时可关闭 metadata 验证，使单元测试不依赖待重新导出的
    二进制；真实 FMU 路径始终保持严格验证。

    Args:
        fmu_path: ``.fmu`` 文件路径。
        env_config: 环境 YAML 中的 ``actions``、``observations``、``economics``。
        verify_metadata: 是否对照 modelDescription 校验变量存在与因果性。

    Returns:
        构建完成的 VariableRegistry。

    Raises:
        KeyError: FMU 中缺少配置引用的变量。
        ValueError: 因果性错误或动作缺少显式边界。
    """
    variables = {}
    if verify_metadata:
        md = read_model_description(str(fmu_path))
        variables = {item.name: item for item in md.modelVariables}
    actions: list[VariableSpec] = []
    for item in env_config["actions"]:
        fmu_name = item.get("fmu_variable", item["name"])
        model_variable = variables.get(fmu_name)
        if model_variable is None and verify_metadata:
            raise KeyError(f"FMU 中不存在动作变量 {fmu_name!r}")
        if model_variable is not None and model_variable.causality != "input":
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

    def parse_outputs(items: list[dict[str, Any]], kind: str) -> dict[str, VariableSpec]:
        """解析配置中的输出变量列表为变量规格字典。

        Args:
            items: 配置项列表。
            kind: 类别描述（如「观测」「经济」），用于报错文案。

        Returns:
            逻辑名到变量规格(VariableSpec)的映射。
        """
        parsed: dict[str, VariableSpec] = {}
        for item in items:
            fmu_name = item.get("fmu_variable", item["name"])
            model_variable = variables.get(fmu_name)
            if model_variable is None and verify_metadata:
                raise KeyError(f"FMU 中不存在{kind}变量 {fmu_name!r}")
            if model_variable is not None and model_variable.causality != "output":
                raise ValueError(f"{fmu_name!r} 不是 FMU output")
            parsed[item["name"]] = VariableSpec(
                name=item["name"], fmu_name=fmu_name,
                unit=item.get("unit") or getattr(model_variable, "unit", None) or "1",
                causality="output", low=item.get("low"), high=item.get("high"),
                description=getattr(model_variable, "description", "") or "",
            )
        return parsed

    outputs = parse_outputs(env_config["observations"], "观测")
    economics = parse_outputs(env_config.get("economics") or [], "经济")
    return VariableRegistry(tuple(actions), outputs, economics)
