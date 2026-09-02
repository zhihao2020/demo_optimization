"""Python ↔ FMU 物理 I/O 会话层。

FMU 负责物理状态与真实物理量；本包负责输入限幅校验与输出异常检测。
不做 reward / 市场结算。运行测试：``pytest tests/``。
"""

from .adapter import FmuAdapter
from .exceptions import FmuSolverError
from .model_info import ModelInfo, read_model_info
from .session import (
    ACTION_NAMES,
    DEFAULT_INITIAL_INPUTS,
    DEFAULT_OUTPUTS,
    DIAGNOSTIC_OUTPUTS,
    FmuSession,
    describe_fmu,
    fmu_platform_supported,
    require_communication_step,
)
from .types import DispatchPlan, SimulationResult
from .validate import validate_inputs, validate_outputs
from .variable_registry import VariableRegistry, build_registry

__all__ = [
    "ACTION_NAMES",  # 调度输入名元组
    "DEFAULT_INITIAL_INPUTS",  # 与 Modelica start 一致的默认初值
    "DEFAULT_OUTPUTS",  # 默认读取的 FMU 输出名
    "DIAGNOSTIC_OUTPUTS",  # 诊断 FMU 只读质量流/SOC，0831 没有这些口
    "DispatchPlan",  # 按时序排列的调度计划
    "FmuAdapter",  # Gymnasium 环境用 FMU 生命周期适配器
    "FmuSession",  # FMI 3.0 固定步长 Co-Simulation 会话
    "FmuSolverError",  # FMU 生命周期/步进/读输出失败
    "ModelInfo",  # modelDescription 摘要
    "SimulationResult",  # rollout 轨迹与 metadata
    "VariableRegistry",  # FMU 变量名/单位/边界注册表
    "build_registry",  # 从 YAML + modelDescription 构建注册表
    "describe_fmu",  # sha256 / guid / modelDescription hash
    "fmu_platform_supported",  # 当前 OS 是否有对应 FMU 二进制
    "read_model_info",  # 只读解析 modelDescription
    "require_communication_step",  # 通信步须整除 fixedInternalStepSize
    "validate_inputs",  # 调度输入边界校验
    "validate_outputs",  # 物理输出合理性校验
]
