"""Python ↔ FMU 物理 I/O 会话层。

FMU 负责物理状态与真实物理量；本包负责输入限幅校验与输出异常检测。
不做 reward / 市场结算。运行测试：`pytest tests/`。
"""

from .adapter import FmuAdapter
from .exceptions import FmuSolverError
from .model_info import ModelInfo, read_model_info
from .session import (
    ACTION_NAMES,
    DEFAULT_INITIAL_INPUTS,
    DEFAULT_OUTPUTS,
    FmuSession,
    fmu_platform_supported,
)
from .types import DispatchPlan, SimulationResult
from .validate import validate_inputs, validate_outputs
from .variable_registry import VariableRegistry, build_registry

__all__ = [
    "ACTION_NAMES",
    "DEFAULT_INITIAL_INPUTS",
    "DEFAULT_OUTPUTS",
    "DispatchPlan",
    "FmuAdapter",
    "FmuSession",
    "FmuSolverError",
    "ModelInfo",
    "SimulationResult",
    "VariableRegistry",
    "build_registry",
    "fmu_platform_supported",
    "read_model_info",
    "validate_inputs",
    "validate_outputs",
]
