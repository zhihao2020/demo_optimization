"""安全给予(GiveSafe) 包：两级安全检查、约束奖励与影子仿真校验。"""

from .safety_result import GiveSafeResult, SafetyCheckResult
from .no_safe_action import NoSafeActionFoundError
from .constraint_reward import ConstraintRewardCalculator
from .constraint_checker import GiveSafeConstraintChecker
from .shadow_fmu_validator import ShadowFmuValidator
from .givesafe_controller import GiveSafeController, load_givesafe_config
from .soft_constraint_shell import (
    SoftConstraintEnv,
    SoftConstraintShell,
    conservative_recover_action,
)

__all__ = [
    "GiveSafeResult",           # 安全给予结果(GiveSafeResult)
    "SafetyCheckResult",        # 安全检查单条结果(SafetyCheckResult)
    "NoSafeActionFoundError",   # 未找到安全动作异常(NoSafeActionFoundError)
    "ConstraintRewardCalculator",  # 约束奖励计算器(ConstraintRewardCalculator)
    "GiveSafeConstraintChecker",   # 安全给予约束检查器(GiveSafeConstraintChecker)
    "ShadowFmuValidator",       # 影子仿真校验器(ShadowFmuValidator)
    "GiveSafeController",       # 安全给予控制器(GiveSafeController)
    "load_givesafe_config",     # 加载安全给予配置(load_givesafe_config)
    "SoftConstraintShell",      # 软约束外壳
    "SoftConstraintEnv",        # 软约束环境包装
    "conservative_recover_action",
]
