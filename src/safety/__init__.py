"""GiveSafe 安全包：策略候选 → Oracle/Shadow 校验 → 安全才执行主 FMU。

拒绝样本写入独立 GiveSafe replay，使用约束奖励（非经济成本伪标签）。
"""

from .safety_result import GiveSafeResult, SafetyCheckResult
from .no_safe_action import NoSafeActionFoundError
from .constraint_reward import ConstraintRewardCalculator
from .constraint_checker import GiveSafeConstraintChecker
from .shadow_fmu_validator import ShadowFmuValidator
from .givesafe_controller import GiveSafeController, load_givesafe_config

__all__ = [
    "GiveSafeResult",
    "SafetyCheckResult",
    "NoSafeActionFoundError",
    "ConstraintRewardCalculator",
    "GiveSafeConstraintChecker",
    "ShadowFmuValidator",
    "GiveSafeController",
    "load_givesafe_config",
]
