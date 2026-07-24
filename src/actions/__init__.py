"""混合动作包：离散 CAES 模式 + 连续幅值，及可行性/安全相关组件的公共导出。"""

from .types import CaesMode, HybridAction, PhysicalFmuAction
from .decoder import HybridActionDecoder
from .validator import HybridActionValidator
from .mode_mask import ModeMask
from .feasible_set import DynamicFeasibleActionSet
from .feasibility_oracle import FeasibilityOracle, PREDICTED_STATE_KEYS
from .failure_taxonomy import classify_failure
from .safety_classifier import SafetyClassifier, FeasibilityCalibrator, SafetyMetrics
from .action_pipeline import SafeActionGenerator
from .caes_min_run import CaesMinimumRunController, MIN_CAES_RUN_STEPS

__all__ = [
    "CaesMode",  # CAES 模式(CaesMode)
    "HybridAction",  # 混合动作(HybridAction)
    "PhysicalFmuAction",  # 物理 FMU 动作(PhysicalFmuAction)
    "HybridActionDecoder",  # 混合动作解码器(HybridActionDecoder)
    "HybridActionValidator",  # 混合动作验证器(HybridActionValidator)
    "ModeMask",  # 模式掩码(ModeMask)
    "DynamicFeasibleActionSet",  # 动态可行动作集(DynamicFeasibleActionSet)
    "FeasibilityOracle",  # 可行性神谕(FeasibilityOracle)
    "PREDICTED_STATE_KEYS",  # 预测状态键(PREDICTED_STATE_KEYS)
    "classify_failure",  # 失败分类(classify_failure)
    "SafetyClassifier",  # 安全性分类器(SafetyClassifier)
    "FeasibilityCalibrator",  # 可行性校准器(FeasibilityCalibrator)
    "SafetyMetrics",  # 安全指标(SafetyMetrics)
    "SafeActionGenerator",  # 安全动作生成器(SafeActionGenerator)
    "CaesMinimumRunController",  # CAES 最短运行控制器(CaesMinimumRunController)
    "MIN_CAES_RUN_STEPS",  # CAES 最短运行步数(MIN_CAES_RUN_STEPS)
]

