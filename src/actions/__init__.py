"""显式混合动作定义：离散 CAES 模式 + 连续幅值。"""
from .types import CaesMode, HybridAction, PhysicalFmuAction
from .decoder import HybridActionDecoder
from .validator import HybridActionValidator
from .mode_mask import ModeMask
from .feasible_set import DynamicFeasibleActionSet
from .feasibility_oracle import FeasibilityOracle, PREDICTED_STATE_KEYS
from .failure_taxonomy import classify_failure
from .safety_classifier import SafetyClassifier, FeasibilityCalibrator, SafetyMetrics
from .action_pipeline import SafeActionGenerator
from .boundary_stress import BoundaryStressTester, BoundaryStressResult
__all__ = [
    "CaesMode",
    "HybridAction",
    "PhysicalFmuAction",
    "HybridActionDecoder",
    "HybridActionValidator",
    "ModeMask",
    "DynamicFeasibleActionSet",
    "FeasibilityOracle",
    "PREDICTED_STATE_KEYS",
    "classify_failure",
    "SafetyClassifier",
    "FeasibilityCalibrator",
    "SafetyMetrics",
    "SafeActionGenerator",
    "BoundaryStressTester",
    "BoundaryStressResult",
]
