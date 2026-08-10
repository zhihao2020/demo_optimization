"""物理动作包：三连续 FMU 指令 + 可行性/安全组件。"""

from .types import CaesMode, PhysicalFmuAction
from .caes_u import (
    CHARGE_HI,
    CHARGE_LO,
    DISCHARGE_HI,
    DISCHARGE_LO,
    apply_mode_mask_to_u,
    apply_mode_mask_to_u_torch,
    mode_from_u,
    physical_dict,
    project_u_caes,
    project_u_caes_torch,
    u_from_mode_mag,
)
from .validator import PhysicalActionValidator, physical_from_dict
from .mode_mask import ModeMask
from .feasible_set import DynamicFeasibleActionSet
from .feasibility_oracle import FeasibilityOracle, PREDICTED_STATE_KEYS
from .failure_taxonomy import classify_failure
from .safety_classifier import SafetyClassifier, FeasibilityCalibrator, SafetyMetrics
from .action_pipeline import SafeActionGenerator
from .caes_min_run import CaesMinimumRunController, MIN_CAES_RUN_STEPS

__all__ = [
    "CaesMode",
    "PhysicalFmuAction",
    "PhysicalActionValidator",
    "physical_from_dict",
    "ModeMask",
    "DynamicFeasibleActionSet",
    "FeasibilityOracle",
    "PREDICTED_STATE_KEYS",
    "classify_failure",
    "SafetyClassifier",
    "FeasibilityCalibrator",
    "SafetyMetrics",
    "SafeActionGenerator",
    "CaesMinimumRunController",
    "MIN_CAES_RUN_STEPS",
    "CHARGE_HI",
    "CHARGE_LO",
    "DISCHARGE_HI",
    "DISCHARGE_LO",
    "apply_mode_mask_to_u",
    "apply_mode_mask_to_u_torch",
    "mode_from_u",
    "physical_dict",
    "project_u_caes",
    "project_u_caes_torch",
    "u_from_mode_mag",
]
