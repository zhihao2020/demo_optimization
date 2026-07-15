"""更新既有动作校验测试：保留物理静态校验语义。"""

import numpy as np
import pytest

from actions import HybridAction, CaesMode, HybridActionDecoder, HybridActionValidator
from actions.types import PhysicalFmuAction
from envs.failures import StaticActionViolation
from envs.action_validator import ActionValidator, ActionConstraintError as LegacyConstraint


def test_caes_forbidden_static_physical():
    v = HybridActionValidator()
    with pytest.raises(StaticActionViolation):
        v.validate_physical_static(PhysicalFmuAction(1.0, 0.0, 0.5))


def test_hybrid_decode_boundaries():
    dec = HybridActionDecoder()
    assert abs(dec.decode(HybridAction(1.0, 0.0, CaesMode.DISCHARGE, 0.0)).u_caes - (-1.0)) < 1e-9
    assert abs(dec.decode(HybridAction(1.0, 0.0, CaesMode.DISCHARGE, 1.0)).u_caes - (-0.33)) < 1e-9
    assert abs(dec.decode(HybridAction(1.0, 0.0, CaesMode.IDLE, 0.7)).u_caes) < 1e-9
    assert abs(dec.decode(HybridAction(1.0, 0.0, CaesMode.CHARGE, 0.0)).u_caes - 0.86) < 1e-9
    assert abs(dec.decode(HybridAction(1.0, 0.0, CaesMode.CHARGE, 1.0)).u_caes - 1.0) < 1e-9


def test_legacy_box_validator_still_rejects_caes_band():
    validator = ActionValidator(
        ("u_tp", "u_battery", "u_caes"),
        ("1", "1", "1"),
        np.array([1 / 3, -1.0, -1.0]),
        np.array([1.0, 1.0, 1.0]),
        {"u_caes": ((-1.0, -0.33), (0.0, 0.0), (0.86, 1.0))},
    )
    with pytest.raises(LegacyConstraint):
        validator.validate(np.array([1.0, 0.0, 0.5], dtype=np.float64))
