"""输入/输出校验单元测试（不依赖真实 FMU 二进制）。

测什么：validate_inputs / validate_outputs 的合法与非法边界。
期望：合法通过；越界或 NaN 等抛 ValueError。
"""

from __future__ import annotations

import math

import pytest

from fmu.validate import validate_inputs, validate_outputs


def _ok_action(**overrides: float) -> dict[str, float]:
    base = {"u_tp": 1.0, "u_battery": 0.0, "u_caes": 0.0}
    base.update(overrides)
    return base


def _ok_outputs(**overrides: float) -> dict[str, float]:
    """构造一组物理上合理的输出快照。"""
    base = {
        "p_curtailment": 0.0,
        "p_unserved": 0.0,
        "battery_soc": 0.5,
        "caes_gas_soc": 0.7,
        "caes_hot_soc": 0.5,
        "caes_cold_soc": 0.5,
        "p_thermal": -1.0e8,
        "p_battery": 0.0,
        "p_caes": 0.0,
        "p_grid": 1.0e7,
        "p_wind_available": -2.0e8,
        "p_wind_actual": -2.0e8,
        "p_pv_available": -1.0e8,
        "p_pv_actual": -1.0e8,
        "p_load_actual": 2.0e8,
        "caes_gas_pressure": 7.0e6,
        "caes_gas_temperature": 300.0,
        "caes_hot_temperature": 450.0,
        "caes_cold_temperature": 280.0,
    }
    base.update(overrides)
    return base


def test_validate_inputs_accepts_nominal() -> None:
    """额定/待机指令应通过。"""
    validate_inputs(_ok_action())
    validate_inputs(_ok_action(u_tp=1.0 / 3.0, u_battery=-1.0, u_caes=-0.33))
    validate_inputs(_ok_action(u_caes=0.86))
    validate_inputs(_ok_action(u_caes=-1.0))
    validate_inputs(_ok_action(u_caes=1.0))


def test_validate_inputs_rejects_tp_below_min() -> None:
    """火电低于最小稳燃负荷率应报错。"""
    with pytest.raises(ValueError, match="u_tp"):
        validate_inputs(_ok_action(u_tp=0.2))


def test_validate_inputs_rejects_caes_forbidden_band() -> None:
    """CAES 中间开区间（如 0.5）应报错。"""
    with pytest.raises(ValueError, match="u_caes"):
        validate_inputs(_ok_action(u_caes=0.5))


def test_validate_inputs_rejects_battery_over_rated() -> None:
    """电池超出 ±1 额定归一化应报错。"""
    with pytest.raises(ValueError, match="u_battery"):
        validate_inputs(_ok_action(u_battery=1.5))


def test_validate_inputs_rejects_missing_key() -> None:
    """缺少任一调度输入应报错。"""
    with pytest.raises(ValueError, match="缺少"):
        validate_inputs({"u_tp": 1.0, "u_battery": 0.0})


def test_validate_outputs_accepts_nominal() -> None:
    """合理物理输出应通过。"""
    validate_outputs(_ok_outputs())


def test_validate_outputs_rejects_nan() -> None:
    """任一输出为 NaN 应报错。"""
    with pytest.raises(ValueError, match="非有限"):
        validate_outputs(_ok_outputs(p_thermal=math.nan))


def test_validate_outputs_rejects_negative_curtailment() -> None:
    """弃电功率为负应报错。"""
    with pytest.raises(ValueError, match="p_curtailment"):
        validate_outputs(_ok_outputs(p_curtailment=-1.0))


def test_validate_outputs_rejects_soc_out_of_range() -> None:
    """SOC 超出 [0,1] 应报错。"""
    with pytest.raises(ValueError, match="battery_soc"):
        validate_outputs(_ok_outputs(battery_soc=1.5))
