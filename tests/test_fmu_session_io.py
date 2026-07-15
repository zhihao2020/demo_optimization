"""会话层 I/O 名与文档/Modelica 对齐检查（不强制加载真实 .fmu）。

测什么：ACTION_NAMES / DEFAULT_OUTPUTS 集合是否完整。
期望：与 PowerSystem_8760h 顶层接口一致。
"""

from __future__ import annotations

from fmu.session import ACTION_NAMES, DEFAULT_INITIAL_INPUTS, DEFAULT_OUTPUTS
from fmu.validate import validate_inputs


def test_action_names_match_modelica_inputs() -> None:
    """三个调度输入名须与 Modelica RealInput 一致。"""
    assert ACTION_NAMES == ("u_tp", "u_battery", "u_caes")


def test_default_outputs_cover_physical_groups() -> None:
    """默认输出须覆盖不平衡/SOC/设备功率/风光/负荷/CAES 热力各组。"""
    names = set(DEFAULT_OUTPUTS)
    required = {
        "p_curtailment",
        "p_unserved",
        "battery_soc",
        "caes_gas_soc",
        "caes_hot_soc",
        "caes_cold_soc",
        "p_thermal",
        "p_battery",
        "p_caes",
        "p_grid",
        "p_wind_available",
        "p_wind_actual",
        "p_pv_available",
        "p_pv_actual",
        "p_load_actual",
        "caes_gas_pressure",
        "caes_gas_temperature",
        "caes_hot_temperature",
        "caes_cold_temperature",
        "economic_cashflow_total",
        "economic_cashflow_wind",
        "economic_cashflow_pv",
        "economic_cashflow_thermal",
        "economic_cashflow_battery",
        "economic_cashflow_caes",
        "economic_cashflow_load",
        "economic_cashflow_grid",
    }
    assert names == required
    # 不泄露内部变量或已删除 penalty 接口；经济输出使用稳定的顶层名称。
    assert "OPT_goal" not in names
    assert "P_res" not in names
    assert "battery_penalty" not in names


def test_default_initial_inputs_are_valid() -> None:
    """默认初值须通过输入校验（可直接用于 reset）。"""
    validate_inputs(DEFAULT_INITIAL_INPUTS)
