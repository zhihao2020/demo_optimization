"""FMU 输入/输出校验。

边界与物理含义见 docs/fmu_input_bounds.md。
Modelica 侧不对调度指令做饱和；越界或数值异常由本模块直接报错。
本模块不做 market / reward 计算。
"""

from __future__ import annotations

import math
from typing import Mapping

# --- 输入边界（PowerSystem_8760h 实例：P_min/P_cap、额定归一化、CAES 最小启机）---
# 火电：最小稳燃出力 / 装机 = 50MW / 150MW = 1/3；不允许停机到 0。
U_TP_MIN = 1.0 / 3.0
U_TP_MAX = 1.0
# 电池：功率指令相对 P_cap 归一化，正充电、负放电。
U_BATTERY_MIN = -1.0
U_BATTERY_MAX = 1.0
# CAES：额定归一化 + 最小启机比例；中间开区间禁止。
# 放电（膨胀）[-1, -0.33]，待机 0，充电（压缩）[0.86, 1]。
U_CAES_DISCHARGE_MAX = -0.33  # 负向中最接近 0 的允许值
U_CAES_CHARGE_MIN = 0.86
U_CAES_ABS_MAX = 1.0
U_CAES_ZERO_EPS = 1e-9

# 输出容差：浮点噪声；溢出哨兵防止 Inf 漏检后的极端值。
_EPS = 1e-9
_POWER_ABS_MAX = 1e12  # W，远超系统装机，仅作异常哨兵

# 需要落入 [0, 1] 的 SOC 输出
_SOC_NAMES = (
    "battery_soc",
    "caes_gas_soc",
    "caes_hot_soc",
    "caes_cold_soc",
)
# 应非负的功率不平衡输出
_NONNEG_POWER_NAMES = ("p_curtailment", "p_unserved")
# 功率类输出（有限 + 溢出哨兵）
_POWER_NAMES = (
    "p_curtailment",
    "p_unserved",
    "p_thermal",
    "p_battery",
    "p_caes",
    "p_grid",
    "p_wind_available",
    "p_wind_actual",
    "p_pv_available",
    "p_pv_actual",
    "p_load_actual",
)
_TEMP_NAMES = (
    "caes_gas_temperature",
    "caes_hot_temperature",
    "caes_cold_temperature",
)


def validate_inputs(action: Mapping[str, float]) -> None:
    """校验调度输入；越界或缺 key / 非有限则抛错。

    规则来源：``docs/fmu_input_bounds.md``（与 PowerSystem_8760h 设备参数一致）。

    Args:
        action: 含 ``u_tp``、``u_battery``、``u_caes`` 的映射。

    Raises:
        ValueError: 缺键、非有限或超出允许区间/集合。
    """
    for name in ("u_tp", "u_battery", "u_caes"):
        if name not in action:
            raise ValueError(f"缺少调度输入 '{name}'")
        value = float(action[name])
        if not math.isfinite(value):
            raise ValueError(f"{name}={value} 不是有限浮点数")

    u_tp = float(action["u_tp"])
    # 火电负荷率：对应 P_min/P_cap ～ P_max/P_cap
    if not (U_TP_MIN - _EPS <= u_tp <= U_TP_MAX + _EPS):
        raise ValueError(
            f"u_tp={u_tp} 超出允许区间 [{U_TP_MIN}, {U_TP_MAX}] "
            f"（火电最小稳燃负荷率 P_min/P_cap）"
        )

    u_battery = float(action["u_battery"])
    if not (U_BATTERY_MIN - _EPS <= u_battery <= U_BATTERY_MAX + _EPS):
        raise ValueError(
            f"u_battery={u_battery} 超出允许区间 [{U_BATTERY_MIN}, {U_BATTERY_MAX}] "
            f"（相对电池 P_cap 归一化，正充负放）"
        )

    u_caes = float(action["u_caes"])
    if abs(u_caes) <= U_CAES_ZERO_EPS:
        return  # 待机
    in_discharge = -U_CAES_ABS_MAX - _EPS <= u_caes <= U_CAES_DISCHARGE_MAX + _EPS
    in_charge = U_CAES_CHARGE_MIN - _EPS <= u_caes <= U_CAES_ABS_MAX + _EPS
    if not (in_discharge or in_charge):
        raise ValueError(
            f"u_caes={u_caes} 不在允许集合 "
            f"[-{U_CAES_ABS_MAX}, {U_CAES_DISCHARGE_MAX}] ∪ {{0}} ∪ "
            f"[{U_CAES_CHARGE_MIN}, {U_CAES_ABS_MAX}] "
            f"（CAES 最小启机比例；中间开区间禁止）"
        )


def validate_outputs(outputs: Mapping[str, float]) -> None:
    """校验 FMU 物理输出；NaN/Inf 或物理不合理则抛错。

    用于尽早发现求解失败或接口错接，避免把坏数送进后续 Python 逻辑。

    Args:
        outputs: FMU 顶层输出名 -> 标量值。

    Raises:
        ValueError: 非有限、SOC 越界、非正压力/温度或非负功率违反等。
    """
    for name, raw in outputs.items():
        value = float(raw)
        # 任何输出出现 NaN/Inf 都视为仿真异常
        if not math.isfinite(value):
            raise ValueError(f"输出 '{name}'={value} 非有限（NaN/Inf）")

    for name in _NONNEG_POWER_NAMES:
        if name not in outputs:
            continue
        value = float(outputs[name])
        # 弃电/缺供拆分后应为非负
        if value < -_EPS:
            raise ValueError(f"输出 '{name}'={value} 应为非负功率 (W)")

    for name in _SOC_NAMES:
        if name not in outputs:
            continue
        value = float(outputs[name])
        if not (0.0 - _EPS <= value <= 1.0 + _EPS):
            raise ValueError(f"输出 '{name}'={value} 应落在 SOC 区间 [0, 1]")

    if "caes_gas_pressure" in outputs:
        pressure = float(outputs["caes_gas_pressure"])
        # 绝对压力，物理上须为正
        if pressure <= 0.0:
            raise ValueError(
                f"输出 'caes_gas_pressure'={pressure} 应 > 0 (Pa)"
            )

    for name in _TEMP_NAMES:
        if name not in outputs:
            continue
        temp = float(outputs[name])
        # 热力学温度 K
        if temp <= 0.0:
            raise ValueError(f"输出 '{name}'={temp} 应 > 0 (K)")

    for name in _POWER_NAMES:
        if name not in outputs:
            continue
        power = float(outputs[name])
        if abs(power) >= _POWER_ABS_MAX:
            raise ValueError(
                f"输出 '{name}'={power} 超出合理功率哨兵 {_POWER_ABS_MAX} W"
            )
