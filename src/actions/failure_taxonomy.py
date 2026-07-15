"""将后验硬约束 / FMU 失败细分为可审计类型。"""

from __future__ import annotations
import re
from typing import Any, Mapping
from envs.failures import FINE_FAILURE_TYPES

_KEY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("battery_soc", "battery_soc"),
    ("caes_gas_soc", "caes_gas_soc"),
    ("caes_hot_soc", "caes_hot_soc"),
    ("caes_cold_soc", "caes_cold_soc"),
    ("caes_gas_pressure", "caes_pressure"),
    ("caes_gas_temperature", "caes_temperature"),
    ("caes_hot_temperature", "caes_temperature"),
    ("caes_cold_temperature", "caes_temperature"),
    ("p_thermal", "thermal_ramp"),
    ("p_grid", "grid_capacity"),
)


def classify_failure(
    *,
    failure_type: str | None,
    reason: str | None = None,
    outputs: Mapping[str, float] | None = None,
    params: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """返回 (fine_failure_type, triggering_constraint)。"""
    reason = reason or ""
    ft = failure_type or ""
    if (
        ft == "NonFiniteOutputFailure"
        or "非有限" in reason
        or "nan" in reason.lower()
        or "inf" in reason.lower()
    ):
        return "nonfinite_output", "nonfinite_output"
    if (
        ft == "FmuNumericalFailure"
        or "solver" in reason.lower()
        or "nonlinear" in reason.lower()
    ):
        return "nonlinear_solver_failure", "nonlinear_solver_failure"
    if ft == "FeasibleSetEmpty":
        return "feasible_set_empty", "feasible_set_empty"
    if "grid" in reason.lower() or "p_grid" in reason or "联络线" in reason:
        return "grid_capacity_violation", "grid_capacity"
    if "ramp" in reason.lower() or "爬坡" in reason:
        return "thermal_ramp_violation", "thermal_ramp"
    # 从 reason 解析变量名与越界方向
    key_match = re.search(
        r"(battery_soc|caes_gas_soc|caes_hot_soc|caes_cold_soc|caes_gas_pressure|"
        r"caes_gas_temperature|caes_hot_temperature|caes_cold_temperature|p_thermal|p_grid)\s*=\s*([-eE0-9.+]+)",
        reason,
    )
    if key_match:
        var = key_match.group(1)
        val = float(key_match.group(2))
        return _direction_for(var, val, reason, params)
    if outputs and params:
        typed = classify_from_outputs(outputs, params)
        if typed != "unknown":
            return typed, typed
    return "unknown", "unknown"


def classify_from_outputs(
    outputs: Mapping[str, float], params: Mapping[str, Any]
) -> str:
    b = params.get("battery", {})
    c = params.get("caes", {})
    g = params.get("grid", {})
    checks: list[tuple[str, float, float, float]] = [
        (
            "battery_soc",
            float(outputs.get("battery_soc", 0.5)),
            float(b.get("SOC_min", 0.1)),
            float(b.get("SOC_max", 0.9)),
        ),
        (
            "caes_gas_soc",
            float(outputs.get("caes_gas_soc", 0.8)),
            float(c.get("gas_SOC_min", 0.6)),
            float(c.get("gas_SOC_max", 1.0)),
        ),
        (
            "caes_hot_soc",
            float(outputs.get("caes_hot_soc", 0.5)),
            float(c.get("hot_SOC_min", 0.05)),
            float(c.get("hot_SOC_max", 0.95)),
        ),
        (
            "caes_cold_soc",
            float(outputs.get("caes_cold_soc", 0.5)),
            float(c.get("cold_SOC_min", 0.05)),
            float(c.get("cold_SOC_max", 0.95)),
        ),
        (
            "caes_gas_pressure",
            float(outputs.get("caes_gas_pressure", 8e6)),
            float(c.get("gas_pressure_min_Pa", 6.5e6)),
            float(c.get("gas_pressure_max_Pa", 9.5e6)),
        ),
    ]
    for name, val, lo, hi in checks:
        if val > hi + 1e-9:
            return _hi_name(name)
        if val < lo - 1e-9:
            return _lo_name(name)
    for tname in (
        "caes_gas_temperature",
        "caes_hot_temperature",
        "caes_cold_temperature",
    ):
        val = float(outputs.get(tname, 300.0))
        if tname == "caes_gas_temperature":
            lo, hi = float(c.get("gas_temp_min_K", 253)), float(
                c.get("gas_temp_max_K", 450)
            )
        elif tname == "caes_hot_temperature":
            lo, hi = float(c.get("hot_temp_min_K", 280)), float(
                c.get("hot_temp_max_K", 550)
            )
        else:
            lo, hi = float(c.get("cold_temp_min_K", 250)), float(
                c.get("cold_temp_max_K", 320)
            )
        if val > hi + 1e-9:
            return "caes_temperature_high"
        if val < lo - 1e-9:
            return "caes_temperature_low"
    p_grid = float(outputs.get("p_grid", 0.0))
    if p_grid > float(g.get("P_max_buy_W", 5e8)) + 1.0:
        return "grid_capacity_violation"
    if p_grid < float(g.get("P_max_sell_W", -5e8)) - 1.0:
        return "grid_capacity_violation"
    for name, val in outputs.items():
        if not _isfinite(val):
            return "nonfinite_output"
    return "unknown"


def _direction_for(
    var: str, val: float, reason: str, params: Mapping[str, Any] | None
) -> tuple[str, str]:
    # reason 中含「越 … 界」时用括号区间判断方向
    bracket = re.search(r"\[([^,\]]+),\s*([^\]]+)\]", reason)
    if bracket:
        lo, hi = float(bracket.group(1)), float(bracket.group(2))
        if val > hi:
            return _hi_name(var), var
        if val < lo:
            return _lo_name(var), var
    if params:
        typed = classify_from_outputs({var: val}, params)
        if typed != "unknown":
            return typed, var
    # fallback：与中点比较
    if "high" in reason.lower() or "上限" in reason or "max" in reason.lower():
        return _hi_name(var), var
    if "low" in reason.lower() or "下限" in reason or "min" in reason.lower():
        return _lo_name(var), var
    return _hi_name(var) if "高" in reason or val > 0.5 else _lo_name(var), var


def _hi_name(var: str) -> str:
    mapping = {
        "battery_soc": "battery_soc_high",
        "caes_gas_soc": "caes_gas_soc_high",
        "caes_hot_soc": "caes_hot_soc_high",
        "caes_cold_soc": "caes_cold_soc_high",
        "caes_gas_pressure": "caes_pressure_high",
        "caes_gas_temperature": "caes_temperature_high",
        "caes_hot_temperature": "caes_temperature_high",
        "caes_cold_temperature": "caes_temperature_high",
        "caes_pressure": "caes_pressure_high",
        "caes_temperature": "caes_temperature_high",
    }
    return mapping.get(var, "unknown")


def _lo_name(var: str) -> str:
    mapping = {
        "battery_soc": "battery_soc_low",
        "caes_gas_soc": "caes_gas_soc_low",
        "caes_hot_soc": "caes_hot_soc_low",
        "caes_cold_soc": "caes_cold_soc_low",
        "caes_gas_pressure": "caes_pressure_low",
        "caes_gas_temperature": "caes_temperature_low",
        "caes_hot_temperature": "caes_temperature_low",
        "caes_cold_temperature": "caes_temperature_low",
        "caes_pressure": "caes_pressure_low",
        "caes_temperature": "caes_temperature_low",
    }
    return mapping.get(var, "unknown")


def _isfinite(val: Any) -> bool:
    try:
        import math

        return math.isfinite(float(val))
    except (TypeError, ValueError):
        return False


def assert_known_fine_type(name: str) -> str:
    if name not in FINE_FAILURE_TYPES:
        return "unknown"
    return name
