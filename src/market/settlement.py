"""Price-taker 购售电逐步结算。

约定与 Modelica Grid 一致：p_grid > 0 购电，p_grid < 0 售电；功率单位 W。
现金流符号：正=收益，负=成本（与 economic_cashflow_* 一致）。
"""

from __future__ import annotations

from typing import Any


def grid_cashflow_cny(
    p_grid_w: float,
    buy_yuan_per_kwh: float,
    sell_yuan_per_kwh: float,
    dt_hours: float,
) -> float:
    """单步电网现金流（CNY）。

    cost = (λ_buy * max(P,0) - λ_sell * max(-P,0)) * dt
    其中 P 为 MW，λ 为 元/MWh（由 元/kWh * 1000 换算）。
    cashflow = -cost
    """
    p_mw = float(p_grid_w) * 1e-6
    buy_mwh = float(buy_yuan_per_kwh) * 1000.0
    sell_mwh = float(sell_yuan_per_kwh) * 1000.0
    dt = float(dt_hours)
    if dt < 0:
        raise ValueError("dt_hours 不能为负")
    cost = (buy_mwh * max(p_mw, 0.0) - sell_mwh * max(-p_mw, 0.0)) * dt
    return -float(cost)


def settle_grid_step(
    p_grid_w: float,
    buy_yuan_per_kwh: float,
    sell_yuan_per_kwh: float,
    dt_hours: float,
) -> dict[str, float]:
    """返回现金流、成本分项与电量，便于日志与后评估。"""
    p_mw = float(p_grid_w) * 1e-6
    dt = float(dt_hours)
    buy_mwh = float(buy_yuan_per_kwh) * 1000.0
    sell_mwh = float(sell_yuan_per_kwh) * 1000.0
    energy_buy_mwh = max(p_mw, 0.0) * dt
    energy_sell_mwh = max(-p_mw, 0.0) * dt
    buy_cost = buy_mwh * energy_buy_mwh
    sell_revenue = sell_mwh * energy_sell_mwh
    cashflow = -buy_cost + sell_revenue
    return {
        "market_grid_cashflow": float(cashflow),
        "market_grid_cost": float(buy_cost - sell_revenue),
        "market_buy_cost": float(buy_cost),
        "market_sell_revenue": float(sell_revenue),
        "market_energy_buy_mwh": float(energy_buy_mwh),
        "market_energy_sell_mwh": float(energy_sell_mwh),
        "market_buy_yuan_per_kwh": float(buy_yuan_per_kwh),
        "market_sell_yuan_per_kwh": float(sell_yuan_per_kwh),
        "p_grid_mw": float(p_mw),
    }


def peak_valley_arbitrage_cny(
    energy_charge_valley_mwh: float,
    energy_discharge_peak_mwh: float,
    valley_buy_yuan_per_kwh: float,
    peak_sell_or_avoid_yuan_per_kwh: float,
) -> float:
    """峰谷套利粗算：谷充成本 + 峰放（售电或避免购电）收益。"""
    charge_cost = energy_charge_valley_mwh * valley_buy_yuan_per_kwh * 1000.0
    discharge_value = energy_discharge_peak_mwh * peak_sell_or_avoid_yuan_per_kwh * 1000.0
    return float(discharge_value - charge_cost)


def composition_terms(step_logs: list[dict[str, Any]]) -> dict[str, float]:
    """从逐步 settlement 日志汇总年/周购售与费用构成。"""
    buy_cost = sell_rev = e_buy = e_sell = 0.0
    for row in step_logs:
        buy_cost += float(row.get("market_buy_cost", 0.0))
        sell_rev += float(row.get("market_sell_revenue", 0.0))
        e_buy += float(row.get("market_energy_buy_mwh", 0.0))
        e_sell += float(row.get("market_energy_sell_mwh", 0.0))
    return {
        "total_buy_cost_cny": buy_cost,
        "total_sell_revenue_cny": sell_rev,
        "net_grid_cost_cny": buy_cost - sell_rev,
        "energy_buy_mwh": e_buy,
        "energy_sell_mwh": e_sell,
    }
