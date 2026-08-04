"""全投资现金流、NPV、IRR、LCOE、静态回收期。

口径：
- 期初一次性投资（含税全额）为负现金流；
- 运营期年净现金流 = 年节省额或年净运营收益 - 年 O&M（简化）；
- LCOE = 折现成本现值 / 折现供电量；
- 不做价格年上涨；不变价。
"""

from __future__ import annotations

from typing import Sequence


def project_cashflows(
    capex_cny: float,
    annual_net_benefit_cny: Sequence[float],
) -> list[float]:
    """返回 [CF0, CF1, ...]，CF0 = -capex。"""
    if capex_cny < 0:
        raise ValueError("capex_cny 应为非负投资额")
    return [-float(capex_cny)] + [float(x) for x in annual_net_benefit_cny]


def npv(cashflows: Sequence[float], discount_rate: float) -> float:
    """净现值。discount_rate 为小数，如 0.08。"""
    r = float(discount_rate)
    if r <= -1.0:
        raise ValueError("discount_rate 非法")
    total = 0.0
    for t, cf in enumerate(cashflows):
        total += float(cf) / ((1.0 + r) ** t)
    return float(total)


def irr(cashflows: Sequence[float], *, lo: float = -0.99, hi: float = 10.0, tol: float = 1e-7, max_iter: int = 200) -> float:
    """内部收益率（二分）；无解时返回 nan。"""
    cfs = [float(x) for x in cashflows]
    if not cfs or all(c >= 0 for c in cfs) or all(c <= 0 for c in cfs):
        return float("nan")

    def f(rate: float) -> float:
        return npv(cfs, rate)

    flo, fhi = f(lo), f(hi)
    # 扩展上界
    expand = 0
    while flo * fhi > 0 and expand < 20:
        hi *= 1.5
        fhi = f(hi)
        expand += 1
    if flo * fhi > 0:
        return float("nan")
    a, b = lo, hi
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fm = f(mid)
        if abs(fm) < tol or abs(b - a) < tol:
            return float(mid)
        if flo * fm <= 0:
            b, fhi = mid, fm
        else:
            a, flo = mid, fm
    return float(0.5 * (a + b))


def simple_payback_years(cashflows: Sequence[float]) -> float:
    """静态回收期（累计现金流转正）；不回收返回 +inf。"""
    acc = 0.0
    for t, cf in enumerate(cashflows):
        acc += float(cf)
        if t > 0 and acc >= 0:
            prev = acc - float(cf)
            if float(cf) == 0:
                return float(t)
            return float(t - 1) + (-prev / float(cf))
    return float("inf")


def lcoe(
    capex_cny: float,
    annual_om_cny: Sequence[float],
    annual_energy_kwh: Sequence[float],
    discount_rate: float,
    *,
    residual_value_cny: float = 0.0,
) -> float:
    """度电成本 元/kWh：折现成本 / 折现电量。"""
    r = float(discount_rate)
    if r <= -1.0:
        raise ValueError("discount_rate 非法")
    n = max(len(annual_om_cny), len(annual_energy_kwh))
    if n == 0:
        return float("nan")
    cost_pv = float(capex_cny)
    energy_pv = 0.0
    for t in range(1, n + 1):
        om = float(annual_om_cny[t - 1]) if t - 1 < len(annual_om_cny) else 0.0
        e = float(annual_energy_kwh[t - 1]) if t - 1 < len(annual_energy_kwh) else 0.0
        disc = (1.0 + r) ** t
        cost_pv += om / disc
        energy_pv += e / disc
    if residual_value_cny:
        cost_pv -= float(residual_value_cny) / ((1.0 + r) ** n)
    if energy_pv <= 0:
        return float("nan")
    return float(cost_pv / energy_pv)


def annual_savings_cny(
    baseline_grid_cost_cny: float,
    optimized_grid_cost_cny: float,
    *,
    other_om_delta_cny: float = 0.0,
) -> float:
    """相对全网购电基线的年节省额。"""
    return float(baseline_grid_cost_cny - optimized_grid_cost_cny - other_om_delta_cny)
