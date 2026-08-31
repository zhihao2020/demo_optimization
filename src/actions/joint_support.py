"""Analytic grid-coupled joint support. Not a cartesian product of device boxes.

Power sign convention matches FeasibilityOracle.predict_p_grid:
    p_grid = P_T u_T - P_B u_B - P_C u_C - R
    R = p_wind + p_pv + p_load  (FMU: load +, wind/PV generation -)
    grid buy +, sell -.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class GridCoupling:
    """Plant ratings and current residual used by the joint decoder."""

    p_thermal: float
    p_battery: float
    p_caes: float
    residual: float
    g_min: float
    g_max: float


def residual_from_outputs(outputs: Mapping[str, Any]) -> float:
    """Net wind + PV + load (W), same terms as ``predict_p_grid``."""
    p_wind = float(outputs.get("p_wind_actual", outputs.get("p_wind_available", 0.0)) or 0.0)
    p_pv = float(outputs.get("p_pv_actual", outputs.get("p_pv_available", 0.0)) or 0.0)
    p_load = float(outputs.get("p_load_actual", 0.0) or 0.0)
    return p_wind + p_pv + p_load


def coupling_from_feasible(feasible: Any) -> GridCoupling | None:
    """Recover coupling stored on ``DynamicFeasibleActionSet.metadata``."""
    meta = getattr(feasible, "metadata", None) or {}
    if not meta.get("joint_grid_coupling"):
        return None
    return GridCoupling(
        p_thermal=float(meta["p_cap_thermal_W"]),
        p_battery=float(meta["p_cap_battery_W"]),
        p_caes=float(meta["p_cap_caes_W"]),
        residual=float(meta["grid_residual_W"]),
        g_min=float(meta["grid_g_min_W"]),
        g_max=float(meta["grid_g_max_W"]),
    )


def coupling_from_oracle(oracle: Any, outputs: Mapping[str, Any]) -> GridCoupling:
    """Build coupling from a live oracle and current FMU outputs."""
    params = oracle.params
    g = params["grid"]
    gm = float((oracle.margins.get("grid") or {}).get("margin_W", 0.0) or 0.0)
    return GridCoupling(
        p_thermal=float(params["thermal"]["P_cap_W"]),
        p_battery=float(params["battery"]["P_cap_W"]),
        p_caes=float(params["caes"]["P_cap_W"]),
        residual=residual_from_outputs(outputs),
        g_min=float(g["P_max_sell_W"]) + gm,
        g_max=float(g["P_max_buy_W"]) - gm,
    )


def predict_p_grid_u(ctx: GridCoupling, u_tp: float, u_bat: float, u_caes: float) -> float:
    """p_grid(W) from the three commands (same identity as the oracle)."""
    return (
        ctx.p_thermal * float(u_tp)
        - ctx.p_battery * float(u_bat)
        - ctx.p_caes * float(u_caes)
        - ctx.residual
    )


def _intersect(lo: float, hi: float, glo: float, ghi: float) -> tuple[float, float] | None:
    nlo = max(float(lo), float(glo))
    nhi = min(float(hi), float(ghi))
    if nlo > nhi + 1e-12:
        return None
    return (nlo, nhi)


def caes_grid_window(ctx: GridCoupling, tp_lo: float, tp_hi: float, bat_lo: float, bat_hi: float) -> tuple[float, float]:
    """u_C interval for which some (u_T, u_B) in the device boxes meets the grid band."""
    p_t, p_b, p_c = ctx.p_thermal, ctx.p_battery, ctx.p_caes
    r, gmin, gmax = ctx.residual, ctx.g_min, ctx.g_max
    lo = (p_t * tp_lo - p_b * bat_hi - r - gmax) / p_c
    hi = (p_t * tp_hi - p_b * bat_lo - r - gmin) / p_c
    if lo > hi:
        lo, hi = hi, lo
    return (float(lo), float(hi))


def intersect_caes_interval(
    device_lo: float, device_hi: float, ctx: GridCoupling, tp_lo: float, tp_hi: float, bat_lo: float, bat_hi: float
) -> tuple[float, float] | None:
    glo, ghi = caes_grid_window(ctx, tp_lo, tp_hi, bat_lo, bat_hi)
    return _intersect(device_lo, device_hi, glo, ghi)


def thermal_window(
    ctx: GridCoupling,
    u_caes: float,
    tp_lo: float,
    tp_hi: float,
    bat_lo: float,
    bat_hi: float,
) -> tuple[float, float] | None:
    """Thermal interval such that some battery command still meets the grid band."""
    p_t, p_b, p_c = ctx.p_thermal, ctx.p_battery, ctx.p_caes
    r, gmin, gmax = ctx.residual, ctx.g_min, ctx.g_max
    lo = (gmin + p_b * bat_lo + p_c * u_caes + r) / p_t
    hi = (gmax + p_b * bat_hi + p_c * u_caes + r) / p_t
    return _intersect(tp_lo, tp_hi, lo, hi)


def battery_window(
    ctx: GridCoupling,
    u_tp: float,
    u_caes: float,
    bat_lo: float,
    bat_hi: float,
) -> tuple[float, float] | None:
    """Battery interval conditioned on chosen thermal and CAES commands."""
    p_t, p_b, p_c = ctx.p_thermal, ctx.p_battery, ctx.p_caes
    r, gmin, gmax = ctx.residual, ctx.g_min, ctx.g_max
    lo = (p_t * u_tp - p_c * u_caes - r - gmax) / p_b
    hi = (p_t * u_tp - p_c * u_caes - r - gmin) / p_b
    return _intersect(bat_lo, bat_hi, lo, hi)


def tighten_caes_modes(
    ctx: GridCoupling,
    tp_lo: float,
    tp_hi: float,
    bat_lo: float,
    bat_hi: float,
    dis: tuple[float, float] | None,
    chg: tuple[float, float] | None,
    idle_ok: bool,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None, bool, bool, bool]:
    """Intersect charge/discharge bands with the grid window; drop empty modes."""
    new_dis = None
    if dis is not None:
        new_dis = intersect_caes_interval(dis[0], dis[1], ctx, tp_lo, tp_hi, bat_lo, bat_hi)
    new_chg = None
    if chg is not None:
        new_chg = intersect_caes_interval(chg[0], chg[1], ctx, tp_lo, tp_hi, bat_lo, bat_hi)
    idle = bool(idle_ok)
    if idle:
        # Idle is feasible iff some (u_T, u_B) works at u_C=0.
        idle = thermal_window(ctx, 0.0, tp_lo, tp_hi, bat_lo, bat_hi) is not None
    return new_dis, new_chg, new_dis is not None, idle, new_chg is not None


def decode_joint_torch(
    u_tp,
    u_bat,
    u_caes,
    tp_lo,
    tp_hi,
    bat_lo,
    bat_hi,
    residual,
    g_min,
    g_max,
    p_t,
    p_b,
    p_c,
):
    """Batched differentiable re-map onto grid-conditioned intervals."""
    import torch

    def _ix(lo, hi, glo, ghi):
        nlo = torch.maximum(lo, glo)
        nhi = torch.minimum(hi, ghi)
        empty = nlo > nhi
        nlo = torch.where(empty, lo, nlo)
        nhi = torch.where(empty, hi, nhi)
        return nlo, nhi

    tp_star_lo = (g_min + p_b * bat_lo + p_c * u_caes + residual) / p_t
    tp_star_hi = (g_max + p_b * bat_hi + p_c * u_caes + residual) / p_t
    tlo, thi = _ix(tp_lo, tp_hi, tp_star_lo, tp_star_hi)
    span = (tp_hi - tp_lo).clamp_min(1e-12)
    alpha = ((u_tp - tp_lo) / span).clamp(0.0, 1.0)
    u_tp2 = tlo + alpha * (thi - tlo)
    bat_star_lo = (p_t * u_tp2 - p_c * u_caes - residual - g_max) / p_b
    bat_star_hi = (p_t * u_tp2 - p_c * u_caes - residual - g_min) / p_b
    blo, bhi = _ix(bat_lo, bat_hi, bat_star_lo, bat_star_hi)
    span_b = (bat_hi - bat_lo).clamp_min(1e-12)
    beta = ((u_bat - bat_lo) / span_b).clamp(0.0, 1.0)
    u_bat2 = blo + beta * (bhi - blo)
    return u_tp2, u_bat2


def decode_joint_numpy(
    ctx: GridCoupling,
    u_tp_raw_lo: float,
    u_tp_raw_hi: float,
    u_bat_raw_lo: float,
    u_bat_raw_hi: float,
    u_caes: float,
    z_tp: float,
    z_bat: float,
) -> tuple[float, float]:
    """Map unit thermal/battery scores through grid-conditioned intervals.

    ``z_tp`` / ``z_bat`` are already in the physical device interval (sigmoid-mapped).
    They are re-scaled into the tightened window when that window is nonempty.
    """
    tp_win = thermal_window(ctx, u_caes, u_tp_raw_lo, u_tp_raw_hi, u_bat_raw_lo, u_bat_raw_hi)
    if tp_win is None:
        u_tp = float(np.clip(z_tp, u_tp_raw_lo, u_tp_raw_hi))
    else:
        span = max(u_tp_raw_hi - u_tp_raw_lo, 1e-12)
        alpha = float(np.clip((z_tp - u_tp_raw_lo) / span, 0.0, 1.0))
        u_tp = tp_win[0] + alpha * (tp_win[1] - tp_win[0])
    bat_win = battery_window(ctx, u_tp, u_caes, u_bat_raw_lo, u_bat_raw_hi)
    if bat_win is None:
        u_bat = float(np.clip(z_bat, u_bat_raw_lo, u_bat_raw_hi))
    else:
        span = max(u_bat_raw_hi - u_bat_raw_lo, 1e-12)
        alpha = float(np.clip((z_bat - u_bat_raw_lo) / span, 0.0, 1.0))
        u_bat = bat_win[0] + alpha * (bat_win[1] - bat_win[0])
    return float(u_tp), float(u_bat)
