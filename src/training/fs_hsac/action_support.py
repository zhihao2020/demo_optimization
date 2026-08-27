"""Convert DynamicFeasibleActionSet into batched tensor supports for FS-HSAC."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from actions.caes_u import CHARGE_HI, CHARGE_LO, DISCHARGE_HI, DISCHARGE_LO
from actions.feasible_set import DynamicFeasibleActionSet

# mode order: discharge, idle, charge
MODE_DISCHARGE = 0
MODE_IDLE = 1
MODE_CHARGE = 2

MIN_SPAN = 1e-4
EPS = 1e-6


def _default_discharge() -> tuple[float, float]:
    return float(DISCHARGE_LO), float(DISCHARGE_HI)


def _default_charge() -> tuple[float, float]:
    return float(CHARGE_LO), float(CHARGE_HI)


def feasible_to_support_dict(feasible: DynamicFeasibleActionSet | Any) -> dict[str, float | bool]:
    """Serialize one feasible set into flat numeric fields for replay."""
    mask = feasible.mode_mask
    dis = getattr(feasible, "u_caes_discharge", None)
    chg = getattr(feasible, "u_caes_charge", None)
    if dis is None:
        d_lo, d_hi = _default_discharge()
        dis_ok = bool(mask.discharge)
    else:
        d_lo, d_hi = float(dis[0]), float(dis[1])
        dis_ok = bool(mask.discharge) and (max(d_lo, d_hi) - min(d_lo, d_hi) >= MIN_SPAN)
    if chg is None:
        c_lo, c_hi = _default_charge()
        chg_ok = bool(mask.charge)
    else:
        c_lo, c_hi = float(chg[0]), float(chg[1])
        chg_ok = bool(mask.charge) and (max(c_lo, c_hi) - min(c_lo, c_hi) >= MIN_SPAN)
    # keep ordered lo <= hi for discharge (both negative) and charge
    d_lo, d_hi = (min(d_lo, d_hi), max(d_lo, d_hi))
    c_lo, c_hi = (min(c_lo, c_hi), max(c_lo, c_hi))
    idle_ok = bool(mask.idle)
    if not (dis_ok or idle_ok or chg_ok):
        idle_ok = True
    return {
        "u_tp_low": float(feasible.u_tp_low),
        "u_tp_high": float(feasible.u_tp_high),
        "u_battery_low": float(feasible.u_battery_low),
        "u_battery_high": float(feasible.u_battery_high),
        "mode_discharge": dis_ok,
        "mode_idle": idle_ok,
        "mode_charge": chg_ok,
        "u_caes_discharge_low": d_lo,
        "u_caes_discharge_high": d_hi,
        "u_caes_charge_low": c_lo,
        "u_caes_charge_high": c_hi,
    }


def stack_supports(
    supports: Sequence[Mapping[str, Any]],
    *,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Stack a list of support dicts into batch tensors."""
    n = len(supports)
    if n == 0:
        raise ValueError("empty support batch")
    mode_mask = torch.zeros(n, 3, dtype=torch.bool)
    u_tp_low = torch.zeros(n)
    u_tp_high = torch.zeros(n)
    u_bat_low = torch.zeros(n)
    u_bat_high = torch.zeros(n)
    dis_lo = torch.zeros(n)
    dis_hi = torch.zeros(n)
    chg_lo = torch.zeros(n)
    chg_hi = torch.zeros(n)
    for i, s in enumerate(supports):
        mode_mask[i, 0] = bool(s["mode_discharge"])
        mode_mask[i, 1] = bool(s["mode_idle"])
        mode_mask[i, 2] = bool(s["mode_charge"])
        u_tp_low[i] = float(s["u_tp_low"])
        u_tp_high[i] = float(s["u_tp_high"])
        u_bat_low[i] = float(s.get("u_battery_low", s.get("u_bat_low")))
        u_bat_high[i] = float(s.get("u_battery_high", s.get("u_bat_high")))
        dis_lo[i] = float(s["u_caes_discharge_low"])
        dis_hi[i] = float(s["u_caes_discharge_high"])
        chg_lo[i] = float(s["u_caes_charge_low"])
        chg_hi[i] = float(s["u_caes_charge_high"])
    # drop degenerate continuous modes
    dis_span = (dis_hi - dis_lo).abs()
    chg_span = (chg_hi - chg_lo).abs()
    mode_mask[:, 0] = mode_mask[:, 0] & (dis_span >= MIN_SPAN)
    mode_mask[:, 2] = mode_mask[:, 2] & (chg_span >= MIN_SPAN)
    empty = ~mode_mask.any(dim=-1)
    if empty.any():
        mode_mask = mode_mask.clone()
        mode_mask[empty, MODE_IDLE] = True
    out = {
        "mode_mask": mode_mask,
        "u_tp_low": u_tp_low,
        "u_tp_high": u_tp_high,
        "u_bat_low": u_bat_low,
        "u_bat_high": u_bat_high,
        "dis_lo": dis_lo,
        "dis_hi": dis_hi,
        "chg_lo": chg_lo,
        "chg_hi": chg_hi,
        "n_modes": mode_mask.sum(dim=-1).to(dtype=torch.float32),
    }
    if device is not None:
        out = {k: v.to(device) for k, v in out.items()}
    return out


def support_from_feasible_batch(
    feasibles: Sequence[DynamicFeasibleActionSet | Any],
    *,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    return stack_supports([feasible_to_support_dict(f) for f in feasibles], device=device)


def map_unit_to_interval(
    y: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
) -> torch.Tensor:
    """Map y in (0,1) to [lo, hi]."""
    return lo + y * (hi - lo)


def interval_log_jacobian(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """log |du/dy| for u = lo + y*(hi-lo), y=sigmoid(z)."""
    return torch.log((hi - lo).clamp_min(MIN_SPAN))


def sigmoid_log_jacobian(z: torch.Tensor) -> torch.Tensor:
    """log |dy/dz| for y = sigmoid(z)."""
    # y*(1-y) = sigmoid(z)*sigmoid(-z)
    return -torch.nn.functional.softplus(-z) - torch.nn.functional.softplus(z)


def decode_mode_u(
    mode_idx: torch.Tensor,
    mag01: torch.Tensor,
    dis_lo: torch.Tensor,
    dis_hi: torch.Tensor,
    chg_lo: torch.Tensor,
    chg_hi: torch.Tensor,
) -> torch.Tensor:
    """Decode mode index + normalized mag into physical u_caes."""
    mag01 = mag01.clamp(0.0, 1.0)
    u_dis = map_unit_to_interval(mag01, dis_lo, dis_hi)
    u_chg = map_unit_to_interval(mag01, chg_lo, chg_hi)
    u = torch.zeros_like(mag01)
    u = torch.where(mode_idx == MODE_DISCHARGE, u_dis, u)
    u = torch.where(mode_idx == MODE_CHARGE, u_chg, u)
    return u


def encode_mag01_from_u(
    u: torch.Tensor,
    mode_idx: torch.Tensor,
    dis_lo: torch.Tensor,
    dis_hi: torch.Tensor,
    chg_lo: torch.Tensor,
    chg_hi: torch.Tensor,
) -> torch.Tensor:
    """Invert decode for continuous modes; idle -> 0."""
    mag_dis = ((u - dis_lo) / (dis_hi - dis_lo).clamp_min(MIN_SPAN)).clamp(0.0, 1.0)
    mag_chg = ((u - chg_lo) / (chg_hi - chg_lo).clamp_min(MIN_SPAN)).clamp(0.0, 1.0)
    mag = torch.zeros_like(u)
    mag = torch.where(mode_idx == MODE_DISCHARGE, mag_dis, mag)
    mag = torch.where(mode_idx == MODE_CHARGE, mag_chg, mag)
    return mag


def one_hot_modes(idx: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.one_hot(idx.long(), 3).to(dtype=torch.float32)


def numpy_support_keys() -> tuple[str, ...]:
    return (
        "u_tp_low",
        "u_tp_high",
        "u_battery_low",
        "u_battery_high",
        "mode_discharge",
        "mode_idle",
        "mode_charge",
        "u_caes_discharge_low",
        "u_caes_discharge_high",
        "u_caes_charge_low",
        "u_caes_charge_high",
    )
