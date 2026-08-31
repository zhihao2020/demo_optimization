"""CAES 物理指令：决策是 (mode, magnitude)，写入 FMU 仍是一个 ``u_caes``。

合法集仍是断开的三段，问题仍非凸。``hybrid_caes`` 路径由模式头+幅值头解码；
旧路径把一个 tanh 标量投影到三段（空隙→0）。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .types import CaesMode

# 放电区两端 -1.0 ~ -0.33；充电区两端 0.86 ~ 1.0；闲 = 0
DISCHARGE_LO = -1.0
DISCHARGE_HI = -0.33
CHARGE_LO = 0.86
CHARGE_HI = 1.0
IDLE_U = 0.0

_EPS = 1e-6


def mode_from_u(u: float) -> CaesMode:
    """由物理 u_caes 派生模式（锁/最短运行/日志），非策略输出维。"""
    if abs(float(u)) <= _EPS:
        return CaesMode.IDLE
    return CaesMode.DISCHARGE if float(u) < 0.0 else CaesMode.CHARGE


def in_discharge_band(u: float) -> bool:
    return DISCHARGE_LO - _EPS <= float(u) <= DISCHARGE_HI + _EPS


def in_charge_band(u: float) -> bool:
    return CHARGE_LO - _EPS <= float(u) <= CHARGE_HI + _EPS


def in_idle(u: float) -> bool:
    return abs(float(u)) <= _EPS


def is_legal_u_caes(u: float) -> bool:
    """是否落在合法集 [-1,-0.33]∪{0}∪[0.86,1]。"""
    return in_discharge_band(u) or in_idle(u) or in_charge_band(u)


def project_u_caes(u: float) -> float:
    """将任意标量投影到合法三段：空隙 → 0；带内夹到端点。

    仅数值稳定/合法化，不改变物理合法集（仍非凸）。
    """
    u = float(u)
    if not np.isfinite(u):
        return IDLE_U
    if u < DISCHARGE_LO:
        return DISCHARGE_LO
    if DISCHARGE_LO <= u <= DISCHARGE_HI:
        return u
    if CHARGE_LO <= u <= CHARGE_HI:
        return u
    if u > CHARGE_HI:
        return CHARGE_HI
    # 空隙 (DISCHARGE_HI, CHARGE_LO) \ {0} → idle
    return IDLE_U


def project_u_caes_torch(u: torch.Tensor) -> torch.Tensor:
    """Batch 版 project_u_caes（元素级）。"""
    u = u.float()
    out = torch.zeros_like(u)
    # discharge band
    dis = (u >= DISCHARGE_LO) & (u <= DISCHARGE_HI)
    out = torch.where(dis, u, out)
    out = torch.where(u < DISCHARGE_LO, torch.full_like(u, DISCHARGE_LO), out)
    # charge band
    chg = (u >= CHARGE_LO) & (u <= CHARGE_HI)
    out = torch.where(chg, u, out)
    out = torch.where(u > CHARGE_HI, torch.full_like(u, CHARGE_HI), out)
    # gap → 0 already
    return out


def apply_mode_mask_to_u(u: float, *, discharge: bool, charge: bool, idle: bool = True) -> float:
    """按 mode mask 禁止方向：禁止放则 u<0→0；禁止充则 u>0→0。"""
    u = project_u_caes(u)
    if u < 0.0 and not discharge:
        return IDLE_U if idle else u
    if u > 0.0 and not charge:
        return IDLE_U if idle else u
    return u


def apply_mode_mask_to_u_torch(u: torch.Tensor, mode_mask: torch.Tensor) -> torch.Tensor:
    """mode_mask (B,3) bool: [discharge, idle, charge]。非法方向 → 0。"""
    u = project_u_caes_torch(u)
    # mode indices: DISCHARGE=0, IDLE=1, CHARGE=2
    dis_ok = mode_mask[:, 0]
    chg_ok = mode_mask[:, 2]
    u = torch.where((u < 0) & (~dis_ok), torch.zeros_like(u), u)
    u = torch.where((u > 0) & (~chg_ok), torch.zeros_like(u), u)
    return u


def clamp_u_caes_to_spec(u: float, feasible: Any) -> tuple[float, bool]:
    """把 u_caes 的幅值夹进该方向的安全子区间。

    只处理幅值，不处理方向：方向是否合法由模式掩码表达，非法方向仍须被
    ``PhysicalActionValidator`` 拒绝并计入审计——这是智能体必须学会规避的约束，
    也是「非法动作永不触达 FMU」这一性质的依据。而安全幅值子区间逐步随状态变化，
    智能体无从预知，故按投影处理并单独审计。

    Args:
        u: 原始 CAES 指令。
        feasible: DynamicFeasibleActionSet，需含 mode_mask 与幅值区间。

    Returns:
        (投影后的 u_caes, 幅值是否被夹紧)。

    Raises:
        无。
    """
    original = float(u)
    mask = feasible.mode_mask
    if original < 0.0:
        if not mask.discharge:
            return original, False
        span = getattr(feasible, "u_caes_discharge", None)
    elif original > 0.0:
        if not mask.charge:
            return original, False
        span = getattr(feasible, "u_caes_charge", None)
    else:
        return original, False
    if span is None:
        return original, False
    lo, hi = float(span[0]), float(span[1])
    out = float(np.clip(original, min(lo, hi), max(lo, hi)))
    return out, bool(abs(out - original) > _EPS)


def u_from_mode_mag(mode: CaesMode | int, mag: float) -> float:
    """mode + mag∈[0,1] → 合法带内 u_caes。idle 忽略幅值。"""
    mode_i = int(mode)
    mag = 0.0 if mode_i == int(CaesMode.IDLE) else float(np.clip(mag, 0.0, 1.0))
    if mode_i == int(CaesMode.DISCHARGE):
        return float(DISCHARGE_LO + mag * (DISCHARGE_HI - DISCHARGE_LO))
    if mode_i == int(CaesMode.CHARGE):
        return float(CHARGE_LO + mag * (CHARGE_HI - CHARGE_LO))
    return IDLE_U


def u_from_mode_onehot_torch(onehot: torch.Tensor, mag: torch.Tensor) -> torch.Tensor:
    """(B,3) one-hot [dis, idle, chg] and mag∈[0,1] → u_caes on the static envelope."""
    mag = mag.clamp(0.0, 1.0)
    u_dis = DISCHARGE_LO + mag * (DISCHARGE_HI - DISCHARGE_LO)
    u_chg = CHARGE_LO + mag * (CHARGE_HI - CHARGE_LO)
    return onehot[:, 0] * u_dis + onehot[:, 2] * u_chg


def caes_intervals_from_feasible(feasible: Any) -> dict[str, float]:
    """Oracle intervals with static-envelope fallback when a direction is closed."""
    dis = getattr(feasible, "u_caes_discharge", None)
    chg = getattr(feasible, "u_caes_charge", None)
    d_lo, d_hi = (float(dis[0]), float(dis[1])) if dis is not None else (DISCHARGE_LO, DISCHARGE_HI)
    c_lo, c_hi = (float(chg[0]), float(chg[1])) if chg is not None else (CHARGE_LO, CHARGE_HI)
    return {
        "u_caes_discharge_low": d_lo,
        "u_caes_discharge_high": d_hi,
        "u_caes_charge_low": c_lo,
        "u_caes_charge_high": c_hi,
    }


def feasible_bound_dict(feasible: Any) -> dict[str, float]:
    """Thermal/battery + CAES oracle intervals for replay."""
    out = {
        "u_tp_low": float(feasible.u_tp_low),
        "u_tp_high": float(feasible.u_tp_high),
        "u_battery_low": float(feasible.u_battery_low),
        "u_battery_high": float(feasible.u_battery_high),
    }
    out.update(caes_intervals_from_feasible(feasible))
    meta = getattr(feasible, "metadata", None) or {}
    if meta.get("joint_grid_coupling"):
        out["grid_residual_W"] = float(meta["grid_residual_W"])
        out["grid_g_min_W"] = float(meta["grid_g_min_W"])
        out["grid_g_max_W"] = float(meta["grid_g_max_W"])
        out["p_cap_thermal_W"] = float(meta["p_cap_thermal_W"])
        out["p_cap_battery_W"] = float(meta["p_cap_battery_W"])
        out["p_cap_caes_W"] = float(meta["p_cap_caes_W"])
    return out


def u_from_mode_mag_feasible(feasible: Any, mode: CaesMode | int, mag: float) -> float:
    """Decode (mode, mag) onto the current oracle interval, not the static envelope."""
    mode_i = int(mode)
    mag = 0.0 if mode_i == int(CaesMode.IDLE) else float(np.clip(mag, 0.0, 1.0))
    iv = caes_intervals_from_feasible(feasible)
    if mode_i == int(CaesMode.DISCHARGE):
        lo, hi = iv["u_caes_discharge_low"], iv["u_caes_discharge_high"]
        return float(lo + mag * (hi - lo))
    if mode_i == int(CaesMode.CHARGE):
        lo, hi = iv["u_caes_charge_low"], iv["u_caes_charge_high"]
        return float(lo + mag * (hi - lo))
    return IDLE_U


def u_from_mode_onehot_dynamic(
    onehot: torch.Tensor,
    mag: torch.Tensor,
    dis_lo: torch.Tensor,
    dis_hi: torch.Tensor,
    chg_lo: torch.Tensor,
    chg_hi: torch.Tensor,
) -> torch.Tensor:
    """Decode onto state-dependent [l_D,h_D] / [l_C,h_C]. Idle is a point mass at 0."""
    mag = mag.clamp(0.0, 1.0)
    u_dis = dis_lo + mag * (dis_hi - dis_lo)
    u_chg = chg_lo + mag * (chg_hi - chg_lo)
    return onehot[:, 0] * u_dis + onehot[:, 2] * u_chg


def legalize_mode_mask(mode_mask: torch.Tensor) -> torch.Tensor:
    """(B,3) or (3,) bool mask; if a row is all-false, allow every mode."""
    legal = mode_mask.to(dtype=torch.bool)
    if legal.dim() == 1:
        legal = legal.view(1, -1)
    if legal.size(-1) != 3:
        raise ValueError(f"mode_mask last dim must be 3, got {tuple(legal.shape)}")
    fill = legal.any(dim=-1, keepdim=True)
    return torch.where(fill, legal, torch.ones_like(legal))


def mask_mode_logits(logits: torch.Tensor, mode_mask: torch.Tensor) -> torch.Tensor:
    """Illegal modes → -1e9 so softmax/argmax cannot pick them."""
    if logits.dim() == 1:
        logits = logits.view(1, -1)
    legal = legalize_mode_mask(mode_mask)
    if legal.size(0) == 1 and logits.size(0) > 1:
        legal = legal.expand(logits.size(0), -1)
    return logits.masked_fill(~legal, -1.0e9)


def gumbel_mode_onehot(
    logits: torch.Tensor,
    tau: float = 1.0,
    *,
    deterministic: bool = False,
    soft_for_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked logits → (one-hot [B,3], idx [B]). Straight-through when training."""
    if deterministic:
        idx = logits.argmax(dim=-1)
        onehot = torch.nn.functional.one_hot(idx, 3).to(dtype=logits.dtype)
        return onehot, idx
    gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-6, 1.0)))
    y_soft = torch.softmax((logits + gumbel) / max(float(tau), 1e-3), dim=-1)
    idx = y_soft.argmax(dim=-1)
    y_hard = torch.nn.functional.one_hot(idx, 3).to(dtype=y_soft.dtype)
    onehot = y_hard + y_soft - y_soft.detach() if soft_for_grad else y_hard
    return onehot, idx


def mode_index_from_u_torch(u: torch.Tensor) -> torch.Tensor:
    """Physical u_caes → mode index 0=dis, 1=idle, 2=chg."""
    u = u.float()
    idx = torch.ones_like(u, dtype=torch.long)
    idx = torch.where(u < -_EPS, torch.zeros_like(idx), idx)
    idx = torch.where(u > _EPS, torch.full_like(idx, 2), idx)
    return idx


def mag_from_u_torch(u: torch.Tensor) -> torch.Tensor:
    """Legal-band u → [0,1] magnitude (idle → 0)."""
    u = u.float()
    mag_dis = ((u - DISCHARGE_LO) / (DISCHARGE_HI - DISCHARGE_LO)).clamp(0.0, 1.0)
    mag_chg = ((u - CHARGE_LO) / (CHARGE_HI - CHARGE_LO)).clamp(0.0, 1.0)
    mag = torch.where(u < -_EPS, mag_dis, torch.where(u > _EPS, mag_chg, torch.zeros_like(u)))
    return mag


def perturb_u_caes_keep_mode(u: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Add noise inside the current legal band; idle stays idle."""
    u = u.float()
    noise = noise.float()
    idle = u.abs() <= _EPS
    dis = u < -_EPS
    u_n = u + noise
    u_dis = u_n.clamp(DISCHARGE_LO, DISCHARGE_HI)
    u_chg = u_n.clamp(CHARGE_LO, CHARGE_HI)
    out = torch.where(dis, u_dis, u_chg)
    return torch.where(idle, torch.zeros_like(u), out)


def compressor_expander_bits(mode: CaesMode | int) -> tuple[int, int]:
    """Cui-style (u_com, u_tur): charge→(1,0), discharge→(0,1), idle→(0,0)."""
    m = int(mode)
    if m == int(CaesMode.CHARGE):
        return 1, 0
    if m == int(CaesMode.DISCHARGE):
        return 0, 1
    return 0, 0


def startup_events(prev_mode: CaesMode | int, next_mode: CaesMode | int) -> int:
    """Number of compressor/expander start–stop events (0, 1, or 2)."""
    c0, t0 = compressor_expander_bits(prev_mode)
    c1, t1 = compressor_expander_bits(next_mode)
    return int(abs(c1 - c0) + abs(t1 - t0))


def mag_from_u(u: float) -> float:
    """诊断：合法带内 u → [0,1] 幅值。"""
    u = float(u)
    mode = mode_from_u(u)
    if mode == CaesMode.IDLE:
        return 0.0
    if mode == CaesMode.DISCHARGE:
        return float(np.clip((u - DISCHARGE_LO) / (DISCHARGE_HI - DISCHARGE_LO), 0.0, 1.0))
    return float(np.clip((u - CHARGE_LO) / (CHARGE_HI - CHARGE_LO), 0.0, 1.0))


def np_as_scalar(value: Any) -> float:
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        return float(value[0] if len(value) else 0.0)
    return float(value)


def physical_dict(
    u_tp: float,
    u_battery: float,
    u_caes: float,
) -> dict[str, np.ndarray]:
    """环境可 step 的三连续动作字典。"""
    return {
        "u_tp": np.asarray([float(u_tp)], dtype=np.float32),
        "u_battery": np.asarray([float(u_battery)], dtype=np.float32),
        "u_caes": np.asarray([float(project_u_caes(u_caes))], dtype=np.float32),
    }
