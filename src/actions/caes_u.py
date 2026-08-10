"""CAES 物理指令 u_caes：合法三段、投影、模式派生（仅锁/日志）。

策略与 FMU 共用同一个连续标量 ``u_caes``；不引入 mode/magnitude 动作维。
合法集仍是断开的三段，问题仍非凸——本模块只做表示与数值合法化。
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


def u_from_mode_mag(mode: CaesMode | int, mag: float) -> float:
    """内部/测试辅助：mode+mag → u_caes（非策略主路径）。"""
    mode_i = int(mode)
    mag = 0.0 if mode_i == int(CaesMode.IDLE) else float(np.clip(mag, 0.0, 1.0))
    if mode_i == int(CaesMode.DISCHARGE):
        return float(DISCHARGE_LO + mag * (DISCHARGE_HI - DISCHARGE_LO))
    if mode_i == int(CaesMode.CHARGE):
        return float(CHARGE_LO + mag * (CHARGE_HI - CHARGE_LO))
    return IDLE_U


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
