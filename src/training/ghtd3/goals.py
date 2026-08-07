"""分层 goal：贴合 Modelica 的 5 维厂站意图 + 转移/内在奖励/市场 prior。"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# 能量主状态 + 热过程（聚合用）
ENERGY_SOC_KEYS = ("battery_soc", "caes_gas_soc")
PROCESS_SOC_KEYS = ("caes_hot_soc", "caes_cold_soc")
ALL_SOC_KEYS = ENERGY_SOC_KEYS + PROCESS_SOC_KEYS
# 兼容旧代码
DEFAULT_SOC_KEYS = ENERGY_SOC_KEYS

# goal 分量索引
G_BAT, G_GAS, G_TH, G_UTP, G_ARB = 0, 1, 2, 3, 4
GOAL_NAMES = ("d_bat", "d_gas", "d_th", "u_tp_bias", "arb")


def default_goal_boxes() -> tuple[np.ndarray, np.ndarray]:
    """Modelica 对齐的默认 goal 盒。"""
    low = np.asarray([-0.30, -0.12, -0.10, -0.20, 0.0], dtype=np.float32)
    high = np.asarray([0.30, 0.12, 0.10, 0.20, 1.0], dtype=np.float32)
    return low, high


def extract_soc(outputs: dict[str, float], keys: Sequence[str] = ENERGY_SOC_KEYS) -> np.ndarray:
    return np.asarray([float(outputs[k]) for k in keys], dtype=np.float32)


def extract_soc_from_obs(obs: np.ndarray, n: int = 2) -> np.ndarray:
    """物理观测前 n 维为 SoC（ObservationBuilder 顺序）。"""
    o = np.asarray(obs, dtype=np.float32).ravel()
    return o[:n].copy()


def extract_plant_state(outputs: dict[str, float]) -> dict[str, float]:
    """从 FMU 输出抽取厂站意图相关状态。"""
    bat = float(outputs.get("battery_soc", 0.5))
    gas = float(outputs.get("caes_gas_soc", 0.8))
    hot = float(outputs.get("caes_hot_soc", 0.5))
    cold = float(outputs.get("caes_cold_soc", 0.5))
    return {
        "battery_soc": bat,
        "caes_gas_soc": gas,
        "caes_hot_soc": hot,
        "caes_cold_soc": cold,
        "th_mean": 0.5 * (hot + cold),
        "th_diff": hot - cold,
    }


def plant_intent_vector(outputs: dict[str, float]) -> np.ndarray:
    """意图空间状态 [bat, gas, th_mean] 用于 residual goal 跟踪。"""
    st = extract_plant_state(outputs)
    return np.asarray([st["battery_soc"], st["caes_gas_soc"], st["th_mean"]], dtype=np.float32)


def clip_goal(goal: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(np.asarray(goal, dtype=np.float32), low), high)


def goal_transition_intent(
    intent_t: np.ndarray,
    goal_t: np.ndarray,
    intent_tp1: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """对 bat/gas/th 三分量做 s+g-s'；u_tp_bias 与 arb 保持（窗级意图）。"""
    g = np.asarray(goal_t, dtype=np.float32).copy()
    if intent_t.size >= 3 and g.size >= 3:
        g[:3] = np.asarray(intent_t[:3], dtype=np.float32) + g[:3] - np.asarray(intent_tp1[:3], dtype=np.float32)
    return clip_goal(g, low, high)


def goal_transition(
    soc_t: np.ndarray,
    goal_t: np.ndarray,
    soc_tp1: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """兼容旧 2 维 API；5 维时仅更新前 min 维能量分量。"""
    g = np.asarray(goal_t, dtype=np.float32).copy()
    n = min(len(soc_t), len(soc_tp1), 2, len(g))
    g[:n] = np.asarray(soc_t[:n], dtype=np.float32) + g[:n] - np.asarray(soc_tp1[:n], dtype=np.float32)
    return clip_goal(g, low, high)


def structured_intrinsic_reward(
    intent_t: np.ndarray,
    goal_t: np.ndarray,
    intent_tp1: np.ndarray,
    u_tp: float,
    u_tp_hybrid: float,
    r_ext: float,
    *,
    alpha: float,
    weights: Sequence[float] | None = None,
    relax_thermal_floor: bool = False,
) -> tuple[float, dict[str, float]]:
    """5 维 Modelica 对齐内在奖励。

    跟踪：bat/gas/th residual + 火电偏置 |u_tp - clip(u_H + g_u)|。
    arb 不进 e，仅由执行侧缩放残差。

    When ``relax_thermal_floor`` is True (absolute GC / no hybrid teacher), do not
    force u_tgt ≥ 1/3 — that floor biases high thermal and fights economic J.
    """
    g = np.asarray(goal_t, dtype=np.float64).ravel()
    it = np.asarray(intent_t, dtype=np.float64).ravel()
    ip = np.asarray(intent_tp1, dtype=np.float64).ravel()
    w = np.asarray(weights if weights is not None else (1.5, 1.2, 0.25, 0.4), dtype=np.float64)
    # residual on first 3 intent dims
    n = min(3, it.size, ip.size, g.size)
    e = it[:n] + g[:n] - ip[:n]
    w_e = w[:n]
    track_sq = float(np.sum(w_e * e * e))
    # thermal bias track
    g_u = float(g[G_UTP]) if g.size > G_UTP else 0.0
    base_u = float(u_tp_hybrid)
    if relax_thermal_floor:
        # g_u is a load-rate bias in goal box (~[-0.2,0.2]); center around mid load.
        u_tgt = float(np.clip(0.5 + g_u + 0.15 * (base_u - 0.5), 0.0, 1.0))
    else:
        u_tgt = float(np.clip(base_u + g_u, 1.0 / 3.0, 1.0))
    e_u = abs(float(u_tp) - u_tgt)
    w_u = float(w[3]) if w.size > 3 else 0.4
    track_sq += w_u * e_u * e_u
    track = float(np.sqrt(max(track_sq, 0.0)))
    r_int = -track + float(alpha) * float(r_ext)
    return r_int, {
        "goal_tracking_error": track,
        "e_bat": float(e[0]) if n > 0 else 0.0,
        "e_gas": float(e[1]) if n > 1 else 0.0,
        "e_th": float(e[2]) if n > 2 else 0.0,
        "e_u_tp": float(e_u),
        "u_tgt": float(u_tgt),
        "intrinsic_reward": r_int,
        "extrinsic_reward": float(r_ext),
        "intrinsic_alpha": float(alpha),
        "g_arb": float(g[G_ARB]) if g.size > G_ARB else 0.0,
    }


def intrinsic_reward(
    soc_t: np.ndarray,
    goal_t: np.ndarray,
    soc_tp1: np.ndarray,
    r_ext: float,
    alpha: float,
) -> tuple[float, dict[str, float]]:
    """兼容旧 2 维调用。"""
    g = np.asarray(goal_t, dtype=np.float64).ravel()
    residual = np.asarray(soc_t, dtype=np.float64).ravel()[:2] + g[:2] - np.asarray(soc_tp1, dtype=np.float64).ravel()[:2]
    track = float(np.linalg.norm(residual, ord=2))
    r_int = -track + float(alpha) * float(r_ext)
    return r_int, {
        "goal_tracking_error": track,
        "intrinsic_reward": r_int,
        "extrinsic_reward": float(r_ext),
        "intrinsic_alpha": float(alpha),
    }


def actual_delta_soc(soc_t: np.ndarray, soc_tp1: np.ndarray) -> np.ndarray:
    return np.asarray(soc_tp1, dtype=np.float32) - np.asarray(soc_t, dtype=np.float32)


def residual_scale_from_goal(
    goal: np.ndarray,
    *,
    alpha0: float = 0.0,
    alpha_max: float = 0.30,
) -> float:
    """g_arb ∈ [0,1] → 残差混合系数，默认封顶 0.30 以保护 Hybrid 下界。"""
    g = np.asarray(goal, dtype=np.float64).ravel()
    arb = float(g[G_ARB]) if g.size > G_ARB else 0.0
    arb = float(np.clip(arb, 0.0, 1.0))
    a0 = float(np.clip(alpha0, 0.0, 1.0))
    amax = float(np.clip(alpha_max, 0.0, 1.0))
    # 残差只做“微调”，避免冲掉 Hybrid 强执行器
    return float(np.clip(a0 + (amax - a0) * arb, 0.0, amax))


def market_conditioned_goal_prior(
    buy_price: float | None,
    soc_now: np.ndarray,
    soc_init: np.ndarray | None,
    *,
    goal_low: np.ndarray,
    goal_high: np.ndarray,
    charge_threshold: float = 0.40,
    discharge_threshold: float = 0.90,
    recovery: bool = False,
    strength: float = 0.12,
    th_mean: float | None = None,
) -> np.ndarray:
    """5 维市场/回收 prior（贴合 Modelica 厂站意图）。"""
    low = np.asarray(goal_low, dtype=np.float32)
    high = np.asarray(goal_high, dtype=np.float32)
    g = np.zeros(len(low), dtype=np.float32)
    sn = np.asarray(soc_now, dtype=np.float32).ravel()

    if recovery and soc_init is not None:
        si = np.asarray(soc_init, dtype=np.float32).ravel()
        if sn.size >= 1 and g.size > G_BAT:
            g[G_BAT] = float(si[0] - sn[0])
        if sn.size >= 2 and g.size > G_GAS:
            g[G_GAS] = float(si[1] - sn[1])
        if g.size > G_ARB:
            g[G_ARB] = 0.05
        if g.size > G_UTP:
            g[G_UTP] = 0.0
        return clip_goal(g, low, high)

    if buy_price is None:
        if g.size > G_ARB:
            g[G_ARB] = 0.3
        return clip_goal(g, low, high)

    s = float(strength)
    if buy_price <= charge_threshold:
        # 谷：充电、压火电、提高套利
        if g.size > G_BAT:
            g[G_BAT] = s
        if g.size > G_GAS:
            g[G_GAS] = 0.35 * s
        if g.size > G_UTP:
            g[G_UTP] = -0.12
        if g.size > G_ARB:
            g[G_ARB] = 0.75
        if g.size > G_TH and th_mean is not None and abs(g[G_GAS]) > 1e-6:
            g[G_TH] = float(np.clip(0.5 - float(th_mean), -0.08, 0.08))
    elif buy_price >= discharge_threshold:
        if g.size > G_BAT:
            g[G_BAT] = -s
        if g.size > G_GAS:
            g[G_GAS] = -0.30 * s
        if g.size > G_UTP:
            g[G_UTP] = -0.05
        if g.size > G_ARB:
            g[G_ARB] = 0.85
    else:
        if g.size > G_ARB:
            g[G_ARB] = 0.35
    return clip_goal(g, low, high)


def blend_goal_with_prior(
    goal: np.ndarray,
    prior: np.ndarray,
    *,
    prior_weight: float,
    goal_low: np.ndarray,
    goal_high: np.ndarray,
) -> np.ndarray:
    w = float(np.clip(prior_weight, 0.0, 1.0))
    g = (1.0 - w) * np.asarray(goal, dtype=np.float32) + w * np.asarray(prior, dtype=np.float32)
    return clip_goal(g, goal_low, goal_high)


def achieved_goal_from_cycle(
    intent0: np.ndarray,
    intent1: np.ndarray,
    mean_u_tp: float,
    mean_u_tp_hybrid: float,
    storage_throughput: float,
    *,
    goal_low: np.ndarray,
    goal_high: np.ndarray,
    thr_ref: float = 800.0,
) -> np.ndarray:
    """由 c 窗实际轨迹构造 hindsight 5 维 goal。"""
    low = np.asarray(goal_low, dtype=np.float32)
    high = np.asarray(goal_high, dtype=np.float32)
    g = np.zeros(len(low), dtype=np.float32)
    d = np.asarray(intent1, dtype=np.float32) - np.asarray(intent0, dtype=np.float32)
    n = min(3, d.size, g.size)
    g[:n] = d[:n]
    if g.size > G_UTP:
        g[G_UTP] = float(mean_u_tp - mean_u_tp_hybrid)
    if g.size > G_ARB:
        g[G_ARB] = float(np.clip(storage_throughput / max(thr_ref, 1.0), 0.0, 1.0))
    return clip_goal(g, low, high)
