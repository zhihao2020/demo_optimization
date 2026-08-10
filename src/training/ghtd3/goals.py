"""Inventory goals for hierarchical HMSD: default 2D [Δbat, Δgas].

Optional higher-dimensional boxes are still supported via ``goal_dim`` in YAML
(legacy ablation only). Mainline training uses 2D + plain historical HER.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Energy inventory SoC keys (observation / terminal checks)
ENERGY_SOC_KEYS = ("battery_soc", "caes_gas_soc")
PROCESS_SOC_KEYS = ("caes_hot_soc", "caes_cold_soc")
ALL_SOC_KEYS = ENERGY_SOC_KEYS + PROCESS_SOC_KEYS
DEFAULT_SOC_KEYS = ENERGY_SOC_KEYS

# Indices when goal_dim > 2 (legacy configs only)
G_BAT, G_GAS, G_TH, G_UTP, G_ARB = 0, 1, 2, 3, 4
GOAL_NAMES = ("d_bat", "d_gas", "d_th", "u_tp_bias", "arb")


def default_goal_boxes(goal_dim: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Goal box. Default 2D [Δbat, Δgas]; longer boxes for legacy YAML only."""
    low5 = np.asarray([-0.30, -0.12, -0.10, -0.20, 0.0], dtype=np.float32)
    high5 = np.asarray([0.30, 0.12, 0.10, 0.20, 1.0], dtype=np.float32)
    d = int(goal_dim)
    if d <= 2:
        return low5[:2].copy(), high5[:2].copy()
    if d >= 5:
        return low5.copy(), high5.copy()
    return low5[:d].copy(), high5[:d].copy()


def extract_soc(outputs: dict[str, float], keys: Sequence[str] = ENERGY_SOC_KEYS) -> np.ndarray:
    return np.asarray([float(outputs[k]) for k in keys], dtype=np.float32)


def extract_soc_from_obs(obs: np.ndarray, n: int = 2) -> np.ndarray:
    """First n dims of physical obs are SoC (ObservationBuilder order)."""
    o = np.asarray(obs, dtype=np.float32).ravel()
    return o[:n].copy()


def extract_plant_state(outputs: dict[str, float]) -> dict[str, float]:
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
    """Plant inventory state used for residual goal tracking: [bat, gas, th_mean]."""
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
    """Residual goal: s + g - s'. Updates first min(goal, intent, 3) inventory dims."""
    g = np.asarray(goal_t, dtype=np.float32).copy()
    it = np.asarray(intent_t, dtype=np.float32).ravel()
    ip = np.asarray(intent_tp1, dtype=np.float32).ravel()
    n = min(g.size, it.size, ip.size, 3)
    if n > 0:
        g[:n] = it[:n] + g[:n] - ip[:n]
    return clip_goal(g, low, high)


def goal_transition(
    soc_t: np.ndarray,
    goal_t: np.ndarray,
    soc_tp1: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """SoC residual transition (energy dims only)."""
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
    """Intrinsic reward: inventory residual tracking + α * extrinsic.

    Mainline (goal_dim=2): tracks bat/gas only.
    If goal_dim > G_UTP (legacy), also shapes u_tp bias.
    """
    g = np.asarray(goal_t, dtype=np.float64).ravel()
    it = np.asarray(intent_t, dtype=np.float64).ravel()
    ip = np.asarray(intent_tp1, dtype=np.float64).ravel()
    w = np.asarray(weights if weights is not None else (1.5, 1.2), dtype=np.float64)
    n = min(it.size, ip.size, g.size, 3)
    e = it[:n] + g[:n] - ip[:n]
    w_e = w[:n] if w.size >= n else np.pad(w, (0, n - w.size), constant_values=1.0)
    track_sq = float(np.sum(w_e * e * e))
    e_u = 0.0
    u_tgt = float(u_tp)
    if g.size > G_UTP:
        g_u = float(g[G_UTP])
        base_u = float(u_tp_hybrid)
        if relax_thermal_floor:
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


def actual_delta_soc(soc_t: np.ndarray, soc_tp1: np.ndarray) -> np.ndarray:
    return np.asarray(soc_tp1, dtype=np.float32) - np.asarray(soc_t, dtype=np.float32)


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
    """Optional market/recovery goal prior (off by default; enable ``market_goal_prior``).

    Fills inventory dims present in the goal box (2D mainline: bat/gas only).
    """
    _ = th_mean  # reserved for legacy >2D boxes
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
        return clip_goal(g, low, high)

    if buy_price is None:
        return clip_goal(g, low, high)

    s = float(strength)
    if buy_price <= charge_threshold:
        if g.size > G_BAT:
            g[G_BAT] = s
        if g.size > G_GAS:
            g[G_GAS] = 0.35 * s
    elif buy_price >= discharge_threshold:
        if g.size > G_BAT:
            g[G_BAT] = -s
        if g.size > G_GAS:
            g[G_GAS] = -0.30 * s
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
    mean_u_tp: float = 0.0,
    mean_u_tp_hybrid: float = 0.0,
    storage_throughput: float = 0.0,
    *,
    goal_low: np.ndarray,
    goal_high: np.ndarray,
    thr_ref: float = 800.0,
) -> np.ndarray:
    """Hindsight goal from a high-level window: inventory Δ (mainline 2D).

    Extra args kept for call-site compatibility; only used when goal_dim > 2.
    """
    low = np.asarray(goal_low, dtype=np.float32)
    high = np.asarray(goal_high, dtype=np.float32)
    g = np.zeros(len(low), dtype=np.float32)
    d = np.asarray(intent1, dtype=np.float32) - np.asarray(intent0, dtype=np.float32)
    n = min(d.size, g.size, 3)
    g[:n] = d[:n]
    # Legacy >2D: fill optional bias / arb if present
    if g.size > G_UTP:
        g[G_UTP] = float(mean_u_tp - mean_u_tp_hybrid)
    if g.size > G_ARB:
        g[G_ARB] = float(np.clip(storage_throughput / max(float(thr_ref), 1.0), 0.0, 1.0))
    return clip_goal(g, low, high)
