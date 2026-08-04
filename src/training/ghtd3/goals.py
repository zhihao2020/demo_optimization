"""分层 goal：SoC 增量、转移与内在奖励。"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# 与 env_config observations 前两项一致
DEFAULT_SOC_KEYS = ("battery_soc", "caes_gas_soc")


def extract_soc(outputs: dict[str, float], keys: Sequence[str] = DEFAULT_SOC_KEYS) -> np.ndarray:
    return np.asarray([float(outputs[k]) for k in keys], dtype=np.float32)


def extract_soc_from_obs(obs: np.ndarray, n: int = 2) -> np.ndarray:
    """物理观测前 n 维为 SoC（见 ObservationBuilder 顺序）。"""
    o = np.asarray(obs, dtype=np.float32).ravel()
    return o[:n].copy()


def clip_goal(goal: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(np.asarray(goal, dtype=np.float32), low), high)


def goal_transition(
    soc_t: np.ndarray,
    goal_t: np.ndarray,
    soc_tp1: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """论文式 (31)：g' = s + g - s'（在 SoC 子空间）。"""
    g_next = np.asarray(soc_t, dtype=np.float32) + np.asarray(goal_t, dtype=np.float32) - np.asarray(
        soc_tp1, dtype=np.float32
    )
    return clip_goal(g_next, low, high)


def intrinsic_reward(
    soc_t: np.ndarray,
    goal_t: np.ndarray,
    soc_tp1: np.ndarray,
    r_ext: float,
    alpha: float,
) -> tuple[float, dict[str, float]]:
    """论文式 (30)：跟踪 residual + α * 外在奖励。"""
    residual = np.asarray(soc_t, dtype=np.float64) + np.asarray(goal_t, dtype=np.float64) - np.asarray(
        soc_tp1, dtype=np.float64
    )
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
) -> np.ndarray:
    """市场条件目标先验（创新）：低价充 / 高价放；回收段指向初始 SoC。

    对齐 Ochoa 多时间尺度“上层做能量计划”的思想，但动作是 SoC 增量 goal（Cui GHTD3），
    并用外生电价（price-taker）调制。
    """
    g = np.zeros(len(goal_low), dtype=np.float32)
    if recovery and soc_init is not None:
        # 上层直接下发“回到初始 SoC”的跨时段计划
        g = np.asarray(soc_init, dtype=np.float32) - np.asarray(soc_now, dtype=np.float32)
        return clip_goal(g, goal_low, goal_high)
    if buy_price is None:
        return clip_goal(g, goal_low, goal_high)
    # 主调 battery；CAES gas 弱耦合（避免过度 CAES 拖垮热/冷罐 SOC）
    if buy_price <= charge_threshold:
        g[0] = float(strength)
        if g.size > 1:
            g[1] = float(0.25 * strength)
    elif buy_price >= discharge_threshold:
        g[0] = float(-strength)
        if g.size > 1:
            g[1] = float(-0.2 * strength)
    return clip_goal(g, goal_low, goal_high)


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
