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
G_WEAR = 2
G_THB = 3
G_CAES = 3
GOAL_NAMES = ("d_bat", "d_gas", "d_th", "u_tp_bias", "arb")
U_TP_MIN = 1.0 / 3.0


def goal_budget_layout(
    *,
    wear_budget: bool = False,
    caes_budget: bool = False,
    thermal_budget: bool = False,
    carbon_budget: bool = False,
) -> dict[str, int]:
    """Map budget names to goal indices after the two inventory dims."""
    idx: dict[str, int] = {}
    i = 2
    if wear_budget:
        idx["wear"] = i
        i += 1
    if caes_budget:
        idx["caes"] = i
        i += 1
    if thermal_budget:
        idx["thermal"] = i
        i += 1
    if carbon_budget:
        idx["carbon"] = i
    return idx


def default_goal_boxes(
    goal_dim: int = 2,
    *,
    wear_budget: bool = False,
    thermal_budget: bool = False,
    caes_budget: bool = False,
    carbon_budget: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Goal box. 2D inventory; then optional wear / caes / thermal / carbon budgets."""
    low5 = np.asarray([-0.30, -0.12, -0.10, -0.20, 0.0], dtype=np.float32)
    high5 = np.asarray([0.30, 0.12, 0.10, 0.20, 1.0], dtype=np.float32)
    d = int(goal_dim)
    if wear_budget or thermal_budget or caes_budget or carbon_budget:
        low = [-0.30, -0.12]
        high = [0.30, 0.12]
        layout = goal_budget_layout(
            wear_budget=wear_budget,
            caes_budget=caes_budget,
            thermal_budget=thermal_budget,
            carbon_budget=carbon_budget,
        )
        if "wear" in layout:
            low.append(0.0)
            high.append(0.50)
        if "caes" in layout:
            low.append(0.0)
            high.append(1.0)
        if "thermal" in layout:
            low.append(0.0)
            high.append(1.0)
        if "carbon" in layout:
            # 块内允许的火电额外负荷占比（配速）；与 thermal 同形
            low.append(0.0)
            high.append(1.0)
        low_a = np.asarray(low, dtype=np.float32)
        high_a = np.asarray(high, dtype=np.float32)
        if d > low_a.size:
            extra = d - low_a.size
            low_a = np.concatenate([low_a, np.zeros(extra, dtype=np.float32)])
            high_a = np.concatenate([high_a, np.ones(extra, dtype=np.float32)])
        return low_a[:d].copy(), high_a[:d].copy()
    if d <= 2:
        return low5[:2].copy(), high5[:2].copy()
    if d >= 5:
        return low5.copy(), high5.copy()
    return low5[:d].copy(), high5[:d].copy()


def battery_soc_discharge(intent_t: np.ndarray, intent_tp1: np.ndarray) -> float:
    """Battery SoC drop over one hour (0 if charging or idle)."""
    it = np.asarray(intent_t, dtype=np.float64).ravel()
    ip = np.asarray(intent_tp1, dtype=np.float64).ravel()
    if it.size < 1 or ip.size < 1:
        return 0.0
    return float(max(0.0, it[0] - ip[0]))


# Match src/config/device_params.yaml + FeasibilityOracle SoC step.
DEFAULT_BAT_P_CAP_W = 1.0e8
DEFAULT_BAT_E_CAP_J = 1.8e12
DEFAULT_BAT_ETA = 0.85
DEFAULT_DT_S = 3600.0


def max_discharge_u(
    remain_soc: float,
    *,
    p_cap_w: float = DEFAULT_BAT_P_CAP_W,
    e_cap_j: float = DEFAULT_BAT_E_CAP_J,
    eta: float = DEFAULT_BAT_ETA,
    dt_s: float = DEFAULT_DT_S,
) -> float:
    """Largest |u_battery| whose one-hour discharge SoC drop stays within remain_soc."""
    remain = max(float(remain_soc), 0.0)
    denom = float(p_cap_w) * float(dt_s)
    if denom <= 0.0 or remain <= 0.0:
        return 0.0
    return float(remain * float(e_cap_j) * float(eta) / denom)


def clip_discharge_to_budget(
    u_battery: float,
    remain_soc: float,
    *,
    p_cap_w: float = DEFAULT_BAT_P_CAP_W,
    e_cap_j: float = DEFAULT_BAT_E_CAP_J,
    eta: float = DEFAULT_BAT_ETA,
    dt_s: float = DEFAULT_DT_S,
) -> float:
    """Charge unrestricted; discharge cannot exceed remaining SoC quota."""
    u = float(u_battery)
    if u >= 0.0:
        return u
    cap = max_discharge_u(remain_soc, p_cap_w=p_cap_w, e_cap_j=e_cap_j, eta=eta, dt_s=dt_s)
    return float(max(u, -cap))


def clip_thermal_to_budget(
    u_tp: float,
    remain_frac: float,
    *,
    c: int = 8,
    u_min: float = U_TP_MIN,
    u_max: float = 1.0,
) -> float:
    """Cap extra thermal load so this hour does not exceed remaining block fraction."""
    span = max(float(c) * (1.0 - float(u_min)), 1e-6)
    allowed = float(u_min) + max(float(remain_frac), 0.0) * span
    hi = min(float(u_max), allowed)
    return float(min(max(float(u_tp), float(u_min)), hi))


def caes_on_frac(u_caes: float, *, c: int = 8) -> float:
    """One on-hour as a fraction of the block (0 if idle)."""
    return 0.0 if abs(float(u_caes)) <= 1e-6 else 1.0 / max(int(c), 1)


def enforce_budget_on_action(
    action: dict,
    goal: np.ndarray,
    *,
    wear_budget: bool = False,
    thermal_budget: bool = False,
    caes_budget: bool = False,
    carbon_budget: bool = False,
    wear_enforce: bool = True,
    thermal_enforce: bool = True,
    caes_enforce: bool = True,
    carbon_enforce: bool = True,
    c: int = 8,
    layout: dict[str, int] | None = None,
    p_cap_w: float = DEFAULT_BAT_P_CAP_W,
    e_cap_j: float = DEFAULT_BAT_E_CAP_J,
    eta: float = DEFAULT_BAT_ETA,
    dt_s: float = DEFAULT_DT_S,
) -> dict:
    """Rewrite device commands so issued block quotas cannot be exceeded."""
    out = dict(action)
    g = np.asarray(goal, dtype=np.float32).ravel()
    lay = layout or goal_budget_layout(
        wear_budget=wear_budget,
        caes_budget=caes_budget,
        thermal_budget=thermal_budget,
        carbon_budget=carbon_budget,
    )
    iw = lay.get("wear", G_WEAR)
    ic = lay.get("caes", G_CAES)
    ith = lay.get("thermal", G_THB)
    icarb = lay.get("carbon", G_THB)
    if wear_budget and wear_enforce and g.size > iw:
        try:
            u_b = float(np.asarray(out.get("u_battery", 0.0)).reshape(-1)[0])
        except Exception:
            u_b = 0.0
        u_b = clip_discharge_to_budget(
            u_b, float(g[iw]), p_cap_w=p_cap_w, e_cap_j=e_cap_j, eta=eta, dt_s=dt_s
        )
        cur = out.get("u_battery")
        if isinstance(cur, np.ndarray):
            out["u_battery"] = np.asarray([u_b], dtype=np.float32)
        else:
            out["u_battery"] = u_b
    if caes_budget and caes_enforce and g.size > ic and float(g[ic]) <= 1e-8:
        cur = out.get("u_caes")
        if isinstance(cur, np.ndarray):
            out["u_caes"] = np.asarray([0.0], dtype=np.float32)
        else:
            out["u_caes"] = 0.0
    # thermal / carbon 均钳制 u_tp；同时开启时取更紧上限
    u_tp_cap = None
    if thermal_budget and thermal_enforce and g.size > ith:
        u_tp_cap = float(g[ith]) if u_tp_cap is None else min(u_tp_cap, float(g[ith]))
    if carbon_budget and carbon_enforce and g.size > icarb:
        u_tp_cap = float(g[icarb]) if u_tp_cap is None else min(u_tp_cap, float(g[icarb]))
    if u_tp_cap is not None:
        try:
            u_tp = float(np.asarray(out.get("u_tp", U_TP_MIN)).reshape(-1)[0])
        except Exception:
            u_tp = U_TP_MIN
        u_tp = clip_thermal_to_budget(u_tp, float(u_tp_cap), c=c)
        cur = out.get("u_tp")
        if isinstance(cur, np.ndarray):
            out["u_tp"] = np.asarray([u_tp], dtype=np.float32)
        else:
            out["u_tp"] = u_tp
    return out


def thermal_extra_frac(u_tp: float, *, c: int = 8, u_min: float = U_TP_MIN) -> float:
    """Extra thermal load this hour as a fraction of the block extra budget."""
    span = max(float(c) * (1.0 - float(u_min)), 1e-6)
    return float(max(0.0, float(u_tp) - float(u_min)) / span)


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
    *,
    wear_used: float = 0.0,
    wear_budget: bool = False,
    thermal_used: float = 0.0,
    thermal_budget: bool = False,
    caes_used: float = 0.0,
    caes_budget: bool = False,
    carbon_used: float = 0.0,
    carbon_budget: bool = False,
    layout: dict[str, int] | None = None,
) -> np.ndarray:
    """Residual inventory goal s+g-s' on bat/gas.

    Budget dims count down. Legacy 3rd dim is thermal residual only when no
    budget flags are set.
    """
    g = np.asarray(goal_t, dtype=np.float32).copy()
    it = np.asarray(intent_t, dtype=np.float32).ravel()
    ip = np.asarray(intent_tp1, dtype=np.float32).ravel()
    n_inv = min(g.size, it.size, ip.size, 2)
    if n_inv > 0:
        g[:n_inv] = it[:n_inv] + g[:n_inv] - ip[:n_inv]
    lay = layout or goal_budget_layout(
        wear_budget=wear_budget,
        caes_budget=caes_budget,
        thermal_budget=thermal_budget,
        carbon_budget=carbon_budget,
    )
    if wear_budget and "wear" in lay and g.size > lay["wear"]:
        g[lay["wear"]] = max(0.0, float(g[lay["wear"]]) - float(wear_used))
    if caes_budget and "caes" in lay and g.size > lay["caes"]:
        g[lay["caes"]] = max(0.0, float(g[lay["caes"]]) - float(caes_used))
    if thermal_budget and "thermal" in lay and g.size > lay["thermal"]:
        g[lay["thermal"]] = max(0.0, float(g[lay["thermal"]]) - float(thermal_used))
    if carbon_budget and "carbon" in lay and g.size > lay["carbon"]:
        g[lay["carbon"]] = max(0.0, float(g[lay["carbon"]]) - float(carbon_used))
    elif (
        (not wear_budget)
        and (not thermal_budget)
        and (not caes_budget)
        and (not carbon_budget)
        and g.size > 2
    ):
        n = min(g.size, it.size, ip.size, 3)
        if n > 2:
            g[2:n] = it[2:n] + g[2:n] - ip[2:n]
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
    wear_used: float | None = None,
    thermal_used: float | None = None,
    caes_used: float | None = None,
    layout: dict[str, int] | None = None,
) -> np.ndarray:
    """Hindsight goal from a high-level window: inventory Δ plus used budgets."""
    low = np.asarray(goal_low, dtype=np.float32)
    high = np.asarray(goal_high, dtype=np.float32)
    g = np.zeros(len(low), dtype=np.float32)
    d = np.asarray(intent1, dtype=np.float32) - np.asarray(intent0, dtype=np.float32)
    n_inv = min(d.size, g.size, 2)
    g[:n_inv] = d[:n_inv]
    lay = layout or {}
    iw = lay.get("wear", G_WEAR if g.size > G_WEAR and g.size < 5 else None)
    if iw is not None and g.size > iw:
        if wear_used is not None:
            g[iw] = float(max(0.0, wear_used))
        else:
            i0 = np.asarray(intent0, dtype=np.float32).ravel()
            i1 = np.asarray(intent1, dtype=np.float32).ravel()
            if i0.size >= 1 and i1.size >= 1:
                g[iw] = float(max(0.0, i0[0] - i1[0]))
    ic = lay.get("caes")
    if ic is not None and g.size > ic and caes_used is not None:
        g[ic] = float(max(0.0, min(1.0, caes_used)))
    ith = lay.get("thermal", G_THB if "thermal" in lay or (g.size > G_THB and thermal_used is not None) else None)
    if ith is not None and g.size > ith and thermal_used is not None:
        g[ith] = float(max(0.0, min(1.0, thermal_used)))
    if not lay and g.size > G_UTP:
        g[G_UTP] = float(mean_u_tp - mean_u_tp_hybrid)
    if not lay and g.size > G_ARB:
        g[G_ARB] = float(np.clip(storage_throughput / max(float(thr_ref), 1.0), 0.0, 1.0))
    return clip_goal(g, low, high)
