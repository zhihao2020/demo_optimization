"""逆动力学 / hindsight 残差 MLE：预热 Δ_θ(s,g)，使 goal 能驱动 a−a_H。

不学正向世界模型；拟合
  β⊙tanh(Δ_θ(s_norm,g)) ≈ clip(a_exec − a_H(s), −β, β)
其中 g 为 hindsight 达成 intent 增量（5 维）。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from controllers.price_aware_rule import PriceAwareRuleController
from controllers.rule_based_controller import RuleBasedController
from envs.power_system_env import PowerSystemEnv
from training.hybrid_td3.train import annual_episode_start_seconds

from .agent import GHTD3Agent
from .goals import (
    G_ARB,
    G_UTP,
    clip_goal,
    plant_intent_vector,
)


def _achieved_goal_step(
    intent0: np.ndarray,
    intent1: np.ndarray,
    u_tp: float,
    u_tp_h: float,
    mag: float,
    mode: int,
    goal_low: np.ndarray,
    goal_high: np.ndarray,
) -> np.ndarray:
    g = np.zeros(len(goal_low), dtype=np.float32)
    d = np.asarray(intent1, dtype=np.float32) - np.asarray(intent0, dtype=np.float32)
    n = min(3, d.size, g.size)
    g[:n] = d[:n]
    if g.size > G_UTP:
        g[G_UTP] = float(u_tp - u_tp_h)
    if g.size > G_ARB:
        thr = abs(float(mag)) if int(mode) != 1 else 0.0
        g[G_ARB] = float(np.clip(thr, 0.0, 1.0))
    return clip_goal(g, goal_low, goal_high)


def collect_residual_mle_demos(
    env: PowerSystemEnv,
    agent: GHTD3Agent,
    *,
    n_windows: int = 4,
    seed: int = 0,
    cfg: dict[str, Any] | None = None,
    hybrid_noise: float = 0.12,
) -> dict[str, np.ndarray]:
    """混合 Hybrid(噪声) + 峰谷规则，构造 (s, g_ach, a_H, a_exec)。"""
    cfg = dict(cfg or {})
    if agent._hybrid_anchor is None:
        raise RuntimeError("residual MLE requires hybrid_anchor")

    rule = (
        PriceAwareRuleController(env)
        if getattr(env, "market_enabled", False)
        else RuleBasedController(env)
    )
    episode_steps = int(env.episode_steps)
    fmu_cfg = env.config["fmu"]
    windows = max(int(n_windows), 1)

    obs_l, goal_l = [], []
    a_h_tp, a_h_bat, a_h_mode, a_h_mag = [], [], [], []
    a_tp, a_bat, a_mode, a_mag = [], [], [], []
    tp_lo, tp_hi, bat_lo, bat_hi, masks = [], [], [], [], []

    for w in range(windows):
        start = annual_episode_start_seconds(fmu_cfg, episode_steps, w)
        obs, _ = env.reset(seed=seed + 17 + w, options={"start_time": start})
        use_rule = (w % 2) == 1
        for _ in range(episode_steps):
            try:
                feasible = env.get_feasible_action_spec()
            except Exception:
                break
            outs0 = env.last_outputs or {}
            intent0 = plant_intent_vector(outs0) if outs0 else np.zeros(3, np.float32)
            a_h = agent._hybrid_anchor.act_scalars(obs, feasible, deterministic=True)

            if use_rule:
                action = rule.predict(obs, deterministic=True)
            else:
                # Hybrid + 连续噪声 → 非零残差标签
                action = {
                    "u_tp": np.asarray(
                        [
                            float(
                                np.clip(
                                    a_h["u_tp"] + hybrid_noise * np.random.randn(),
                                    feasible.u_tp_low,
                                    feasible.u_tp_high,
                                )
                            )
                        ],
                        dtype=np.float32,
                    ),
                    "u_battery": np.asarray(
                        [
                            float(
                                np.clip(
                                    a_h["u_battery"] + hybrid_noise * np.random.randn(),
                                    feasible.u_battery_low,
                                    feasible.u_battery_high,
                                )
                            )
                        ],
                        dtype=np.float32,
                    ),
                    "caes_mode": int(a_h["caes_mode"]),
                    "caes_magnitude": np.asarray(
                        [
                            float(
                                np.clip(
                                    a_h["caes_magnitude"] + 0.5 * hybrid_noise * abs(np.random.randn()),
                                    0.0,
                                    1.0,
                                )
                            )
                        ],
                        dtype=np.float32,
                    ),
                }
                if int(action["caes_mode"]) == 1:
                    action["caes_magnitude"] = np.asarray([0.0], dtype=np.float32)

            next_obs, _, term, trunc, info = env.step(action)
            if not info.get("transition_valid"):
                break
            outs1 = info.get("observations") or env.last_outputs or {}
            intent1 = plant_intent_vector(outs1) if outs1 else intent0
            u_tp = float(np.asarray(action["u_tp"]).ravel()[0])
            u_bat = float(np.asarray(action["u_battery"]).ravel()[0])
            mode = int(action["caes_mode"])
            mag = float(np.asarray(action["caes_magnitude"]).ravel()[0])
            g = _achieved_goal_step(
                intent0,
                intent1,
                u_tp,
                float(a_h["u_tp"]),
                mag,
                mode,
                agent.goal_low,
                agent.goal_high,
            )
            # 混入少量市场 prior 扰动，拓宽 g 覆盖
            if np.random.rand() < 0.25:
                g = clip_goal(
                    g + 0.05 * np.random.randn(agent.goal_dim).astype(np.float32),
                    agent.goal_low,
                    agent.goal_high,
                )

            obs_l.append(np.asarray(obs, dtype=np.float32))
            goal_l.append(g)
            a_h_tp.append(float(a_h["u_tp"]))
            a_h_bat.append(float(a_h["u_battery"]))
            a_h_mode.append(int(a_h["caes_mode"]))
            a_h_mag.append(float(a_h["caes_magnitude"]))
            a_tp.append(u_tp)
            a_bat.append(u_bat)
            a_mode.append(mode)
            a_mag.append(mag)
            tp_lo.append(float(feasible.u_tp_low))
            tp_hi.append(float(feasible.u_tp_high))
            bat_lo.append(float(feasible.u_battery_low))
            bat_hi.append(float(feasible.u_battery_high))
            masks.append(feasible.mode_mask.as_bool_array())

            obs = next_obs
            if term or trunc:
                break

    if not obs_l:
        raise RuntimeError("residual MLE: no demos collected")
    return {
        "obs": np.stack(obs_l).astype(np.float32),
        "goal": np.stack(goal_l).astype(np.float32),
        "a_h_tp": np.asarray(a_h_tp, dtype=np.float32),
        "a_h_bat": np.asarray(a_h_bat, dtype=np.float32),
        "a_h_mode": np.asarray(a_h_mode, dtype=np.int64),
        "a_h_mag": np.asarray(a_h_mag, dtype=np.float32),
        "a_tp": np.asarray(a_tp, dtype=np.float32),
        "a_bat": np.asarray(a_bat, dtype=np.float32),
        "a_mode": np.asarray(a_mode, dtype=np.int64),
        "a_mag": np.asarray(a_mag, dtype=np.float32),
        "u_tp_low": np.asarray(tp_lo, dtype=np.float32),
        "u_tp_high": np.asarray(tp_hi, dtype=np.float32),
        "u_bat_low": np.asarray(bat_lo, dtype=np.float32),
        "u_bat_high": np.asarray(bat_hi, dtype=np.float32),
        "mode_mask": np.stack(masks).astype(bool),
    }


def mle_pretrain_residual(
    agent: GHTD3Agent,
    demos: dict[str, np.ndarray],
    *,
    epochs: int = 25,
    batch_size: int = 128,
    mode_weight: float = 0.0,
    fit_mode: bool = False,
) -> dict[str, float]:
    """MSE 拟合连续动作残差（默认不学 mode，避免 CAES 乱切）。"""
    device = agent.device
    n = int(demos["obs"].shape[0])
    if n < 8:
        return {"n": float(n), "epochs": 0.0, "final_loss": float("nan")}

    obs_t = torch.as_tensor(demos["obs"], device=device)
    goal_t = torch.as_tensor(demos["goal"], device=device)
    ah_tp = torch.as_tensor(demos["a_h_tp"], device=device)
    ah_bat = torch.as_tensor(demos["a_h_bat"], device=device)
    ah_mode = torch.as_tensor(demos["a_h_mode"], device=device, dtype=torch.long)
    ah_mag = torch.as_tensor(demos["a_h_mag"], device=device)
    a_tp = torch.as_tensor(demos["a_tp"], device=device)
    a_bat = torch.as_tensor(demos["a_bat"], device=device)
    a_mode = torch.as_tensor(demos["a_mode"], device=device, dtype=torch.long)
    a_mag = torch.as_tensor(demos["a_mag"], device=device)
    tp_lo = torch.as_tensor(demos["u_tp_low"], device=device)
    tp_hi = torch.as_tensor(demos["u_tp_high"], device=device)
    bat_lo = torch.as_tensor(demos["u_bat_low"], device=device)
    bat_hi = torch.as_tensor(demos["u_bat_high"], device=device)
    mask = torch.as_tensor(demos["mode_mask"], device=device)

    # 目标残差（动作空间，截断到 β）
    d_tp_star = (a_tp - ah_tp).clamp(-agent.beta_tp, agent.beta_tp)
    d_bat_star = (a_bat - ah_bat).clamp(-agent.beta_bat, agent.beta_bat)
    d_mag_star = (a_mag - ah_mag).clamp(-agent.beta_mag, agent.beta_mag)

    opt = agent.lo_actor_opt
    agent.lo_actor.train()
    last = 0.0
    steps = max(1, n // batch_size)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(steps):
            idx = perm[s * batch_size : (s + 1) * batch_size]
            if idx.numel() < 4:
                continue
            o = agent._prep_obs_low(obs_t[idx])
            g = goal_t[idx]
            bt, bb, bm = agent._betas_from_goal(g)
            out = agent.lo_actor.residual_compose(
                o,
                g,
                ah_tp[idx],
                ah_bat[idx],
                ah_mode[idx],
                ah_mag[idx],
                tp_lo[idx],
                tp_hi[idx],
                bat_lo[idx],
                bat_hi[idx],
                mask[idx],
                beta_tp=bt,
                beta_bat=bb,
                beta_mag=bm,
                mode_margin=agent.mode_margin,
                mode_override=bool(fit_mode and agent.mode_override),
                logit_clip=agent.logit_clip,
                soft_mode_for_grad=True,
                deterministic=True,
                explore_noise_std=0.0,
            )
            d_tp = out["u_tp"] - ah_tp[idx]
            d_bat = out["u_battery"] - ah_bat[idx]
            d_mag = out["caes_magnitude"] - ah_mag[idx]
            loss = (
                F.mse_loss(d_tp, d_tp_star[idx])
                + F.mse_loss(d_bat, d_bat_star[idx])
                + 0.5 * F.mse_loss(d_mag, d_mag_star[idx])
            )
            # 绝对动作锚定（连续）
            loss = loss + 0.25 * (
                F.mse_loss(out["u_tp"], a_tp[idx]) + F.mse_loss(out["u_battery"], a_bat[idx])
            )
            if fit_mode and mode_weight > 0:
                diff = (a_mode[idx] != ah_mode[idx]).float()
                if float(diff.mean()) > 0.01:
                    logp = F.log_softmax(out["logits_mode"], dim=-1)
                    ce = F.nll_loss(logp, a_mode[idx], reduction="none")
                    loss = loss + mode_weight * (ce * diff).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.lo_actor.parameters(), 5.0)
            opt.step()
            last = float(loss.item())

    agent.lo_actor_t.load_state_dict(agent.lo_actor.state_dict())
    return {
        "n": float(n),
        "epochs": float(epochs),
        "final_loss": last,
        "fit_mode": float(fit_mode),
    }
