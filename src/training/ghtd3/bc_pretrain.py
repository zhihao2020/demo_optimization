"""GHTD3 底层 BC：用峰谷规则演示 + 市场 prior/实际 ΔSoC 作为 goal。

对齐 Hybrid 的 BC→RL 成功路径，并在分层设定下为低层提供可跟踪目标。
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
    DEFAULT_SOC_KEYS,
    clip_goal,
    extract_soc,
    extract_soc_from_obs,
    market_conditioned_goal_prior,
    plant_intent_vector,
    G_BAT,
    G_GAS,
    G_TH,
    G_UTP,
    G_ARB,
)


def _inv_sigmoid_target(u: torch.Tensor, low: torch.Tensor, high: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    span = (high - low).clamp_min(1e-8)
    p = ((u - low) / span).clamp(eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


def collect_hierarchical_demos(
    env: PowerSystemEnv,
    agent: GHTD3Agent,
    *,
    n_windows: int = 4,
    seed: int = 0,
    price_aware: bool = True,
    cfg: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """采集规则轨迹；goal = 市场 prior 与实际一步 ΔSoC 的混合。"""
    cfg = dict(cfg or {})
    if price_aware and getattr(env, "market_enabled", False):
        ctrl = PriceAwareRuleController(env)
    else:
        ctrl = RuleBasedController(env)

    episode_steps = int(env.episode_steps)
    fmu_cfg = env.config["fmu"]
    windows = max(int(n_windows), 1)

    obs_l, goal_l = [], []
    u_tp_l, u_bat_l, u_caes_l = [], [], []
    tp_lo, tp_hi, bat_lo, bat_hi, masks = [], [], [], [], []

    for w in range(windows):
        start = annual_episode_start_seconds(fmu_cfg, episode_steps, w)
        obs, _ = env.reset(seed=seed + w, options={"start_time": start})
        for _ in range(episode_steps):
            try:
                feasible = env.get_feasible_action_spec()
            except Exception:
                break
            action = ctrl.predict(obs, deterministic=True)
            outs0 = env.last_outputs or {}
            intent0 = plant_intent_vector(outs0) if outs0 else extract_soc_from_obs(obs, 2)
            soc0 = extract_soc_from_obs(obs, 2)

            buy = None
            if getattr(env, "price_profile", None) is not None:
                try:
                    buy, _ = env.price_profile.prices_at(float(env.adapter.time))
                except Exception:
                    buy = None
            soc_init = None
            if env.initial_soc is not None:
                soc_init = extract_soc(env.initial_soc, DEFAULT_SOC_KEYS)
            rem = int(env.episode_steps - env.step_index)
            recovery = rem <= int(cfg.get("recovery_goal_horizon_steps", 36) or 0)
            th_mean = float(intent0[2]) if intent0.size > 2 else None
            prior = market_conditioned_goal_prior(
                buy,
                soc0,
                soc_init,
                goal_low=agent.goal_low,
                goal_high=agent.goal_high,
                charge_threshold=float(cfg.get("charge_threshold", 0.40)),
                discharge_threshold=float(cfg.get("discharge_threshold", 0.90)),
                recovery=recovery,
                strength=float(cfg.get("market_prior_strength", 0.14)),
                th_mean=th_mean,
            )

            next_obs, _, term, trunc, info = env.step(action)
            if not info.get("transition_valid", True):
                break
            outs = info.get("observations") or env.last_outputs or {}
            intent1 = plant_intent_vector(outs) if outs else intent0
            # 5 维 hindsight：Δintent + u_tp bias=0 + 弱 arb
            goal = np.zeros(agent.goal_dim, dtype=np.float32)
            d = intent1 - intent0
            n = min(3, d.size, goal.size)
            goal[:n] = d[:n]
            if goal.size > G_UTP:
                goal[G_UTP] = 0.0
            if goal.size > G_ARB:
                thr = abs(float(np.asarray(action.get("u_battery", 0)).reshape(-1)[0])) + abs(
                    float(np.asarray(action.get("u_caes", 0)).reshape(-1)[0])
                )
                goal[G_ARB] = float(np.clip(thr, 0.0, 1.0))
            goal = clip_goal(0.5 * goal + 0.5 * prior, agent.goal_low, agent.goal_high)

            obs_l.append(np.asarray(obs, dtype=np.float32).ravel())
            goal_l.append(goal.astype(np.float32))
            u_tp_l.append(float(np.asarray(action["u_tp"]).ravel()[0]))
            u_bat_l.append(float(np.asarray(action["u_battery"]).ravel()[0]))
            u_caes_l.append(float(np.asarray(action["u_caes"]).ravel()[0]))
            tp_lo.append(float(feasible.u_tp_low))
            tp_hi.append(float(feasible.u_tp_high))
            bat_lo.append(float(feasible.u_battery_low))
            bat_hi.append(float(feasible.u_battery_high))
            masks.append(feasible.mode_mask.as_bool_array().astype(np.bool_))

            obs = next_obs
            if term or trunc:
                break

    if not obs_l:
        raise RuntimeError("GHTD3 BC 演示采集失败")

    return {
        "obs": np.stack(obs_l).astype(np.float32),
        "goal": np.stack(goal_l).astype(np.float32),
        "u_tp": np.asarray(u_tp_l, dtype=np.float32),
        "u_battery": np.asarray(u_bat_l, dtype=np.float32),
        "u_caes": np.asarray(u_caes_l, dtype=np.float32),
        "u_tp_low": np.asarray(tp_lo, dtype=np.float32),
        "u_tp_high": np.asarray(tp_hi, dtype=np.float32),
        "u_bat_low": np.asarray(bat_lo, dtype=np.float32),
        "u_bat_high": np.asarray(bat_hi, dtype=np.float32),
        "mode_mask": np.stack(masks).astype(np.bool_),
    }


def behavior_clone_low_actor(
    agent: GHTD3Agent,
    demos: dict[str, np.ndarray],
    *,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> dict[str, float]:
    actor = agent.lo_actor
    device = agent.device
    actor.train()
    opt = torch.optim.Adam(actor.parameters(), lr=lr)

    n = int(demos["obs"].shape[0])
    idx = np.arange(n)
    history: list[float] = []

    obs_t = torch.as_tensor(demos["obs"], device=device)
    goal_t = torch.as_tensor(demos["goal"], device=device)
    u_tp_t = torch.as_tensor(demos["u_tp"], device=device)
    u_bat_t = torch.as_tensor(demos["u_battery"], device=device)
    u_caes_t = torch.as_tensor(demos["u_caes"], device=device)
    tp_lo = torch.as_tensor(demos["u_tp_low"], device=device)
    tp_hi = torch.as_tensor(demos["u_tp_high"], device=device)
    bat_lo = torch.as_tensor(demos["u_bat_low"], device=device)
    bat_hi = torch.as_tensor(demos["u_bat_high"], device=device)
    mask = torch.as_tensor(demos["mode_mask"], device=device)

    z_tp_tgt = _inv_sigmoid_target(u_tp_t, tp_lo, tp_hi)
    z_bat_tgt = _inv_sigmoid_target(u_bat_t, bat_lo, bat_hi)
    # u_caes already in [-1,1] legal bands; regress pre-tanh via atanh-ish soft target
    z_caes_tgt = torch.atanh(u_caes_t.clamp(-0.999, 0.999))

    for _ in range(epochs):
        np.random.shuffle(idx)
        epoch_loss = 0.0
        steps = 0
        for start in range(0, n, batch_size):
            b = idx[start : start + batch_size]
            out = actor.forward_logits(obs_t[b], goal_t[b])
            loss_tp = F.mse_loss(out["z_tp"], z_tp_tgt[b])
            loss_bat = F.mse_loss(out["z_bat"], z_bat_tgt[b])
            loss_caes = F.mse_loss(out["z_caes"], z_caes_tgt[b])
            loss = 2.0 * loss_tp + 8.0 * loss_bat + 2.0 * loss_caes
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
            opt.step()
            epoch_loss += float(loss.item())
            steps += 1
        history.append(epoch_loss / max(steps, 1))

    # 同步 target
    agent.lo_actor_t.load_state_dict(agent.lo_actor.state_dict())
    actor.eval()

    with torch.no_grad():
        pred = actor.act(
            obs_t, goal_t, tp_lo, tp_hi, bat_lo, bat_hi, mask, deterministic=True, explore_noise_std=0.0
        )
        mae_tp = float((pred["u_tp"] - u_tp_t).abs().mean().item())
        mae_bat = float((pred["u_battery"] - u_bat_t).abs().mean().item())
        mae_caes = float((pred["u_caes"] - u_caes_t).abs().mean().item())

    return {
        "n_demos": float(n),
        "epochs": float(epochs),
        "final_loss": float(history[-1]) if history else float("nan"),
        "mae_u_tp": mae_tp,
        "mae_u_battery": mae_bat,
        "mae_u_caes": mae_caes,
    }


def bc_pretrain_high_goals(
    agent: GHTD3Agent,
    demos: dict[str, np.ndarray],
    *,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 5e-4,
) -> dict[str, float]:
    """高层 BC：让 high actor 拟合演示中的 goal（市场 prior + 实际 ΔSoC）。"""
    actor = agent.hi_actor
    device = agent.device
    actor.train()
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    n = int(demos["obs"].shape[0])
    idx = np.arange(n)
    history: list[float] = []

    obs_t = torch.as_tensor(demos["obs"], device=device)
    goal_t = torch.as_tensor(demos["goal"], device=device)
    gl = agent._goal_low_t.unsqueeze(0).expand(batch_size, -1)
    gh = agent._goal_high_t.unsqueeze(0).expand(batch_size, -1)

    for _ in range(epochs):
        np.random.shuffle(idx)
        epoch_loss = 0.0
        steps = 0
        for start in range(0, n, batch_size):
            b = idx[start : start + batch_size]
            bs = len(b)
            pred = actor(obs_t[b], gl[:bs], gh[:bs])
            loss = F.mse_loss(pred, goal_t[b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
            opt.step()
            epoch_loss += float(loss.item())
            steps += 1
        history.append(epoch_loss / max(steps, 1))

    agent.hi_actor_t.load_state_dict(agent.hi_actor.state_dict())
    actor.eval()
    return {
        "n_demos": float(n),
        "epochs": float(epochs),
        "final_loss": float(history[-1]) if history else float("nan"),
    }
