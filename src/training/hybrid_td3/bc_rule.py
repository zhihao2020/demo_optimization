"""规则策略行为克隆（BC）预热。

采集 RuleBasedController 在全年周窗上的演示，监督训练 HybridActor，
使策略先贴合强安全/经济基线，再交给 TD3 微调。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from controllers.price_aware_rule import PriceAwareRuleController
from controllers.rule_based_controller import RuleBasedController
from envs.power_system_env import PowerSystemEnv
from training.hybrid_td3.actor import HybridActor
from training.hybrid_td3.algorithm import HybridTD3
from training.hybrid_td3.train import annual_episode_start_seconds


def _make_rule_controller(env: PowerSystemEnv, *, price_aware: bool = True):
    if price_aware and getattr(env, "market_enabled", False):
        return PriceAwareRuleController(env)
    return RuleBasedController(env)


def collect_rule_demos(
    env: PowerSystemEnv,
    *,
    n_windows: int | None = None,
    seed: int = 0,
    price_aware: bool = True,
) -> dict[str, np.ndarray]:
    """在多个 168h 周窗上滚动规则策略，返回批量化演示。"""
    ctrl = _make_rule_controller(env, price_aware=price_aware)
    episode_steps = int(env.episode_steps)
    fmu_cfg = env.config["fmu"]
    annual_h = int(fmu_cfg.get("annual_horizon_hours") or episode_steps)
    windows = int(np.ceil(annual_h / episode_steps))
    if n_windows is not None:
        windows = min(windows, int(n_windows))

    obs_list: list[np.ndarray] = []
    u_tp_list: list[float] = []
    u_bat_list: list[float] = []
    u_caes_list: list[float] = []
    tp_lo: list[float] = []
    tp_hi: list[float] = []
    bat_lo: list[float] = []
    bat_hi: list[float] = []
    masks: list[np.ndarray] = []

    for w in range(windows):
        start = annual_episode_start_seconds(fmu_cfg, episode_steps, w)
        obs, _ = env.reset(seed=seed + w, options={"start_time": start})
        for _ in range(episode_steps):
            try:
                feasible = env.get_feasible_action_spec()
            except Exception:
                break
            action = ctrl.predict(obs, deterministic=True)
            obs_list.append(np.asarray(obs, dtype=np.float32).ravel())
            u_tp_list.append(float(np.asarray(action["u_tp"]).ravel()[0]))
            u_bat_list.append(float(np.asarray(action["u_battery"]).ravel()[0]))
            u_caes_list.append(float(np.asarray(action["u_caes"]).ravel()[0]))
            tp_lo.append(float(feasible.u_tp_low))
            tp_hi.append(float(feasible.u_tp_high))
            bat_lo.append(float(feasible.u_battery_low))
            bat_hi.append(float(feasible.u_battery_high))
            masks.append(feasible.mode_mask.as_bool_array().astype(np.bool_))

            obs, _, term, trunc, info = env.step(action)
            if not info.get("transition_valid", True) or term or trunc:
                break

    if not obs_list:
        raise RuntimeError("规则演示采集失败：无有效样本")

    return {
        "obs": np.stack(obs_list).astype(np.float32),
        "u_tp": np.asarray(u_tp_list, dtype=np.float32),
        "u_battery": np.asarray(u_bat_list, dtype=np.float32),
        "u_caes": np.asarray(u_caes_list, dtype=np.float32),
        "u_tp_low": np.asarray(tp_lo, dtype=np.float32),
        "u_tp_high": np.asarray(tp_hi, dtype=np.float32),
        "u_bat_low": np.asarray(bat_lo, dtype=np.float32),
        "u_bat_high": np.asarray(bat_hi, dtype=np.float32),
        "mode_mask": np.stack(masks).astype(np.bool_),
    }


def _inv_sigmoid_target(u: torch.Tensor, low: torch.Tensor, high: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """将有界动作反解为 actor 的 pre-sigmoid logit 目标。"""
    span = (high - low).clamp_min(1e-8)
    p = ((u - low) / span).clamp(eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


def behavior_clone_actor(
    actor: HybridActor,
    demos: dict[str, np.ndarray],
    *,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: torch.device | str | None = None,
    mode_weight: float = 2.0,
    mag_weight: float = 0.25,
    tp_weight: float = 2.0,
    bat_weight: float = 8.0,
) -> dict[str, float]:
    """监督学习：火电/电池 logit MSE；压空为模式 CE + 幅值 MSE（参数化）或 z_caes MSE。"""
    from actions.caes_u import mag_from_u_torch, mask_mode_logits, mode_index_from_u_torch

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    actor = actor.to(device)
    actor.train()
    opt = torch.optim.Adam(actor.parameters(), lr=lr)

    n = int(demos["obs"].shape[0])
    idx = np.arange(n)
    history: list[float] = []

    obs_t = torch.as_tensor(demos["obs"], device=device)
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
    z_caes_tgt = torch.atanh(u_caes_t.clamp(-0.999, 0.999))

    for _ in range(epochs):
        np.random.shuffle(idx)
        epoch_loss = 0.0
        steps = 0
        for start in range(0, n, batch_size):
            b = idx[start : start + batch_size]
            logits_out = actor.forward_logits(obs_t[b])
            loss_tp = F.mse_loss(logits_out["z_tp"], z_tp_tgt[b])
            loss_bat = F.mse_loss(logits_out["z_bat"], z_bat_tgt[b])
            if getattr(actor, "parameterized_caes", True):
                mode_logits = mask_mode_logits(logits_out["mode_logits"], mask[b])
                mode_tgt = mode_index_from_u_torch(u_caes_t[b])
                mag_tgt = mag_from_u_torch(u_caes_t[b])
                loss_mode = F.cross_entropy(mode_logits, mode_tgt)
                loss_mag = F.mse_loss(torch.sigmoid(logits_out["z_mag"]), mag_tgt)
                loss_caes = mode_weight * loss_mode + mag_weight * loss_mag
            else:
                loss_caes = 2.0 * F.mse_loss(logits_out["z_caes"], z_caes_tgt[b])
            loss = tp_weight * loss_tp + bat_weight * loss_bat + loss_caes
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
            opt.step()
            epoch_loss += float(loss.item())
            steps += 1
        history.append(epoch_loss / max(steps, 1))

    actor.eval()
    with torch.no_grad():
        pred = actor.act(obs_t, tp_lo, tp_hi, bat_lo, bat_hi, mask, deterministic=True, explore_noise_std=0.0)
        mae_tp = float((pred["u_tp"] - u_tp_t).abs().mean().item())
        mae_bat = float((pred["u_battery"] - u_bat_t).abs().mean().item())
        mae_caes = float((pred["u_caes"] - u_caes_t).abs().mean().item())

    return {
        "n_demos": float(n),
        "epochs": float(epochs),
        "final_loss": float(history[-1]) if history else float("nan"),
        "mean_loss": float(np.mean(history)) if history else float("nan"),
        "loss_curve_tail": float(np.mean(history[-5:])) if history else float("nan"),
        "mae_u_tp": mae_tp,
        "mae_u_battery": mae_bat,
        "mae_u_caes": mae_caes,
    }


def run_rule_bc_pretrain(
    run_dir: str | Path = "runs/rule_bc_pretrain",
    *,
    seed: int = 0,
    n_windows: int | None = None,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    forecast_enabled: bool | None = None,
    price_aware: bool = True,
) -> dict[str, Any]:
    """采集规则演示 → BC → 保存可被 TD3 --resume 加载的 checkpoint。"""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "train").mkdir(exist_ok=True)

    env = PowerSystemEnv(run_id=run_dir.name, forecast_enabled=forecast_enabled)
    try:
        demos = collect_rule_demos(env, n_windows=n_windows, seed=seed, price_aware=price_aware)
        obs_dim = int(np.prod(env.observation_space.shape))
        agent = HybridTD3(obs_dim=obs_dim, explore_noise=0.05)
        metrics = behavior_clone_actor(
            agent.actor,
            demos,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=agent.device,
            bat_weight=10.0,
            tp_weight=3.0,
            mode_weight=2.0,
        )
        agent.actor_target.load_state_dict(agent.actor.state_dict())
        # Critic 未训练：保存时同步空 critic，续训时应 --reset-critic
        ckpt = run_dir / "checkpoints" / "hybrid_givesafe_td3.pt"
        agent.save(ckpt)

        # 快速一致性检查：在当前 reset 状态上动作是否接近规则
        obs, _ = env.reset(seed=seed, options={"start_time": 0.0})
        rule = _make_rule_controller(env, price_aware=price_aware).predict(obs)
        feasible = env.get_feasible_action_spec()
        pred = agent.select_action(obs, feasible, deterministic=True)
        match = {
            "rule_u_tp": float(np.asarray(rule["u_tp"]).ravel()[0]),
            "bc_u_tp": float(np.asarray(pred["u_tp"]).ravel()[0]),
            "rule_u_battery": float(np.asarray(rule["u_battery"]).ravel()[0]),
            "bc_u_battery": float(np.asarray(pred["u_battery"]).ravel()[0]),
            "rule_u_caes": float(np.asarray(rule["u_caes"]).ravel()[0]),
            "bc_u_caes": float(np.asarray(pred["u_caes"]).ravel()[0]),
        }
    finally:
        env.close()

    result = {
        "status": "completed",
        "checkpoint": str(ckpt.resolve()),
        "bc_metrics": metrics,
        "action_match_sample": match,
        "n_demo_windows": n_windows,
        "price_aware_rule": bool(price_aware),
    }
    (run_dir / "train" / "bc_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    # 持久化演示规模元数据
    np.savez_compressed(
        run_dir / "train" / "rule_demos_meta.npz",
        n=np.asarray([metrics["n_demos"]]),
        u_tp_mean=np.asarray([float(demos["u_tp"].mean())]),
        u_caes_mean=float(np.mean(demos["u_caes"])),
    )
    return result
