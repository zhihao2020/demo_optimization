"""Hybrid-PPO：随机混合 Actor + V(s) + clipped surrogate。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from training.hybrid_common.stochastic_actor import HybridStochasticActor

from .rollout import RolloutBuffer


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class HybridPPO:
    def __init__(
        self,
        obs_dim: int,
        *,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        lr: float = 3e-4,
        update_epochs: int = 4,
        minibatch_size: int = 64,
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.actor = HybridStochasticActor(obs_dim).to(self.device)
        self.critic = ValueNet(obs_dim).to(self.device)
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )
        # 较小初始探索方差，降低早期 log_prob / ratio 爆炸
        for mod in (
            self.actor.tp_log_std,
            self.actor.bat_log_std,
            self.actor.d_mag_log_std,
            self.actor.c_mag_log_std,
        ):
            nn.init.constant_(mod.bias, -1.0)
        self.total_it = 0
        self.last_metrics: dict[str, float] = {}

    def select_action(self, obs, feasible, deterministic: bool = False) -> dict:
        out = self.actor.act_numpy(obs, feasible, deterministic=deterministic, device=self.device)
        return {
            "u_tp": out["u_tp"],
            "u_battery": out["u_battery"],
            "caes_mode": out["caes_mode"],
            "caes_magnitude": out["caes_magnitude"],
        }

    def select_action_with_stats(self, obs, feasible, deterministic: bool = False) -> dict:
        """训练采集：附带 log_prob 与 V(s)。"""
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
            mask = torch.as_tensor(
                feasible.mode_mask.as_bool_array(), dtype=torch.bool, device=self.device
            ).view(1, 3)
            act = self.actor.forward_action(
                o,
                torch.tensor([feasible.u_tp_low], device=self.device),
                torch.tensor([feasible.u_tp_high], device=self.device),
                torch.tensor([feasible.u_battery_low], device=self.device),
                torch.tensor([feasible.u_battery_high], device=self.device),
                mask,
                deterministic=deterministic,
            )
            value = float(self.critic(o)[0].cpu())
        return {
            "u_tp": np.asarray([float(act["u_tp"][0].cpu())], dtype=np.float32),
            "u_battery": np.asarray([float(act["u_battery"][0].cpu())], dtype=np.float32),
            "caes_mode": int(act["caes_mode"][0].cpu()),
            "caes_magnitude": np.asarray([float(act["caes_magnitude"][0].cpu())], dtype=np.float32),
            "log_prob": float(act["log_prob"][0].cpu()),
            "value": value,
        }

    def value_numpy(self, obs) -> float:
        self.critic.eval()
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
            return float(self.critic(o)[0].cpu())

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        if len(buffer) == 0:
            return {}
        self.total_it += 1
        adv = buffer.advantage[: buffer.pos]
        if not np.isfinite(adv).all():
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        buffer.advantage[: buffer.pos] = adv

        metrics_acc = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
        }
        n_updates = 0
        for _ in range(self.update_epochs):
            for batch in buffer.get_batches(self.minibatch_size):
                obs = torch.as_tensor(batch["obs"], device=self.device)
                obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
                evaluated = self.actor.evaluate_actions(
                    obs,
                    torch.as_tensor(batch["u_tp"], device=self.device),
                    torch.as_tensor(batch["u_battery"], device=self.device),
                    torch.as_tensor(batch["caes_mode"], device=self.device),
                    torch.as_tensor(batch["caes_magnitude"], device=self.device),
                    torch.as_tensor(batch["u_tp_low"], device=self.device),
                    torch.as_tensor(batch["u_tp_high"], device=self.device),
                    torch.as_tensor(batch["u_bat_low"], device=self.device),
                    torch.as_tensor(batch["u_bat_high"], device=self.device),
                    torch.as_tensor(batch["mode_mask"], device=self.device),
                )
                old_lp = torch.as_tensor(batch["log_prob"], device=self.device)
                adv_t = torch.as_tensor(batch["advantage"], device=self.device)
                ret = torch.as_tensor(batch["return_"], device=self.device)
                log_ratio = (evaluated["log_prob"] - old_lp).clamp(-20.0, 20.0)
                ratio = log_ratio.exp()
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t
                policy_loss = -torch.min(surr1, surr2).mean()
                values = self.critic(obs)
                value_loss = ((values - ret) ** 2).mean()
                entropy = evaluated["entropy"].mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                if not torch.isfinite(loss):
                    continue
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), 0.5
                )
                self.opt.step()
                approx_kl = float((old_lp - evaluated["log_prob"]).mean().item())
                metrics_acc["policy_loss"] += float(policy_loss.item())
                metrics_acc["value_loss"] += float(value_loss.item())
                metrics_acc["entropy"] += float(entropy.item())
                metrics_acc["approx_kl"] += approx_kl
                n_updates += 1

        metrics = {k: v / max(n_updates, 1) for k, v in metrics_acc.items()}
        if n_updates == 0:
            self.last_metrics = metrics
            return metrics
        if not all(np.isfinite(v) for v in metrics.values()):
            # 不中断训练：跳过本轮坏更新
            self.last_metrics = {k: 0.0 for k in metrics}
            return self.last_metrics
        self.last_metrics = metrics
        return metrics

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "total_it": self.total_it,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        data = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
        self.total_it = int(data.get("total_it", 0))
