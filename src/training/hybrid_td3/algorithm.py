"""Hybrid-TD3 算法更新循环。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .actor import HybridActor
from .buffer import FilteredReplayBuffer
from .critic import HybridCritic


class HybridTD3:
    def __init__(
        self,
        obs_dim: int,
        *,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_delay: int = 2,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        explore_noise: float = 0.1,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.explore_noise = explore_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.actor = HybridActor(obs_dim).to(self.device)
        self.actor_target = deepcopy(self.actor).to(self.device)
        self.critic = HybridCritic(obs_dim).to(self.device)
        self.critic_target = deepcopy(self.critic).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.total_it = 0
        self.last_metrics: dict[str, float] = {}

    def select_action(self, obs, feasible, deterministic: bool = False) -> dict:
        return self.actor.act_numpy(
            obs,
            feasible,
            deterministic=deterministic,
            device=self.device,
            explore_noise_std=0.0 if deterministic else self.explore_noise,
        )

    def update(self, buffer: FilteredReplayBuffer, batch_size: int = 256) -> dict[str, float]:
        if len(buffer) < batch_size:
            return {}
        self.total_it += 1
        batch = buffer.sample(batch_size)
        obs = torch.as_tensor(batch["obs"], device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], device=self.device)
        u_tp = torch.as_tensor(batch["u_tp"], device=self.device)
        u_bat = torch.as_tensor(batch["u_battery"], device=self.device)
        mode = torch.as_tensor(batch["caes_mode"], device=self.device, dtype=torch.int64)
        mode_oh = F.one_hot(mode, num_classes=3).float()
        mag = torch.as_tensor(batch["caes_magnitude"], device=self.device)
        reward = torch.as_tensor(batch["reward"], device=self.device)
        done = torch.as_tensor(batch["done"], device=self.device)
        next_mask = torch.as_tensor(batch["next_mode_mask"], device=self.device)

        with torch.no_grad():
            next_act = self.actor_target.act(
                next_obs,
                torch.as_tensor(batch["next_u_tp_low"], device=self.device),
                torch.as_tensor(batch["next_u_tp_high"], device=self.device),
                torch.as_tensor(batch["next_u_bat_low"], device=self.device),
                torch.as_tensor(batch["next_u_bat_high"], device=self.device),
                next_mask,
                deterministic=False,
                explore_noise_std=0.0,
            )
            # target policy smoothing 仅作用于连续动作
            noise_tp = (torch.randn_like(next_act["u_tp"]) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
            noise_bat = (torch.randn_like(next_act["u_battery"]) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
            noise_mag = (torch.randn_like(next_act["caes_magnitude"]) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
            n_tp = torch.clamp(
                next_act["u_tp"] + noise_tp,
                torch.as_tensor(batch["next_u_tp_low"], device=self.device),
                torch.as_tensor(batch["next_u_tp_high"], device=self.device),
            )
            n_bat = torch.clamp(
                next_act["u_battery"] + noise_bat,
                torch.as_tensor(batch["next_u_bat_low"], device=self.device),
                torch.as_tensor(batch["next_u_bat_high"], device=self.device),
            )
            n_mag = torch.clamp(next_act["caes_magnitude"] + noise_mag, 0.0, 1.0)
            # 不对离散模式加高斯噪声
            q1_t, q2_t = self.critic_target(next_obs, n_tp, n_bat, next_act["caes_mode_oh"], n_mag)
            target_q = reward + (1.0 - done) * self.gamma * torch.min(q1_t, q2_t)

        q1, q2 = self.critic(obs, u_tp, u_bat, mode_oh, mag)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss_v = 0.0
        if self.total_it % self.policy_delay == 0:
            cur = self.actor.act(
                obs,
                torch.as_tensor(batch["u_tp_low"], device=self.device),
                torch.as_tensor(batch["u_tp_high"], device=self.device),
                torch.as_tensor(batch["u_bat_low"], device=self.device),
                torch.as_tensor(batch["u_bat_high"], device=self.device),
                torch.as_tensor(batch["mode_mask"], device=self.device),
                deterministic=False,
                explore_noise_std=0.0,
            )
            actor_loss = -self.critic.q1_only(
                obs, cur["u_tp"], cur["u_battery"], cur["caes_mode_oh"], cur["caes_magnitude"]
            ).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            actor_loss_v = float(actor_loss.item())

        metrics = {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": actor_loss_v,
            "q1_mean": float(q1.mean().item()),
            "q2_mean": float(q2.mean().item()),
        }
        if not all(np.isfinite(v) for v in metrics.values()):
            raise RuntimeError(f"训练出现非有限指标: {metrics}")
        self.last_metrics = metrics
        return metrics

    def _soft_update(self, src: torch.nn.Module, tgt: torch.nn.Module) -> None:
        for sp, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "total_it": self.total_it,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        data = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
        self.actor_target.load_state_dict(data["actor_target"])
        self.critic_target.load_state_dict(data["critic_target"])
        self.total_it = int(data.get("total_it", 0))
