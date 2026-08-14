"""SAC：随机 Actor + Twin Q + 自动温度 α；动作 (u_tp, u_battery, u_caes)。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from replay import HybridGiveSafeReplayBuffer
from training.hybrid_common.stochastic_actor import HybridStochasticActor
from training.hybrid_td3.critic import HybridCritic


class HybridSAC:
    """SAC：最大熵 RL，三维连续物理动作。

    Applied Energy 迭代加固：自动温度 ``alpha`` 裁剪，避免过渡季/夏季
    出现 alpha→1e17、critic_loss→inf 的 fail-fast 崩溃。
    """

    def __init__(
        self,
        obs_dim: int,
        *,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        target_entropy: float | None = None,
        device: str | None = None,
        alpha_min: float = 1e-4,
        alpha_max: float = 10.0,
        q_clip: float = 200.0,
        skip_nonfinite_update: bool = True,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = float(target_entropy if target_entropy is not None else -3.0)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.log_alpha_min = float(np.log(self.alpha_min))
        self.log_alpha_max = float(np.log(self.alpha_max))
        self.q_clip = float(q_clip)
        self.skip_nonfinite_update = bool(skip_nonfinite_update)
        self.actor = HybridStochasticActor(obs_dim).to(self.device)
        self.critic = HybridCritic(obs_dim).to(self.device)
        self.critic_target = deepcopy(self.critic).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.total_it = 0
        self.nonfinite_skips = 0
        self.last_metrics: dict[str, float] = {}

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.clamp(self.log_alpha_min, self.log_alpha_max).exp()

    def _clamp_log_alpha_(self) -> None:
        with torch.no_grad():
            self.log_alpha.clamp_(self.log_alpha_min, self.log_alpha_max)

    def select_action(self, obs, feasible, deterministic: bool = False) -> dict:
        return self.actor.act_numpy(
            obs, feasible, deterministic=deterministic, device=self.device
        )

    def update(self, buffer: HybridGiveSafeReplayBuffer, batch_size: int = 256) -> dict[str, float]:
        if len(buffer) < batch_size:
            return {}
        self.total_it += 1
        batch = buffer.sample(batch_size)
        obs = torch.as_tensor(batch["obs"], device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], device=self.device)
        u_tp = torch.as_tensor(batch["u_tp"], device=self.device)
        u_bat = torch.as_tensor(batch["u_battery"], device=self.device)
        u_caes = torch.as_tensor(batch["u_caes"], device=self.device)
        reward = torch.as_tensor(batch["reward"], device=self.device)
        done = torch.as_tensor(batch["done"], device=self.device)

        with torch.no_grad():
            next_act = self.actor.act(
                next_obs,
                torch.as_tensor(batch["next_u_tp_low"], device=self.device),
                torch.as_tensor(batch["next_u_tp_high"], device=self.device),
                torch.as_tensor(batch["next_u_bat_low"], device=self.device),
                torch.as_tensor(batch["next_u_bat_high"], device=self.device),
                torch.as_tensor(batch["next_mode_mask"], device=self.device),
                deterministic=False,
            )
            q1_t, q2_t = self.critic_target(
                next_obs,
                next_act["u_tp"],
                next_act["u_battery"],
                next_act["u_caes"],
            )
            min_q = torch.min(q1_t, q2_t) - self.alpha.detach() * next_act["log_prob"]
            if self.q_clip > 0:
                min_q = min_q.clamp(-self.q_clip, self.q_clip)
            target_q = reward + (1.0 - done) * self.gamma * min_q
            if self.q_clip > 0:
                target_q = target_q.clamp(-self.q_clip, self.q_clip)

        q1, q2 = self.critic(obs, u_tp, u_bat, u_caes)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        if not torch.isfinite(critic_loss):
            self.nonfinite_skips += 1
            if self.skip_nonfinite_update:
                self._clamp_log_alpha_()
                return {
                    "critic_loss": float("nan"),
                    "actor_loss": float("nan"),
                    "alpha_loss": float("nan"),
                    "alpha": float(self.alpha.item()),
                    "q1_mean": float("nan"),
                    "entropy": float("nan"),
                    "nonfinite_skips": float(self.nonfinite_skips),
                    "skipped": 1.0,
                }
            raise RuntimeError(f"SAC critic_loss non-finite: {float(critic_loss)}")
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_opt.step()

        new_act = self.actor.act(
            obs,
            torch.as_tensor(batch["u_tp_low"], device=self.device),
            torch.as_tensor(batch["u_tp_high"], device=self.device),
            torch.as_tensor(batch["u_bat_low"], device=self.device),
            torch.as_tensor(batch["u_bat_high"], device=self.device),
            torch.as_tensor(batch["mode_mask"], device=self.device),
            deterministic=False,
        )
        q1_pi, q2_pi = self.critic(
            obs,
            new_act["u_tp"],
            new_act["u_battery"],
            new_act["u_caes"],
        )
        min_q_pi = torch.min(q1_pi, q2_pi)
        if self.q_clip > 0:
            min_q_pi = min_q_pi.clamp(-self.q_clip, self.q_clip)
        actor_loss = (self.alpha.detach() * new_act["log_prob"] - min_q_pi).mean()
        if not torch.isfinite(actor_loss):
            self.nonfinite_skips += 1
            self._clamp_log_alpha_()
            if self.skip_nonfinite_update:
                return {
                    "critic_loss": float(critic_loss.item()),
                    "actor_loss": float("nan"),
                    "alpha_loss": float("nan"),
                    "alpha": float(self.alpha.item()),
                    "q1_mean": float(q1.mean().item()),
                    "entropy": float(new_act["entropy"].mean().item()),
                    "nonfinite_skips": float(self.nonfinite_skips),
                    "skipped": 1.0,
                }
            raise RuntimeError("SAC actor_loss non-finite")
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (new_act["log_prob"] + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self._clamp_log_alpha_()

        self._soft_update(self.critic, self.critic_target)

        metrics = {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "q1_mean": float(q1.mean().item()),
            "entropy": float(new_act["entropy"].mean().item()),
            "nonfinite_skips": float(self.nonfinite_skips),
            "skipped": 0.0,
        }
        if not all(np.isfinite(v) for k, v in metrics.items() if k not in ("nonfinite_skips", "skipped")):
            self.nonfinite_skips += 1
            if self.skip_nonfinite_update:
                metrics["skipped"] = 1.0
                self.last_metrics = metrics
                return metrics
            raise RuntimeError(f"SAC 出现非有限指标: {metrics}")
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
                "critic_target": self.critic_target.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "total_it": self.total_it,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        data = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
        self.critic_target.load_state_dict(data["critic_target"])
        self.log_alpha = data["log_alpha"].to(self.device).clone().detach().requires_grad_(True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.alpha_opt.defaults["lr"])
        self.total_it = int(data.get("total_it", 0))
