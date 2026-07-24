"""混合软演员-评论家(Hybrid-SAC)：随机混合 Actor + Twin Q + 自动温度 α。"""

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
    """混合 SAC(HybridSAC)：最大熵 RL，连续 squashed 高斯 + 离散 Categorical 熵。"""

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
    ):
        """初始化 Actor、Twin Critic、目标 Critic 与可学习 log_α。

        Args:
            obs_dim: 观测维度。
            gamma: 折扣因子。
            tau: 目标 Critic 软更新系数。
            actor_lr: Actor 学习率。
            critic_lr: Critic 学习率。
            alpha_lr: 温度 α 学习率。
            target_entropy: 目标熵；None 时用 -(3 + log(3)) 启发式。
            device: PyTorch 设备。
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gamma = gamma
        self.tau = tau
        # 连续 3 维 + 离散模式：约 -3 - log(3)
        self.target_entropy = float(
            target_entropy if target_entropy is not None else -(3.0 + float(np.log(3.0)))
        )
        self.actor = HybridStochasticActor(obs_dim).to(self.device)
        self.critic = HybridCritic(obs_dim).to(self.device)
        self.critic_target = deepcopy(self.critic).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.total_it = 0
        self.last_metrics: dict[str, float] = {}

    @property
    def alpha(self) -> torch.Tensor:
        """当前温度系数 α = exp(log_alpha)。

        Returns:
            标量 α 张量。
        """
        return self.log_alpha.exp()

    def select_action(self, obs, feasible, deterministic: bool = False) -> dict:
        """采样混合动作。

        Args:
            obs: 环境观测。
            feasible: 可行动作规格。
            deterministic: 是否确定性。

        Returns:
            混合动作字典。
        """
        out = self.actor.act_numpy(obs, feasible, deterministic=deterministic, device=self.device)
        return {
            "u_tp": out["u_tp"],
            "u_battery": out["u_battery"],
            "caes_mode": out["caes_mode"],
            "caes_magnitude": out["caes_magnitude"],
        }

    def update(self, buffer: HybridGiveSafeReplayBuffer, batch_size: int = 256) -> dict[str, float]:
        """从 physical replay 采样并执行 SAC 更新。

        Args:
            buffer: HybridGiveSafeReplayBuffer（仅用 physical 分区采样）。
            batch_size: 批大小。

        Returns:
            含 critic_loss、actor_loss、alpha 等的指标字典。

        Raises:
            RuntimeError: 指标非有限时抛出。
        """
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

        with torch.no_grad():
            next_act = self.actor.forward_action(
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
                next_act["caes_mode_oh"],
                next_act["caes_magnitude"],
            )
            min_q = torch.min(q1_t, q2_t) - self.alpha.detach() * next_act["log_prob"]
            target_q = reward + (1.0 - done) * self.gamma * min_q

        q1, q2 = self.critic(obs, u_tp, u_bat, mode_oh, mag)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        new_act = self.actor.forward_action(
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
            new_act["caes_mode_oh"],
            new_act["caes_magnitude"],
        )
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * new_act["log_prob"] - min_q_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (new_act["log_prob"] + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update(self.critic, self.critic_target)

        metrics = {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "q1_mean": float(q1.mean().item()),
            "entropy": float(new_act["entropy"].mean().item()),
        }
        if not all(np.isfinite(v) for v in metrics.values()):
            raise RuntimeError(f"SAC 出现非有限指标: {metrics}")
        self.last_metrics = metrics
        return metrics

    def _soft_update(self, src: torch.nn.Module, tgt: torch.nn.Module) -> None:
        """Polyak 软更新目标 Critic。

        Args:
            src: 源网络。
            tgt: 目标网络。

        Returns:
            无。
        """
        for sp, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def save(self, path: str | Path) -> None:
        """保存 Actor、Critic 与 log_alpha。

        Args:
            path: 检查点路径。

        Returns:
            无。
        """
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
        """加载检查点并重建 α 优化器。

        Args:
            path: 检查点路径。

        Returns:
            无。
        """
        data = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
        self.critic_target.load_state_dict(data["critic_target"])
        self.log_alpha = data["log_alpha"].to(self.device).clone().detach().requires_grad_(True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.alpha_opt.defaults["lr"])
        self.total_it = int(data.get("total_it", 0))
