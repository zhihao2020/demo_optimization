"""混合近端策略优化(Hybrid-PPO)：随机混合 Actor + V(s) + clipped surrogate。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from training.hybrid_common.stochastic_actor import HybridStochasticActor

from .rollout import RolloutBuffer


class ValueNet(nn.Module):
    """状态价值网络(ValueNet)：输出 V(s) 标量。"""

    def __init__(self, obs_dim: int, hidden: int = 256):
        """初始化三层 MLP 价值头。

        Args:
            obs_dim: 观测维度。
            hidden: 隐层宽度。
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """前向计算 V(s)。

        Args:
            obs: 形状 (B, obs_dim) 的观测。

        Returns:
            形状 (B,) 的价值张量。
        """
        return self.net(obs).squeeze(-1)


class HybridPPO:
    """混合 PPO(HybridPPO)：仅物理有效步进 rollout，GiveSafe 拒绝不参与策略梯度。"""

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
        """初始化 Actor、Critic 与联合优化器。

        Args:
            obs_dim: 观测维度。
            gamma: 折扣因子。
            gae_lambda: GAE λ。
            clip_eps: PPO clip 范围 ε。
            ent_coef: 熵 bonus 系数。
            vf_coef: 价值损失权重。
            lr: Adam 学习率。
            update_epochs: 每批 rollout 的优化轮数。
            minibatch_size: 小批大小。
            device: PyTorch 设备。
        """
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
        """采样混合动作（不含 log_prob，供 GiveSafe propose 用）。

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

    def value_numpy(self, obs) -> float:
        """计算单步 V(s) 标量。

        Args:
            obs: 一维观测数组。

        Returns:
            状态价值浮点数。
        """
        self.critic.eval()
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
            return float(self.critic(o)[0].cpu())

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """对 rollout 缓冲执行 PPO 多 epoch 更新。

        Args:
            buffer: 已计算 GAE 的 RolloutBuffer。

        Returns:
            平均 policy_loss、value_loss、entropy、approx_kl；空 buffer 时返回空字典。
        """
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
        """保存 Actor 与 Critic 权重。

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
                "total_it": self.total_it,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """加载检查点。

        Args:
            path: 检查点路径。

        Returns:
            无。
        """
        data = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
        self.total_it = int(data.get("total_it", 0))
