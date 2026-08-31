"""混合双延迟深度确定性策略梯度(HybridTD3) 算法更新循环。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from actions.caes_u import (
    CHARGE_HI,
    CHARGE_LO,
    DISCHARGE_HI,
    DISCHARGE_LO,
    apply_mode_mask_to_u_torch,
    project_u_caes_torch,
    u_from_mode_onehot_dynamic,
    u_from_mode_onehot_torch,
)
from .actor import HybridActor
from .buffer import FilteredReplayBuffer
from .critic import HybridCritic
from training.hybrid_common.param_caes import infer_parameterized_caes


class HybridTD3:
    """混合 TD3(HybridTD3)：有界混合 Actor + Twin Critic + 延迟策略更新与目标平滑。"""

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
        q_clip: float = 200.0,
        device: str | None = None,
        parameterized_caes: bool = True,
        use_dynamic_support: bool = True,
    ):
        """初始化 Actor/Critic、目标网络与优化器。

        Args:
            obs_dim: 观测维度。
            gamma: 折扣因子。
            tau: 目标网络软更新系数。
            policy_delay: Actor 相对 Critic 的更新间隔（步数）。
            actor_lr: Actor 学习率。
            critic_lr: Critic 学习率。
            explore_noise: 行为策略连续动作探索噪声标准差。
            target_noise: 目标策略平滑噪声标准差。
            noise_clip: 目标平滑噪声 clip 范围。
            device: PyTorch 设备；None 时自动选 CUDA/CPU。
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.explore_noise = explore_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.q_clip = float(q_clip)
        self.parameterized_caes = bool(parameterized_caes)
        self.use_dynamic_support = bool(use_dynamic_support)
        self.obs_dim = int(obs_dim)
        self._actor_lr = float(actor_lr)
        self.actor = HybridActor(
            obs_dim,
            parameterized_caes=self.parameterized_caes,
            use_dynamic_support=self.use_dynamic_support,
        ).to(self.device)
        self.actor_target = deepcopy(self.actor).to(self.device)
        self.critic = HybridCritic(obs_dim, parameterized_caes=self.parameterized_caes).to(
            self.device
        )
        self.critic_target = deepcopy(self.critic).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.total_it = 0
        self.last_metrics: dict[str, float] = {}

    def select_action(self, obs, feasible, deterministic: bool = False) -> dict:
        """根据可行域采样混合动作。

        Args:
            obs: 环境观测。
            feasible: 可行动作规格。
            deterministic: 是否关闭探索噪声。

        Returns:
            混合动作字典。
        """
        return self.actor.act_numpy(
            obs,
            feasible,
            deterministic=deterministic,
            device=self.device,
            explore_noise_std=0.0 if deterministic else self.explore_noise,
        )

    def update(self, buffer: FilteredReplayBuffer, batch_size: int = 256) -> dict[str, float]:
        """从 replay 采样一批并执行 TD3 更新。

        Args:
            buffer: 过滤经济 replay；亦兼容 HybridGiveSafeReplayBuffer 的 sample。
            batch_size: 批大小。

        Returns:
            含 critic_loss、actor_loss、q1_mean 等的指标字典；样本不足时返回空字典。

        Raises:
            RuntimeError: 指标出现 NaN/Inf 时抛出。
        """
        if len(buffer) < batch_size:
            return {}
        self.total_it += 1
        batch = buffer.sample(batch_size)
        b = int(batch["obs"].shape[0])
        dev = self.device

        def _t(key: str, default: float | None = None) -> torch.Tensor:
            if key in batch:
                return torch.as_tensor(batch[key], device=dev)
            assert default is not None
            return torch.full((b,), float(default), device=dev)

        obs = _t("obs")
        next_obs = _t("next_obs")
        u_tp = _t("u_tp")
        u_bat = _t("u_battery")
        u_caes = _t("u_caes")
        reward = _t("reward")
        done = _t("done")
        next_mask = torch.as_tensor(batch["next_mode_mask"], device=dev)
        mask = torch.as_tensor(batch["mode_mask"], device=dev)

        with torch.no_grad():
            next_act = self.actor_target.act(
                next_obs,
                _t("next_u_tp_low"),
                _t("next_u_tp_high"),
                _t("next_u_bat_low"),
                _t("next_u_bat_high"),
                next_mask,
                deterministic=True,
                explore_noise_std=0.0,
                dis_lo=_t("next_dis_lo", DISCHARGE_LO),
                dis_hi=_t("next_dis_hi", DISCHARGE_HI),
                chg_lo=_t("next_chg_lo", CHARGE_LO),
                chg_hi=_t("next_chg_hi", CHARGE_HI),
                use_dynamic_support=self.use_dynamic_support,
                grid_residual=_t("next_grid_residual_W", 0.0),
                grid_g_min=_t("next_grid_g_min_W", -5.0e8),
                grid_g_max=_t("next_grid_g_max_W", 5.0e8),
                p_cap_thermal=_t("p_cap_thermal_W", 1.5e8),
                p_cap_battery=_t("p_cap_battery_W", 1.0e8),
                p_cap_caes=_t("p_cap_caes_W", 1.5e8),
            )
            noise_tp = (torch.randn_like(next_act["u_tp"]) * self.target_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            noise_bat = (torch.randn_like(next_act["u_battery"]) * self.target_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            n_tp = torch.clamp(next_act["u_tp"] + noise_tp, _t("next_u_tp_low"), _t("next_u_tp_high"))
            n_bat = torch.clamp(
                next_act["u_battery"] + noise_bat, _t("next_u_bat_low"), _t("next_u_bat_high")
            )
            if self.parameterized_caes and "mag" in next_act:
                mag = next_act["mag"]
                mag = (mag + (torch.randn_like(mag) * self.target_noise).clamp(
                    -self.noise_clip, self.noise_clip
                )).clamp(0.0, 1.0)
                if self.use_dynamic_support:
                    n_caes = u_from_mode_onehot_dynamic(
                        next_act["mode_onehot"],
                        mag,
                        _t("next_dis_lo", DISCHARGE_LO),
                        _t("next_dis_hi", DISCHARGE_HI),
                        _t("next_chg_lo", CHARGE_LO),
                        _t("next_chg_hi", CHARGE_HI),
                    )
                else:
                    n_caes = u_from_mode_onehot_torch(next_act["mode_onehot"], mag)
            else:
                n_caes = project_u_caes_torch(
                    torch.clamp(
                        next_act["u_caes"]
                        + (torch.randn_like(next_act["u_caes"]) * self.target_noise).clamp(
                            -self.noise_clip, self.noise_clip
                        ),
                        -1.0,
                        1.0,
                    )
                )
            n_caes = apply_mode_mask_to_u_torch(n_caes, next_mask)
            from actions.joint_support import decode_joint_torch

            n_tp, n_bat = decode_joint_torch(
                n_tp,
                n_bat,
                n_caes,
                _t("next_u_tp_low"),
                _t("next_u_tp_high"),
                _t("next_u_bat_low"),
                _t("next_u_bat_high"),
                _t("next_grid_residual_W", 0.0),
                _t("next_grid_g_min_W", -5.0e8),
                _t("next_grid_g_max_W", 5.0e8),
                _t("p_cap_thermal_W", 1.5e8),
                _t("p_cap_battery_W", 1.0e8),
                _t("p_cap_caes_W", 1.5e8),
            )
            q1_t, q2_t = self.critic_target(next_obs, n_tp, n_bat, n_caes)
            q_t = torch.min(q1_t, q2_t).clamp(-self.q_clip, self.q_clip)
            target_q = reward + (1.0 - done) * self.gamma * q_t

        q1, q2 = self.critic(obs, u_tp, u_bat, u_caes)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss_v = 0.0
        if self.total_it % self.policy_delay == 0:
            cur = self.actor.act(
                obs,
                _t("u_tp_low"),
                _t("u_tp_high"),
                _t("u_bat_low"),
                _t("u_bat_high"),
                mask,
                deterministic=False,
                explore_noise_std=0.0,
                dis_lo=_t("dis_lo", DISCHARGE_LO),
                dis_hi=_t("dis_hi", DISCHARGE_HI),
                chg_lo=_t("chg_lo", CHARGE_LO),
                chg_hi=_t("chg_hi", CHARGE_HI),
                use_dynamic_support=self.use_dynamic_support,
                grid_residual=_t("grid_residual_W", 0.0),
                grid_g_min=_t("grid_g_min_W", -5.0e8),
                grid_g_max=_t("grid_g_max_W", 5.0e8),
                p_cap_thermal=_t("p_cap_thermal_W", 1.5e8),
                p_cap_battery=_t("p_cap_battery_W", 1.0e8),
                p_cap_caes=_t("p_cap_caes_W", 1.5e8),
            )
            actor_loss = -self.critic.q1_only(
                obs,
                cur["u_tp"],
                cur["u_battery"],
                cur["u_caes"],
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
        """Polyak 软更新目标网络参数。

        Args:
            src: 源网络。
            tgt: 目标网络。

        Returns:
            无。
        """
        for sp, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def save(self, path: str | Path) -> None:
        """保存 Actor/Critic 及目标网络权重。

        Args:
            path: 检查点文件路径。

        Returns:
            无。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "total_it": self.total_it,
                "parameterized_caes": self.parameterized_caes,
                "use_dynamic_support": self.use_dynamic_support,
            },
            path,
        )

    def load(self, path: str | Path, *, reset_critic: bool = False) -> None:
        data = torch.load(path, map_location=self.device, weights_only=False)
        actor_state = data["actor"]
        flag = infer_parameterized_caes(
            actor_state,
            explicit=data.get("parameterized_caes"),
        )
        dyn = bool(data.get("use_dynamic_support", self.use_dynamic_support))
        self.use_dynamic_support = dyn
        if flag != self.parameterized_caes:
            self.parameterized_caes = flag
            self.actor = HybridActor(
                self.obs_dim,
                parameterized_caes=flag,
                use_dynamic_support=dyn,
            ).to(self.device)
            self.actor_target = deepcopy(self.actor).to(self.device)
            self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self._actor_lr)
            self.critic = HybridCritic(self.obs_dim, parameterized_caes=flag).to(self.device)
            self.critic_target = deepcopy(self.critic).to(self.device)
            self.critic_opt = torch.optim.Adam(
                self.critic.parameters(), lr=self.critic_opt.param_groups[0]["lr"]
            )
        self.actor.load_state_dict(actor_state)
        self.actor_target.load_state_dict(data.get("actor_target", data["actor"]))
        self.actor.use_dynamic_support = dyn
        self.actor_target.use_dynamic_support = dyn
        if reset_critic:
            # Critic 发散时只保留 actor，重估 Q，避免被坏 value 锁死
            self.critic = HybridCritic(
                self.obs_dim, parameterized_caes=self.parameterized_caes
            ).to(self.device)
            self.critic_target = deepcopy(self.critic).to(self.device)
            self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.critic_opt.param_groups[0]["lr"])
            self.total_it = 0
        else:
            self.critic.load_state_dict(data["critic"])
            self.critic_target.load_state_dict(data.get("critic_target", data["critic"]))
            self.total_it = int(data.get("total_it", 0))
