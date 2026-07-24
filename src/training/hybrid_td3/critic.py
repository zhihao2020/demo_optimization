"""双 Q 评论家(HybridCritic)：Q(s, u_tp, u_battery, mode_oh, magnitude)。"""

from __future__ import annotations

import torch
import torch.nn as nn


class HybridCritic(nn.Module):
    """双 Q 网络：动作表达为 obs + u_tp + u_battery + caes_mode one-hot + caes_magnitude。"""

    def __init__(self, obs_dim: int, hidden: int = 256):
        """初始化 twin Q 网络。

        Args:
            obs_dim: 观测维度。
            hidden: 隐层宽度。
        """
        super().__init__()
        act_dim = 1 + 1 + 3 + 1  # tp, bat, mode_oh, mag
        in_dim = obs_dim + act_dim
        self.q1 = self._net(in_dim, hidden)
        self.q2 = self._net(in_dim, hidden)

    @staticmethod
    def _net(in_dim: int, hidden: int) -> nn.Sequential:
        """构建单个 Q 网络 MLP。

        Args:
            in_dim: 输入维度（obs + 动作）。
            hidden: 隐层宽度。

        Returns:
            输出标量 Q 值的 Sequential。
        """
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _pack(self, obs, u_tp, u_bat, mode_oh, mag) -> torch.Tensor:
        """拼接观测与动作为 critic 输入向量。

        Args:
            obs: 观测张量。
            u_tp: 火电动作。
            u_bat: 电池动作。
            mode_oh: CAES 模式 one-hot。
            mag: CAES 幅值。

        Returns:
            形状 (B, in_dim) 的拼接张量。
        """
        return torch.cat(
            [
                obs,
                u_tp.unsqueeze(-1) if u_tp.ndim == 1 else u_tp,
                u_bat.unsqueeze(-1) if u_bat.ndim == 1 else u_bat,
                mode_oh,
                mag.unsqueeze(-1) if mag.ndim == 1 else mag,
            ],
            dim=-1,
        )

    def forward(self, obs, u_tp, u_bat, mode_oh, mag):
        """计算 twin Q 值。

        Args:
            obs: 观测。
            u_tp: 火电动作。
            u_bat: 电池动作。
            mode_oh: 模式 one-hot。
            mag: CAES 幅值。

        Returns:
            (Q1, Q2) 元组，各为形状 (B,) 的张量。
        """
        x = self._pack(obs, u_tp, u_bat, mode_oh, mag)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_only(self, obs, u_tp, u_bat, mode_oh, mag):
        """仅返回 Q1（Actor 损失用）。

        Args:
            obs: 观测。
            u_tp: 火电动作。
            u_bat: 电池动作。
            mode_oh: 模式 one-hot。
            mag: CAES 幅值。

        Returns:
            形状 (B,) 的 Q1 张量。
        """
        x = self._pack(obs, u_tp, u_bat, mode_oh, mag)
        return self.q1(x).squeeze(-1)
