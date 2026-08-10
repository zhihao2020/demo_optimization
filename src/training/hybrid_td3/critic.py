"""双 Q 评论家：Q(s, u_tp, u_battery, u_caes)。"""

from __future__ import annotations

import torch
import torch.nn as nn


class HybridCritic(nn.Module):
    """双 Q 网络：动作表达为 obs + 三个连续 FMU 指令。"""

    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        act_dim = 3
        in_dim = obs_dim + act_dim
        self.q1 = self._net(in_dim, hidden)
        self.q2 = self._net(in_dim, hidden)

    @staticmethod
    def _net(in_dim: int, hidden: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _pack(self, obs, u_tp, u_bat, u_caes) -> torch.Tensor:
        def _col(x):
            return x.unsqueeze(-1) if x.ndim == 1 else x

        return torch.cat([obs, _col(u_tp), _col(u_bat), _col(u_caes)], dim=-1)

    def forward(self, obs, u_tp, u_bat, u_caes):
        x = self._pack(obs, u_tp, u_bat, u_caes)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_only(self, obs, u_tp, u_bat, u_caes):
        x = self._pack(obs, u_tp, u_bat, u_caes)
        return self.q1(x).squeeze(-1)
