"""Twin Q on the executed physical triple. 检查.txt P0: do not pack (e_m, z)."""

from __future__ import annotations

import torch
import torch.nn as nn


class HybridCritic(nn.Module):
    """Q(s, u_tp, u_bat, u_caes). Mode/magnitude stay in the actor decoder."""

    def __init__(self, obs_dim: int, hidden: int = 256, *, parameterized_caes: bool = True):
        super().__init__()
        _ = parameterized_caes
        in_dim = int(obs_dim) + 3
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

    @staticmethod
    def _col(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) if x.ndim == 1 else x

    def _pack(self, obs, u_tp, u_bat, u_caes, *_, **__):
        return torch.cat(
            [obs, self._col(u_tp), self._col(u_bat), self._col(u_caes)],
            dim=-1,
        )

    def forward(self, obs, u_tp, u_bat, u_caes, mode_onehot=None, mag=None):
        x = self._pack(obs, u_tp, u_bat, u_caes)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_only(self, obs, u_tp, u_bat, u_caes, mode_onehot=None, mag=None):
        x = self._pack(obs, u_tp, u_bat, u_caes)
        return self.q1(x).squeeze(-1)
