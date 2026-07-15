"""Twin Critic：Q(s, u_tp, u_battery, mode_oh, magnitude)。"""

from __future__ import annotations

import torch
import torch.nn as nn


class HybridCritic(nn.Module):
    """动作表达：obs + u_tp + u_battery + caes_mode one-hot + caes_magnitude。"""

    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        act_dim = 1 + 1 + 3 + 1  # tp, bat, mode_oh, mag
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

    def _pack(self, obs, u_tp, u_bat, mode_oh, mag) -> torch.Tensor:
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
        x = self._pack(obs, u_tp, u_bat, mode_oh, mag)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_only(self, obs, u_tp, u_bat, mode_oh, mag):
        x = self._pack(obs, u_tp, u_bat, mode_oh, mag)
        return self.q1(x).squeeze(-1)
