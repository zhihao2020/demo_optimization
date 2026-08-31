"""Twin Q: hybrid CAES as (mode one-hot, magnitude), not a scalar on a gapped axis."""

from __future__ import annotations

import torch
import torch.nn as nn

from actions.caes_u import mag_from_u_torch, mode_index_from_u_torch


class HybridCritic(nn.Module):
    """Q(s, u_tp, u_bat, e_m, z) when parameterized; Q(s, u_tp, u_bat, u_caes) otherwise."""

    def __init__(self, obs_dim: int, hidden: int = 256, *, parameterized_caes: bool = True):
        super().__init__()
        self.parameterized_caes = bool(parameterized_caes)
        act_dim = 6 if self.parameterized_caes else 3
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

    @staticmethod
    def _col(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) if x.ndim == 1 else x

    def _pack(
        self,
        obs: torch.Tensor,
        u_tp: torch.Tensor,
        u_bat: torch.Tensor,
        u_caes: torch.Tensor,
        mode_onehot: torch.Tensor | None = None,
        mag: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.parameterized_caes:
            return torch.cat([obs, self._col(u_tp), self._col(u_bat), self._col(u_caes)], dim=-1)
        if mode_onehot is None:
            idx = mode_index_from_u_torch(u_caes)
            mode_onehot = torch.nn.functional.one_hot(idx, num_classes=3).to(
                dtype=obs.dtype, device=obs.device
            )
        if mag is None:
            mag = mag_from_u_torch(u_caes)
        return torch.cat(
            [obs, self._col(u_tp), self._col(u_bat), mode_onehot, self._col(mag)],
            dim=-1,
        )

    def forward(
        self,
        obs,
        u_tp,
        u_bat,
        u_caes,
        mode_onehot: torch.Tensor | None = None,
        mag: torch.Tensor | None = None,
    ):
        x = self._pack(obs, u_tp, u_bat, u_caes, mode_onehot, mag)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_only(
        self,
        obs,
        u_tp,
        u_bat,
        u_caes,
        mode_onehot: torch.Tensor | None = None,
        mag: torch.Tensor | None = None,
    ):
        x = self._pack(obs, u_tp, u_bat, u_caes, mode_onehot, mag)
        return self.q1(x).squeeze(-1)
