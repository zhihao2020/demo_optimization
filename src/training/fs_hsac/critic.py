"""FS-HSAC critic: Q(s, u_tp, u_battery, mode_onehot, mag)."""

from __future__ import annotations

import torch
import torch.nn as nn

from training.fs_hsac.action_support import MODE_CHARGE, MODE_DISCHARGE, MODE_IDLE, one_hot_modes


def _net(in_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )


class FSHSACCritic(nn.Module):
    """Twin Q over explicit hybrid CAES encoding (mode one-hot + normalized mag)."""

    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        # obs + u_tp + u_bat + onehot(3) + mag
        in_dim = int(obs_dim) + 2 + 3 + 1
        self.q1 = _net(in_dim, hidden)
        self.q2 = _net(in_dim, hidden)

    @staticmethod
    def _col(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) if x.ndim == 1 else x

    def _pack(
        self,
        obs: torch.Tensor,
        u_tp: torch.Tensor,
        u_bat: torch.Tensor,
        mode_onehot: torch.Tensor,
        mag: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [obs, self._col(u_tp), self._col(u_bat), mode_onehot, self._col(mag)],
            dim=-1,
        )

    def forward(
        self,
        obs: torch.Tensor,
        u_tp: torch.Tensor,
        u_bat: torch.Tensor,
        mode_onehot: torch.Tensor,
        mag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._pack(obs, u_tp, u_bat, mode_onehot, mag)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q_from_physical(
        self,
        obs: torch.Tensor,
        u_tp: torch.Tensor,
        u_bat: torch.Tensor,
        u_caes: torch.Tensor,
        *,
        dis_lo: torch.Tensor,
        dis_hi: torch.Tensor,
        chg_lo: torch.Tensor,
        chg_hi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode physical u_caes into (mode, mag) then evaluate Q."""
        mode = torch.ones_like(u_caes, dtype=torch.long)
        mode = torch.where(u_caes < -1e-6, torch.zeros_like(mode), mode)
        mode = torch.where(u_caes > 1e-6, torch.full_like(mode, 2), mode)
        mag = torch.zeros_like(u_caes)
        mag = torch.where(
            mode == MODE_DISCHARGE,
            ((u_caes - dis_lo) / (dis_hi - dis_lo).clamp_min(1e-4)).clamp(0.0, 1.0),
            mag,
        )
        mag = torch.where(
            mode == MODE_CHARGE,
            ((u_caes - chg_lo) / (chg_hi - chg_lo).clamp_min(1e-4)).clamp(0.0, 1.0),
            mag,
        )
        onehot = one_hot_modes(mode).to(device=obs.device, dtype=obs.dtype)
        return self.forward(obs, u_tp, u_bat, onehot, mag)
