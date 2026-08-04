"""GHTD3 高/低层 Actor-Critic。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from actions import CaesMode


def _mlp(in_dim: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
    )


class HighLevelActor(nn.Module):
    """s -> goal（有界 SoC 增量）。"""

    def __init__(self, obs_dim: int, goal_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.goal_dim = goal_dim
        self.encoder = _mlp(obs_dim, hidden)
        self.head = nn.Linear(hidden, goal_dim)
        # 小目标先验
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs: torch.Tensor, goal_low: torch.Tensor, goal_high: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.head(self.encoder(obs)))
        # map [-1,1] -> [low, high]
        mid = 0.5 * (goal_high + goal_low)
        half = 0.5 * (goal_high - goal_low)
        return mid + z * half


class HighLevelCritic(nn.Module):
    def __init__(self, obs_dim: int, goal_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.q1 = _mlp(obs_dim + goal_dim, hidden)
        self.q1_out = nn.Linear(hidden, 1)
        self.q2 = _mlp(obs_dim + goal_dim, hidden)
        self.q2_out = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor, goal: torch.Tensor):
        x = torch.cat([obs, goal], dim=-1)
        return self.q1_out(self.q1(x)).squeeze(-1), self.q2_out(self.q2(x)).squeeze(-1)

    def q1_only(self, obs: torch.Tensor, goal: torch.Tensor):
        x = torch.cat([obs, goal], dim=-1)
        return self.q1_out(self.q1(x)).squeeze(-1)


class LowLevelActor(nn.Module):
    """[s, g] -> hybrid action（有界火电/电池 + CAES 模式）。"""

    def __init__(self, obs_dim: int, goal_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.encoder = _mlp(obs_dim + goal_dim, hidden)
        self.thermal_head = nn.Linear(hidden, 1)
        self.battery_head = nn.Linear(hidden, 1)
        self.mode_head = nn.Linear(hidden, 3)
        self.discharge_mag_head = nn.Linear(hidden, 1)
        self.charge_mag_head = nn.Linear(hidden, 1)
        nn.init.constant_(self.thermal_head.bias, 2.0)
        nn.init.constant_(self.battery_head.bias, 0.0)
        with torch.no_grad():
            self.mode_head.bias.zero_()
            self.mode_head.bias[int(CaesMode.IDLE)] = 2.0
            self.mode_head.bias[int(CaesMode.DISCHARGE)] = -1.0
            self.mode_head.bias[int(CaesMode.CHARGE)] = -1.0

    @staticmethod
    def map_bounded(z: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        mapped = low + torch.sigmoid(z) * (high - low)
        return torch.minimum(torch.maximum(mapped, low), high)

    def forward_logits(self, obs: torch.Tensor, goal: torch.Tensor) -> dict[str, torch.Tensor]:
        """BC 用：返回 pre-squash logit。"""
        h = self.encoder(torch.cat([obs, goal], dim=-1))
        return {
            "z_tp": self.thermal_head(h).squeeze(-1),
            "z_bat": self.battery_head(h).squeeze(-1),
            "logits_mode": self.mode_head(h),
            "z_discharge": self.discharge_mag_head(h).squeeze(-1),
            "z_charge": self.charge_mag_head(h).squeeze(-1),
        }

    def act(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
        u_tp_low: torch.Tensor,
        u_tp_high: torch.Tensor,
        u_bat_low: torch.Tensor,
        u_bat_high: torch.Tensor,
        mode_mask: torch.Tensor,
        *,
        deterministic: bool = False,
        explore_noise_std: float = 0.0,
        gumbel_tau: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        h = self.encoder(torch.cat([obs, goal], dim=-1))
        logits = self.mode_head(h).masked_fill(~mode_mask.bool(), -1e9)
        if deterministic:
            mode = torch.argmax(logits, dim=-1)
            mode_oh = F.one_hot(mode, num_classes=3).float()
        else:
            mode_oh = F.gumbel_softmax(logits, tau=gumbel_tau, hard=True, dim=-1)
            mode = torch.argmax(mode_oh, dim=-1)
        u_tp = self.map_bounded(self.thermal_head(h).squeeze(-1), u_tp_low, u_tp_high)
        u_bat = self.map_bounded(self.battery_head(h).squeeze(-1), u_bat_low, u_bat_high)
        mag_d = torch.sigmoid(self.discharge_mag_head(h).squeeze(-1))
        mag_c = torch.sigmoid(self.charge_mag_head(h).squeeze(-1))
        mag = mode_oh[:, 0] * mag_d + mode_oh[:, 2] * mag_c
        if explore_noise_std > 0 and not deterministic:
            u_tp = torch.clamp(u_tp + explore_noise_std * torch.randn_like(u_tp), u_tp_low, u_tp_high)
            u_bat = torch.clamp(u_bat + explore_noise_std * torch.randn_like(u_bat), u_bat_low, u_bat_high)
            mag = torch.clamp(mag + explore_noise_std * torch.randn_like(mag), 0.0, 1.0)
            mag = torch.where(mode == int(CaesMode.IDLE), torch.zeros_like(mag), mag)
        return {
            "u_tp": u_tp,
            "u_battery": u_bat,
            "caes_mode": mode,
            "caes_mode_oh": mode_oh,
            "caes_magnitude": mag,
            "logits_mode": logits,
        }


class LowLevelCritic(nn.Module):
    def __init__(self, obs_dim: int, goal_dim: int = 2, hidden: int = 256):
        super().__init__()
        act_dim = 1 + 1 + 3 + 1
        in_dim = obs_dim + goal_dim + act_dim
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

    def _pack(self, obs, goal, u_tp, u_bat, mode_oh, mag):
        return torch.cat(
            [
                obs,
                goal,
                u_tp.unsqueeze(-1) if u_tp.ndim == 1 else u_tp,
                u_bat.unsqueeze(-1) if u_bat.ndim == 1 else u_bat,
                mode_oh,
                mag.unsqueeze(-1) if mag.ndim == 1 else mag,
            ],
            dim=-1,
        )

    def forward(self, obs, goal, u_tp, u_bat, mode_oh, mag):
        x = self._pack(obs, goal, u_tp, u_bat, mode_oh, mag)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_only(self, obs, goal, u_tp, u_bat, mode_oh, mag):
        x = self._pack(obs, goal, u_tp, u_bat, mode_oh, mag)
        return self.q1(x).squeeze(-1)
