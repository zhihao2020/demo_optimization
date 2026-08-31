"""GHTD3 高/低层 Actor-Critic（物理三元组动作）。"""

from __future__ import annotations

import torch
import torch.nn as nn

from actions.caes_u import (
    apply_mode_mask_to_u_torch,
    legalize_mode_mask,
    project_u_caes_torch,
    u_from_mode_onehot_torch,
)


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
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs: torch.Tensor, goal_low: torch.Tensor, goal_high: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.head(self.encoder(obs)))
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
    """[s, g] → (u_tp, u_battery, u_caes).

    ``hybrid_caes``: mode logits + magnitude; otherwise one tanh scalar projected
    onto the disconnected legal set (legacy).
    """

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int = 2,
        hidden: int = 256,
        *,
        residual_init: bool = False,
        goal_input_scale: float = 1.0,
        continuous_caes: bool = True,
        hybrid_caes: bool = False,
    ):
        super().__init__()
        self.goal_dim = int(goal_dim)
        self.goal_input_scale = float(goal_input_scale)
        self.hybrid_caes = bool(hybrid_caes)
        self.encoder = _mlp(obs_dim + goal_dim, hidden)
        self.thermal_head = nn.Linear(hidden, 1)
        self.battery_head = nn.Linear(hidden, 1)
        self.caes_head = nn.Linear(hidden, 1)
        self.caes_mode_head = nn.Linear(hidden, 3)
        self.caes_mag_head = nn.Linear(hidden, 1)
        nn.init.constant_(self.thermal_head.bias, 2.0)
        nn.init.constant_(self.battery_head.bias, 0.0)
        nn.init.zeros_(self.caes_head.bias)
        nn.init.zeros_(self.caes_mode_head.weight)
        nn.init.zeros_(self.caes_mag_head.weight)
        # Slight idle prior; not a lock. Indices: discharge, idle, charge.
        nn.init.constant_(self.caes_mode_head.bias, 0.0)
        self.caes_mode_head.bias.data[1] = 0.4
        nn.init.zeros_(self.caes_mag_head.bias)
        _ = (residual_init, continuous_caes)

    @staticmethod
    def map_bounded(z: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        mapped = low + torch.sigmoid(z) * (high - low)
        return torch.minimum(torch.maximum(mapped, low), high)

    def _pack(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        if self.goal_input_scale != 1.0:
            goal = goal * self.goal_input_scale
        return torch.cat([obs, goal], dim=-1)

    def forward_logits(self, obs: torch.Tensor, goal: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(self._pack(obs, goal))
        return {
            "z_tp": self.thermal_head(h).squeeze(-1),
            "z_bat": self.battery_head(h).squeeze(-1),
            "z_caes": self.caes_head(h).squeeze(-1),
        }

    def _caes_from_hybrid(
        self,
        h: torch.Tensor,
        mode_mask: torch.Tensor,
        *,
        deterministic: bool,
        explore_noise_std: float,
        gumbel_tau: float,
        soft_mode_for_grad: bool,
    ) -> torch.Tensor:
        logits = self.caes_mode_head(h)
        legal = legalize_mode_mask(mode_mask)
        if legal.size(0) == 1 and logits.size(0) > 1:
            legal = legal.expand(logits.size(0), -1)
        logits = logits.masked_fill(~legal, -1.0e9)
        mag = torch.sigmoid(self.caes_mag_head(h).squeeze(-1))
        if explore_noise_std > 0 and not deterministic:
            mag = (mag + 0.5 * explore_noise_std * torch.randn_like(mag)).clamp(0.0, 1.0)
        use_gumbel = soft_mode_for_grad or (not deterministic and explore_noise_std > 0)
        if use_gumbel:
            gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-6, 1.0)))
            y_soft = torch.softmax((logits + gumbel) / max(float(gumbel_tau), 1e-3), dim=-1)
            idx = y_soft.argmax(dim=-1)
            y_hard = torch.nn.functional.one_hot(idx, 3).to(dtype=y_soft.dtype)
            onehot = y_hard + y_soft - y_soft.detach() if soft_mode_for_grad else y_hard
        else:
            idx = logits.argmax(dim=-1)
            onehot = torch.nn.functional.one_hot(idx, 3).to(dtype=logits.dtype)
        return u_from_mode_onehot_torch(onehot, mag)

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
        soft_mode_for_grad: bool = False,
    ) -> dict[str, torch.Tensor]:
        h = self.encoder(self._pack(obs, goal))
        u_tp = self.map_bounded(self.thermal_head(h).squeeze(-1), u_tp_low, u_tp_high)
        u_bat = self.map_bounded(self.battery_head(h).squeeze(-1), u_bat_low, u_bat_high)
        if self.hybrid_caes:
            u_caes = self._caes_from_hybrid(
                h,
                mode_mask,
                deterministic=deterministic,
                explore_noise_std=explore_noise_std,
                gumbel_tau=gumbel_tau,
                soft_mode_for_grad=soft_mode_for_grad,
            )
        else:
            u_caes = project_u_caes_torch(torch.tanh(self.caes_head(h).squeeze(-1)))
            if explore_noise_std > 0 and not deterministic:
                u_caes = project_u_caes_torch(
                    torch.clamp(u_caes + explore_noise_std * torch.randn_like(u_caes), -1.0, 1.0)
                )
        if explore_noise_std > 0 and not deterministic:
            u_tp = torch.clamp(u_tp + explore_noise_std * torch.randn_like(u_tp), u_tp_low, u_tp_high)
            u_bat = torch.clamp(u_bat + explore_noise_std * torch.randn_like(u_bat), u_bat_low, u_bat_high)
        u_caes = apply_mode_mask_to_u_torch(u_caes, mode_mask)
        return {
            "u_tp": u_tp,
            "u_battery": u_bat,
            "u_caes": u_caes,
        }


class LowLevelCritic(nn.Module):
    def __init__(self, obs_dim: int, goal_dim: int = 2, hidden: int = 256):
        super().__init__()
        act_dim = 3  # u_tp, u_battery, u_caes
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

    def _pack(self, obs, goal, u_tp, u_bat, u_caes):
        def _col(x):
            return x.unsqueeze(-1) if x.ndim == 1 else x

        return torch.cat(
            [obs, goal, _col(u_tp), _col(u_bat), _col(u_caes)],
            dim=-1,
        )

    def forward(self, obs, goal, u_tp, u_bat, u_caes):
        x = self._pack(obs, goal, u_tp, u_bat, u_caes)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_only(self, obs, goal, u_tp, u_bat, u_caes):
        x = self._pack(obs, goal, u_tp, u_bat, u_caes)
        return self.q1(x).squeeze(-1)
