"""TD3 actor：有界火电/电池 + 物理 u_caes。"""

from __future__ import annotations

import torch
import torch.nn as nn

from actions.caes_u import apply_mode_mask_to_u_torch, physical_dict, project_u_caes_torch


def _mlp(in_dim: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
    )


class HybridActor(nn.Module):
    """Actor：输出 (u_tp, u_battery, u_caes)。类名保留以免大面积改 import。"""

    def __init__(self, obs_dim: int, hidden: int = 256, *, continuous_caes: bool = True):
        super().__init__()
        _ = continuous_caes
        self.encoder = _mlp(obs_dim, hidden)
        self.thermal_head = nn.Linear(hidden, 1)
        self.battery_head = nn.Linear(hidden, 1)
        self.caes_head = nn.Linear(hidden, 1)
        nn.init.constant_(self.thermal_head.bias, 2.0)
        nn.init.constant_(self.battery_head.bias, 0.0)
        nn.init.zeros_(self.caes_head.bias)

    def forward_logits(self, obs: torch.Tensor):
        h = self.encoder(obs)
        return {
            "z_tp": self.thermal_head(h).squeeze(-1),
            "z_bat": self.battery_head(h).squeeze(-1),
            "z_caes": self.caes_head(h).squeeze(-1),
        }

    @staticmethod
    def map_bounded(z: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        mapped = low + torch.sigmoid(z) * (high - low)
        return torch.minimum(torch.maximum(mapped, low), high)

    def act(
        self,
        obs: torch.Tensor,
        u_tp_low: torch.Tensor,
        u_tp_high: torch.Tensor,
        u_bat_low: torch.Tensor,
        u_bat_high: torch.Tensor,
        mode_mask: torch.Tensor,
        *,
        deterministic: bool = False,
        gumbel_tau: float = 1.0,
        explore_noise_std: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        _ = gumbel_tau
        out = self.forward_logits(obs)
        u_tp = self.map_bounded(out["z_tp"], u_tp_low, u_tp_high)
        u_bat = self.map_bounded(out["z_bat"], u_bat_low, u_bat_high)
        u_caes = project_u_caes_torch(torch.tanh(out["z_caes"]))
        if explore_noise_std > 0 and not deterministic:
            u_tp = torch.clamp(u_tp + explore_noise_std * torch.randn_like(u_tp), u_tp_low, u_tp_high)
            u_bat = torch.clamp(u_bat + explore_noise_std * torch.randn_like(u_bat), u_bat_low, u_bat_high)
            u_caes = project_u_caes_torch(
                torch.clamp(u_caes + explore_noise_std * torch.randn_like(u_caes), -1.0, 1.0)
            )
        u_caes = apply_mode_mask_to_u_torch(u_caes, mode_mask)
        return {
            "u_tp": u_tp,
            "u_battery": u_bat,
            "u_caes": u_caes,
        }

    def act_numpy(self, obs, feasible, deterministic: bool = True, device="cpu", explore_noise_std: float = 0.0):
        import numpy as np

        self.eval()
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=device).view(1, -1)
            mask = torch.as_tensor(
                feasible.mode_mask.as_bool_array(), dtype=torch.bool, device=device
            ).view(1, 3)
            out = self.act(
                o,
                torch.tensor([feasible.u_tp_low], device=device),
                torch.tensor([feasible.u_tp_high], device=device),
                torch.tensor([feasible.u_battery_low], device=device),
                torch.tensor([feasible.u_battery_high], device=device),
                mask,
                deterministic=deterministic,
                explore_noise_std=0.0 if deterministic else explore_noise_std,
            )
        return physical_dict(
            float(out["u_tp"][0].cpu()),
            float(out["u_battery"][0].cpu()),
            float(out["u_caes"][0].cpu()),
        )
