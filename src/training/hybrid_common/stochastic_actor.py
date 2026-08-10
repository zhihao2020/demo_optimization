"""随机 Actor：连续对角高斯 → (u_tp, u_battery, u_caes)。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from actions.caes_u import apply_mode_mask_to_u_torch, physical_dict, project_u_caes_torch


def _mlp(in_dim: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
    )


class HybridStochasticActor(nn.Module):
    """随机 Actor：火电/电池有界高斯 + u_caes 在 [-1,1] 上投影到合法三段。"""

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.encoder = _mlp(obs_dim, hidden)
        self.tp_mean = nn.Linear(hidden, 1)
        self.bat_mean = nn.Linear(hidden, 1)
        self.caes_mean = nn.Linear(hidden, 1)
        self.tp_log_std = nn.Linear(hidden, 1)
        self.bat_log_std = nn.Linear(hidden, 1)
        self.caes_log_std = nn.Linear(hidden, 1)
        nn.init.constant_(self.tp_mean.bias, 2.0)
        nn.init.zeros_(self.bat_mean.bias)
        nn.init.zeros_(self.caes_mean.bias)

    def _heads(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(obs)
        return {
            "mu_tp": self.tp_mean(h).squeeze(-1),
            "mu_bat": self.bat_mean(h).squeeze(-1),
            "mu_caes": self.caes_mean(h).squeeze(-1),
            "ls_tp": self.tp_log_std(h).squeeze(-1).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX),
            "ls_bat": self.bat_log_std(h).squeeze(-1).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX),
            "ls_caes": self.caes_log_std(h).squeeze(-1).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX),
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
    ) -> dict[str, torch.Tensor]:
        h = self._heads(obs)
        if deterministic:
            z_tp, z_bat, z_caes = h["mu_tp"], h["mu_bat"], h["mu_caes"]
            log_prob = torch.zeros_like(z_tp)
            entropy = torch.zeros_like(z_tp)
        else:
            dist_tp = Normal(h["mu_tp"], h["ls_tp"].exp())
            dist_bat = Normal(h["mu_bat"], h["ls_bat"].exp())
            dist_caes = Normal(h["mu_caes"], h["ls_caes"].exp())
            z_tp = dist_tp.rsample()
            z_bat = dist_bat.rsample()
            z_caes = dist_caes.rsample()
            log_prob = dist_tp.log_prob(z_tp) + dist_bat.log_prob(z_bat) + dist_caes.log_prob(z_caes)
            entropy = dist_tp.entropy() + dist_bat.entropy() + dist_caes.entropy()

        u_tp = self.map_bounded(z_tp, u_tp_low, u_tp_high)
        u_bat = self.map_bounded(z_bat, u_bat_low, u_bat_high)
        u_caes = apply_mode_mask_to_u_torch(project_u_caes_torch(torch.tanh(z_caes)), mode_mask)
        return {
            "u_tp": u_tp,
            "u_battery": u_bat,
            "u_caes": u_caes,
            "log_prob": log_prob,
            "entropy": entropy,
        }

    def evaluate(
        self,
        obs: torch.Tensor,
        u_tp: torch.Tensor,
        u_battery: torch.Tensor,
        u_caes: torch.Tensor,
        u_tp_low: torch.Tensor,
        u_tp_high: torch.Tensor,
        u_bat_low: torch.Tensor,
        u_bat_high: torch.Tensor,
        mode_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Approximate log_prob of executed continuous actions (post-projection)."""
        _ = (mode_mask,)
        h = self._heads(obs)
        # inverse map is ill-defined after project; use current policy sample path for SAC
        dist_tp = Normal(h["mu_tp"], h["ls_tp"].exp())
        dist_bat = Normal(h["mu_bat"], h["ls_bat"].exp())
        dist_caes = Normal(h["mu_caes"], h["ls_caes"].exp())
        # map action back to pre-sigmoid roughly for tp/bat
        def inv_bound(u, lo, hi):
            span = (hi - lo).clamp_min(1e-6)
            y = ((u - lo) / span).clamp(1e-6, 1 - 1e-6)
            return torch.log(y) - torch.log1p(-y)

        z_tp = inv_bound(u_tp, u_tp_low, u_tp_high)
        z_bat = inv_bound(u_battery, u_bat_low, u_bat_high)
        # u_caes is projected; use atanh of clipped for rough density
        z_caes = torch.atanh(u_caes.clamp(-0.999, 0.999))
        log_prob = dist_tp.log_prob(z_tp) + dist_bat.log_prob(z_bat) + dist_caes.log_prob(z_caes)
        entropy = dist_tp.entropy() + dist_bat.entropy() + dist_caes.entropy()
        return {"log_prob": log_prob, "entropy": entropy}

    def act_numpy(self, obs, feasible, deterministic: bool = True, device="cpu"):
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
            )
        return physical_dict(
            float(out["u_tp"][0].cpu()),
            float(out["u_battery"][0].cpu()),
            float(out["u_caes"][0].cpu()),
        )
