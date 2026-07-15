"""有界参数化 Hybrid Actor。连续动作经 sigmoid 映射到动态可行区间；模式经 mask。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from actions import CaesMode, ModeMask


def _mlp(in_dim: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
    )


class HybridActor(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.encoder = _mlp(obs_dim, hidden)
        self.thermal_head = nn.Linear(hidden, 1)
        self.battery_head = nn.Linear(hidden, 1)
        self.mode_head = nn.Linear(hidden, 3)
        self.discharge_mag_head = nn.Linear(hidden, 1)
        self.charge_mag_head = nn.Linear(hidden, 1)

    def forward_logits(self, obs: torch.Tensor):
        h = self.encoder(obs)
        return {
            "z_tp": self.thermal_head(h).squeeze(-1),
            "z_bat": self.battery_head(h).squeeze(-1),
            "logits_mode": self.mode_head(h),
            "z_discharge": self.discharge_mag_head(h).squeeze(-1),
            "z_charge": self.charge_mag_head(h).squeeze(-1),
        }

    @staticmethod
    def map_bounded(z: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        # 映射后显式 clamp，避免 float32 越出动态边界导致校验失败
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
        """mode_mask: (B,3) bool，True=允许。"""
        out = self.forward_logits(obs)
        logits = out["logits_mode"].masked_fill(~mode_mask.bool(), -1e9)
        if deterministic:
            mode = torch.argmax(logits, dim=-1)
            mode_oh = F.one_hot(mode, num_classes=3).float()
        else:
            mode_oh = F.gumbel_softmax(logits, tau=gumbel_tau, hard=True, dim=-1)
            mode = torch.argmax(mode_oh, dim=-1)

        u_tp = self.map_bounded(out["z_tp"], u_tp_low, u_tp_high)
        u_bat = self.map_bounded(out["z_bat"], u_bat_low, u_bat_high)
        mag_d = torch.sigmoid(out["z_discharge"])
        mag_c = torch.sigmoid(out["z_charge"])
        # 按模式选择幅值；IDLE -> 0
        mag = mode_oh[:, 0] * mag_d + mode_oh[:, 2] * mag_c

        if explore_noise_std > 0 and not deterministic:
            # 噪声保持在动态范围内：对连续变量在区间内扰动后重新夹紧到 [low,high]
            u_tp = torch.clamp(u_tp + explore_noise_std * torch.randn_like(u_tp), u_tp_low, u_tp_high)
            u_bat = torch.clamp(u_bat + explore_noise_std * torch.randn_like(u_bat), u_bat_low, u_bat_high)
            mag = torch.clamp(mag + explore_noise_std * torch.randn_like(mag), 0.0, 1.0)
            # 噪声后 IDLE 幅值仍记 0
            mag = torch.where(mode == int(CaesMode.IDLE), torch.zeros_like(mag), mag)

        return {
            "u_tp": u_tp,
            "u_battery": u_bat,
            "caes_mode": mode,
            "caes_mode_oh": mode_oh,
            "caes_magnitude": mag,
            "logits_mode": logits,
        }

    def act_numpy(self, obs, feasible, deterministic: bool = True, device="cpu", explore_noise_std: float = 0.0):
        import numpy as np
        self.eval()
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=device).view(1, -1)
            mask = torch.as_tensor(feasible.mode_mask.as_bool_array(), dtype=torch.bool, device=device).view(1, 3)
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
        return {
            "u_tp": np.asarray([float(out["u_tp"][0].cpu())], dtype=np.float32),
            "u_battery": np.asarray([float(out["u_battery"][0].cpu())], dtype=np.float32),
            "caes_mode": int(out["caes_mode"][0].cpu()),
            "caes_magnitude": np.asarray([float(out["caes_magnitude"][0].cpu())], dtype=np.float32),
        }
