"""TD3 actor：有界火电/电池 + 压空 (mode, magnitude)。"""

from __future__ import annotations

import torch
import torch.nn as nn

from actions.caes_u import (
    apply_mode_mask_to_u_torch,
    clamp_u_caes_to_spec,
    gumbel_mode_onehot,
    legalize_mode_mask,
    mask_mode_logits,
    physical_dict,
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


class HybridActor(nn.Module):
    """Actor：输出 (u_tp, u_battery, u_caes)。类名保留以免大面积改 import。

    默认 ``parameterized_caes=True``：模式头 + 幅值头。``False`` 为旧 tanh 投影消融。
    """

    def __init__(
        self,
        obs_dim: int,
        hidden: int = 256,
        *,
        continuous_caes: bool = True,
        parameterized_caes: bool = True,
        gumbel_tau: float = 1.0,
    ):
        super().__init__()
        _ = continuous_caes
        self.parameterized_caes = bool(parameterized_caes)
        self.gumbel_tau = float(gumbel_tau)
        self.obs_dim = int(obs_dim)
        self.encoder = _mlp(obs_dim, hidden)
        self.thermal_head = nn.Linear(hidden, 1)
        self.battery_head = nn.Linear(hidden, 1)
        nn.init.constant_(self.thermal_head.bias, 2.0)
        nn.init.constant_(self.battery_head.bias, 0.0)
        if self.parameterized_caes:
            self.caes_mode_head = nn.Linear(hidden, 3)
            self.caes_mag_head = nn.Linear(hidden, 1)
            nn.init.zeros_(self.caes_mode_head.weight)
            nn.init.zeros_(self.caes_mag_head.weight)
            nn.init.constant_(self.caes_mode_head.bias, 0.0)
            self.caes_mode_head.bias.data[1] = 0.4
            nn.init.zeros_(self.caes_mag_head.bias)
        else:
            self.caes_head = nn.Linear(hidden, 1)
            nn.init.zeros_(self.caes_head.bias)

    def forward_logits(self, obs: torch.Tensor):
        h = self.encoder(obs)
        out = {
            "z_tp": self.thermal_head(h).squeeze(-1),
            "z_bat": self.battery_head(h).squeeze(-1),
        }
        if self.parameterized_caes:
            out["mode_logits"] = self.caes_mode_head(h)
            out["z_mag"] = self.caes_mag_head(h).squeeze(-1)
        else:
            out["z_caes"] = self.caes_head(h).squeeze(-1)
        return out

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
        tau = gumbel_tau if gumbel_tau is not None else self.gumbel_tau
        out = self.forward_logits(obs)
        u_tp = self.map_bounded(out["z_tp"], u_tp_low, u_tp_high)
        u_bat = self.map_bounded(out["z_bat"], u_bat_low, u_bat_high)
        if self.parameterized_caes:
            logits = mask_mode_logits(out["mode_logits"], mode_mask)
            onehot, _idx = gumbel_mode_onehot(
                logits,
                tau,
                deterministic=deterministic,
                soft_for_grad=not deterministic,
            )
            mag = torch.sigmoid(out["z_mag"])
            if explore_noise_std > 0 and not deterministic:
                mag = (mag + 0.5 * explore_noise_std * torch.randn_like(mag)).clamp(0.0, 1.0)
            u_caes = u_from_mode_onehot_torch(onehot, mag)
        else:
            u_caes = project_u_caes_torch(torch.tanh(out["z_caes"]))
            if explore_noise_std > 0 and not deterministic:
                u_caes = project_u_caes_torch(
                    torch.clamp(u_caes + explore_noise_std * torch.randn_like(u_caes), -1.0, 1.0)
                )
        if explore_noise_std > 0 and not deterministic:
            u_tp = torch.clamp(u_tp + explore_noise_std * torch.randn_like(u_tp), u_tp_low, u_tp_high)
            u_bat = torch.clamp(u_bat + explore_noise_std * torch.randn_like(u_bat), u_bat_low, u_bat_high)
        mask_b = legalize_mode_mask(mode_mask)
        if mask_b.size(0) == 1 and u_caes.size(0) > 1:
            mask_b = mask_b.expand(u_caes.size(0), -1)
        u_caes = apply_mode_mask_to_u_torch(u_caes, mask_b)
        return {
            "u_tp": u_tp,
            "u_battery": u_bat,
            "u_caes": u_caes,
        }

    def act_numpy(self, obs, feasible, deterministic: bool = True, device="cpu", explore_noise_std: float = 0.0):
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
        u_caes, _ = clamp_u_caes_to_spec(float(out["u_caes"][0].cpu()), feasible)
        return physical_dict(
            float(out["u_tp"][0].cpu()),
            float(out["u_battery"][0].cpu()),
            float(u_caes),
        )
