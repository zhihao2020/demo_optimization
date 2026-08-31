"""TD3 actor：有界火电/电池 + 压空 (mode, magnitude)。"""

from __future__ import annotations

import torch
import torch.nn as nn

from actions.caes_u import (
    CHARGE_HI,
    CHARGE_LO,
    DISCHARGE_HI,
    DISCHARGE_LO,
    apply_mode_mask_to_u_torch,
    caes_intervals_from_feasible,
    gumbel_mode_onehot,
    legalize_mode_mask,
    mask_mode_logits,
    mode_from_u,
    physical_dict,
    project_u_caes_torch,
    snap_to_interval_endpoint,
    u_from_mode_onehot_dynamic,
    u_from_mode_onehot_torch,
)
from actions.joint_support import coupling_from_feasible, decode_joint_numpy


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
        use_dynamic_support: bool = True,
        gumbel_tau: float = 1.0,
    ):
        super().__init__()
        _ = continuous_caes
        self.parameterized_caes = bool(parameterized_caes)
        self.use_dynamic_support = bool(use_dynamic_support)
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
            nn.init.zeros_(self.caes_mode_head.bias)
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

    @staticmethod
    def _bound(value, batch: int, fill: float, device) -> torch.Tensor:
        if value is None:
            return torch.full((batch,), float(fill), device=device)
        t = torch.as_tensor(value, dtype=torch.float32, device=device).reshape(-1)
        if t.numel() == 1 and batch > 1:
            t = t.expand(batch)
        return t

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
        dis_lo=None,
        dis_hi=None,
        chg_lo=None,
        chg_hi=None,
        use_dynamic_support: bool | None = None,
        grid_residual=None,
        grid_g_min=None,
        grid_g_max=None,
        p_cap_thermal=None,
        p_cap_battery=None,
        p_cap_caes=None,
    ) -> dict[str, torch.Tensor]:
        tau = gumbel_tau if gumbel_tau is not None else self.gumbel_tau
        if use_dynamic_support is None:
            use_dynamic_support = self.use_dynamic_support
        b = int(obs.size(0))
        device = obs.device
        out = self.forward_logits(obs)
        u_tp = self.map_bounded(out["z_tp"], u_tp_low, u_tp_high)
        u_bat = self.map_bounded(out["z_bat"], u_bat_low, u_bat_high)
        d_lo = self._bound(dis_lo, b, DISCHARGE_LO, device)
        d_hi = self._bound(dis_hi, b, DISCHARGE_HI, device)
        c_lo = self._bound(chg_lo, b, CHARGE_LO, device)
        c_hi = self._bound(chg_hi, b, CHARGE_HI, device)
        onehot = None
        mag = None
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
                mag = (mag + explore_noise_std * torch.randn_like(mag)).clamp(0.0, 1.0)
            if use_dynamic_support:
                u_caes = u_from_mode_onehot_dynamic(onehot, mag, d_lo, d_hi, c_lo, c_hi)
            else:
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
        if grid_residual is not None:
            from actions.joint_support import decode_joint_torch

            u_tp, u_bat = decode_joint_torch(
                u_tp,
                u_bat,
                u_caes,
                u_tp_low,
                u_tp_high,
                u_bat_low,
                u_bat_high,
                self._bound(grid_residual, b, 0.0, device),
                self._bound(grid_g_min, b, -5.0e8, device),
                self._bound(grid_g_max, b, 5.0e8, device),
                self._bound(p_cap_thermal, b, 1.5e8, device),
                self._bound(p_cap_battery, b, 1.0e8, device),
                self._bound(p_cap_caes, b, 1.5e8, device),
            )
        packed = {
            "u_tp": u_tp,
            "u_battery": u_bat,
            "u_caes": u_caes,
        }
        if onehot is not None:
            packed["mode_onehot"] = onehot
            packed["mag"] = mag
            packed["caes_mode_onehot"] = onehot
            packed["caes_magnitude"] = mag
        return packed

    def act_numpy(self, obs, feasible, deterministic: bool = True, device="cpu", explore_noise_std: float = 0.0):
        self.eval()
        iv = caes_intervals_from_feasible(feasible)
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
                dis_lo=iv["u_caes_discharge_low"],
                dis_hi=iv["u_caes_discharge_high"],
                chg_lo=iv["u_caes_charge_low"],
                chg_hi=iv["u_caes_charge_high"],
                use_dynamic_support=self.use_dynamic_support,
            )
        u_caes = float(out["u_caes"][0].cpu())
        u_tp = float(out["u_tp"][0].cpu())
        u_bat = float(out["u_battery"][0].cpu())
        original_u = float(u_caes)
        snapped = False
        mode = mode_from_u(u_caes)
        if "caes_mode_onehot" in out:
            oh = out["caes_mode_onehot"][0].detach().cpu().numpy()
            mode = int(oh.argmax())
        span = None
        if mode == 0:
            span = feasible.u_caes_discharge
        elif mode == 2:
            span = feasible.u_caes_charge
        if span is not None:
            lo, hi = min(float(span[0]), float(span[1])), max(float(span[0]), float(span[1]))
            u_caes, snapped = snap_to_interval_endpoint(u_caes, lo, hi)
        ctx = coupling_from_feasible(feasible)
        if ctx is not None:
            u_tp, u_bat = decode_joint_numpy(
                ctx,
                float(feasible.u_tp_low),
                float(feasible.u_tp_high),
                float(feasible.u_battery_low),
                float(feasible.u_battery_high),
                float(u_caes),
                u_tp,
                u_bat,
            )
        packed = physical_dict(u_tp, u_bat, float(u_caes))
        packed["caes_endpoint_snapped"] = bool(snapped)
        packed["caes_endpoint_snap_delta"] = float(u_caes - original_u)
        if "caes_mode_onehot" in out:
            packed["caes_mode_onehot"] = out["caes_mode_onehot"][0].detach().cpu().numpy()
            packed["caes_magnitude"] = float(out["caes_magnitude"][0].cpu())
        return packed
