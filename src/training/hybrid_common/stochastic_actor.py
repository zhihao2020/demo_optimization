"""随机 Actor：火电/电池连续高斯 + 压空 (mode, magnitude)。"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from actions.caes_u import (
    apply_mode_mask_to_u_torch,
    clamp_u_caes_to_spec,
    gumbel_mode_onehot,
    legalize_mode_mask,
    mag_from_u_torch,
    mask_mode_logits,
    mode_index_from_u_torch,
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


class HybridStochasticActor(nn.Module):
    """SAC actor：有界火电/电池 + 压空参数化动作 (mode, mag)。

    ``parameterized_caes=False`` 保留旧 tanh 投影，仅作消融。
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        obs_dim: int,
        hidden: int = 256,
        *,
        parameterized_caes: bool = True,
        gumbel_tau: float = 1.0,
    ):
        super().__init__()
        self.parameterized_caes = bool(parameterized_caes)
        self.gumbel_tau = float(gumbel_tau)
        self.obs_dim = int(obs_dim)
        self.encoder = _mlp(obs_dim, hidden)
        self.tp_mean = nn.Linear(hidden, 1)
        self.bat_mean = nn.Linear(hidden, 1)
        self.tp_log_std = nn.Linear(hidden, 1)
        self.bat_log_std = nn.Linear(hidden, 1)
        nn.init.constant_(self.tp_mean.bias, 2.0)
        nn.init.zeros_(self.bat_mean.bias)
        if self.parameterized_caes:
            self.caes_mode_head = nn.Linear(hidden, 3)
            self.caes_mag_mean = nn.Linear(hidden, 1)
            self.caes_mag_log_std = nn.Linear(hidden, 1)
            nn.init.zeros_(self.caes_mode_head.weight)
            nn.init.zeros_(self.caes_mag_mean.weight)
            nn.init.constant_(self.caes_mode_head.bias, 0.0)
            self.caes_mode_head.bias.data[1] = 0.4
            nn.init.zeros_(self.caes_mag_mean.bias)
        else:
            self.caes_mean = nn.Linear(hidden, 1)
            self.caes_log_std = nn.Linear(hidden, 1)
            nn.init.zeros_(self.caes_mean.bias)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def _heads(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self._encode(obs)
        out = {
            "h": h,
            "mu_tp": self.tp_mean(h).squeeze(-1),
            "mu_bat": self.bat_mean(h).squeeze(-1),
            "ls_tp": self.tp_log_std(h).squeeze(-1).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX),
            "ls_bat": self.bat_log_std(h).squeeze(-1).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX),
        }
        if self.parameterized_caes:
            out["mode_logits"] = self.caes_mode_head(h)
            out["mu_mag"] = self.caes_mag_mean(h).squeeze(-1)
            out["ls_mag"] = self.caes_mag_log_std(h).squeeze(-1).clamp(
                self.LOG_STD_MIN, self.LOG_STD_MAX
            )
        else:
            out["mu_caes"] = self.caes_mean(h).squeeze(-1)
            out["ls_caes"] = self.caes_log_std(h).squeeze(-1).clamp(
                self.LOG_STD_MIN, self.LOG_STD_MAX
            )
        return out

    @staticmethod
    def map_bounded(z: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        mapped = low + torch.sigmoid(z) * (high - low)
        return torch.minimum(torch.maximum(mapped, low), high)

    def _sample_gauss(self, mu: torch.Tensor, log_std: torch.Tensor, *, deterministic: bool):
        dist = Normal(mu, log_std.exp())
        z = mu if deterministic else dist.rsample()
        log_prob = dist.log_prob(z)
        entropy = dist.entropy()
        return z, log_prob, entropy

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
        z_tp, lp_tp, ent_tp = self._sample_gauss(h["mu_tp"], h["ls_tp"], deterministic=deterministic)
        z_bat, lp_bat, ent_bat = self._sample_gauss(
            h["mu_bat"], h["ls_bat"], deterministic=deterministic
        )
        u_tp = self.map_bounded(z_tp, u_tp_low, u_tp_high)
        u_bat = self.map_bounded(z_bat, u_bat_low, u_bat_high)
        if self.parameterized_caes:
            logits = mask_mode_logits(h["mode_logits"], mode_mask)
            onehot, idx = gumbel_mode_onehot(
                logits,
                self.gumbel_tau,
                deterministic=deterministic,
                soft_for_grad=not deterministic,
            )
            z_mag, lp_mag, ent_mag = self._sample_gauss(
                h["mu_mag"], h["ls_mag"], deterministic=deterministic
            )
            mag = torch.sigmoid(z_mag)
            u_caes = u_from_mode_onehot_torch(onehot, mag)
            log_mode = torch.log_softmax(logits, dim=-1).gather(-1, idx.unsqueeze(-1)).squeeze(-1)
            probs = torch.softmax(logits, dim=-1).clamp_min(1e-8)
            ent_mode = -(probs * probs.log()).sum(dim=-1)
            log_prob = lp_tp + lp_bat + lp_mag + log_mode
            entropy = ent_tp + ent_bat + ent_mag + ent_mode
        else:
            z_caes, lp_caes, ent_caes = self._sample_gauss(
                h["mu_caes"], h["ls_caes"], deterministic=deterministic
            )
            u_caes = project_u_caes_torch(torch.tanh(z_caes))
            log_prob = lp_tp + lp_bat + lp_caes
            entropy = ent_tp + ent_bat + ent_caes
        mask_b = legalize_mode_mask(mode_mask)
        if mask_b.size(0) == 1 and u_caes.size(0) > 1:
            mask_b = mask_b.expand(u_caes.size(0), -1)
        u_caes = apply_mode_mask_to_u_torch(u_caes, mask_b)
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
        """Approximate log_prob of executed actions."""
        h = self._heads(obs)
        dist_tp = Normal(h["mu_tp"], h["ls_tp"].exp())
        dist_bat = Normal(h["mu_bat"], h["ls_bat"].exp())

        def inv_bound(u, lo, hi):
            span = (hi - lo).clamp_min(1e-6)
            y = ((u - lo) / span).clamp(1e-6, 1 - 1e-6)
            return torch.log(y) - torch.log1p(-y)

        z_tp = inv_bound(u_tp, u_tp_low, u_tp_high)
        z_bat = inv_bound(u_battery, u_bat_low, u_bat_high)
        log_prob = dist_tp.log_prob(z_tp) + dist_bat.log_prob(z_bat)
        entropy = dist_tp.entropy() + dist_bat.entropy()
        if self.parameterized_caes:
            logits = mask_mode_logits(h["mode_logits"], mode_mask)
            idx = mode_index_from_u_torch(u_caes)
            mag = mag_from_u_torch(u_caes).clamp(1e-6, 1 - 1e-6)
            z_mag = torch.log(mag) - torch.log1p(-mag)
            dist_mag = Normal(h["mu_mag"], h["ls_mag"].exp())
            log_mode = torch.log_softmax(logits, dim=-1).gather(-1, idx.unsqueeze(-1)).squeeze(-1)
            log_prob = log_prob + dist_mag.log_prob(z_mag) + log_mode
            probs = torch.softmax(logits, dim=-1).clamp_min(1e-8)
            entropy = entropy + dist_mag.entropy() - (probs * probs.log()).sum(dim=-1)
        else:
            dist_caes = Normal(h["mu_caes"], h["ls_caes"].exp())
            z_caes = torch.atanh(u_caes.clamp(-0.999, 0.999))
            log_prob = log_prob + dist_caes.log_prob(z_caes)
            entropy = entropy + dist_caes.entropy()
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
        u_caes, _ = clamp_u_caes_to_spec(float(out["u_caes"][0].cpu()), feasible)
        return physical_dict(
            float(out["u_tp"][0].cpu()),
            float(out["u_battery"][0].cpu()),
            float(u_caes),
        )
