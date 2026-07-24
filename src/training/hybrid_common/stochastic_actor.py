"""随机混合 Actor：连续对角高斯（sigmoid 映射到动态边界）+ CAES 模式 Categorical。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from actions import CaesMode


def _mlp(in_dim: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
    )


def _logit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x) - torch.log1p(-x)


class HybridStochasticActor(nn.Module):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.encoder = _mlp(obs_dim, hidden)
        self.tp_mean = nn.Linear(hidden, 1)
        self.bat_mean = nn.Linear(hidden, 1)
        self.mode_head = nn.Linear(hidden, 3)
        self.d_mag_mean = nn.Linear(hidden, 1)
        self.c_mag_mean = nn.Linear(hidden, 1)
        self.tp_log_std = nn.Linear(hidden, 1)
        self.bat_log_std = nn.Linear(hidden, 1)
        self.d_mag_log_std = nn.Linear(hidden, 1)
        self.c_mag_log_std = nn.Linear(hidden, 1)

    def _heads(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(obs)
        return {
            "mu_tp": self.tp_mean(h).squeeze(-1),
            "mu_bat": self.bat_mean(h).squeeze(-1),
            "logits_mode": self.mode_head(h),
            "mu_d": self.d_mag_mean(h).squeeze(-1),
            "mu_c": self.c_mag_mean(h).squeeze(-1),
            "ls_tp": self.tp_log_std(h).squeeze(-1).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX),
            "ls_bat": self.bat_log_std(h).squeeze(-1).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX),
            "ls_d": self.d_mag_log_std(h).squeeze(-1).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX),
            "ls_c": self.c_mag_log_std(h).squeeze(-1).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX),
        }

    @staticmethod
    def map_bounded(z: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        mapped = low + torch.sigmoid(z) * (high - low)
        return torch.minimum(torch.maximum(mapped, low), high)

    @staticmethod
    def _squash_log_prob(
        dist: Normal, z: torch.Tensor, low: torch.Tensor, high: torch.Tensor
    ) -> torch.Tensor:
        """z -> sigmoid -> [low, high] 的密度修正。"""
        s = torch.sigmoid(z)
        log_det = torch.log(s * (1.0 - s) + 1e-6) + torch.log((high - low).clamp_min(1e-6))
        return dist.log_prob(z) - log_det

    def forward_action(
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
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        h = self._heads(obs)
        logits = h["logits_mode"].masked_fill(
            ~mode_mask.bool(), torch.finfo(h["logits_mode"].dtype).min / 2
        )
        if not torch.isfinite(logits).all():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            logits = logits.masked_fill(
                ~mode_mask.bool(), torch.finfo(logits.dtype).min / 2
            )
        mode_dist = Categorical(logits=logits)
        if deterministic:
            mode = torch.argmax(logits, dim=-1)
        else:
            mode = mode_dist.sample()
        mode_oh = F.one_hot(mode, num_classes=3).float()

        def _cont(mu, ls, low, high):
            mu = torch.nan_to_num(mu, nan=0.0)
            ls = torch.nan_to_num(ls, nan=-1.0).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
            std = ls.exp()
            dist = Normal(mu, std)
            z = mu if deterministic else dist.rsample()
            u = self.map_bounded(z, low, high)
            lp = self._squash_log_prob(dist, z, low, high)
            ent = dist.entropy()
            return u, lp, ent, z, dist

        zero = torch.zeros_like(u_tp_low)
        one = torch.ones_like(u_tp_low)
        u_tp, lp_tp, e_tp, _, _ = _cont(h["mu_tp"], h["ls_tp"], u_tp_low, u_tp_high)
        u_bat, lp_bat, e_bat, _, _ = _cont(h["mu_bat"], h["ls_bat"], u_bat_low, u_bat_high)
        mag_d, lp_d, e_d, _, _ = _cont(h["mu_d"], h["ls_d"], zero, one)
        mag_c, lp_c, e_c, _, _ = _cont(h["mu_c"], h["ls_c"], zero, one)

        mag = torch.where(
            mode == int(CaesMode.DISCHARGE),
            mag_d,
            torch.where(mode == int(CaesMode.CHARGE), mag_c, torch.zeros_like(mag_d)),
        )
        lp_mag = torch.where(
            mode == int(CaesMode.DISCHARGE),
            lp_d,
            torch.where(mode == int(CaesMode.CHARGE), lp_c, torch.zeros_like(lp_d)),
        )
        e_mag = torch.where(
            mode == int(CaesMode.DISCHARGE),
            e_d,
            torch.where(mode == int(CaesMode.CHARGE), e_c, torch.zeros_like(e_d)),
        )
        lp_mode = mode_dist.log_prob(mode)
        log_prob = lp_tp + lp_bat + lp_mag + lp_mode
        entropy = e_tp + e_bat + e_mag + mode_dist.entropy()
        return {
            "u_tp": u_tp,
            "u_battery": u_bat,
            "caes_mode": mode,
            "caes_mode_oh": mode_oh,
            "caes_magnitude": mag,
            "log_prob": log_prob,
            "entropy": entropy,
            "logits_mode": logits,
        }

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        u_tp: torch.Tensor,
        u_battery: torch.Tensor,
        caes_mode: torch.Tensor,
        caes_magnitude: torch.Tensor,
        u_tp_low: torch.Tensor,
        u_tp_high: torch.Tensor,
        u_bat_low: torch.Tensor,
        u_bat_high: torch.Tensor,
        mode_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """给定动作的 log_prob / entropy（PPO 用）。"""
        h = self._heads(obs)
        logits = h["logits_mode"].masked_fill(
            ~mode_mask.bool(), torch.finfo(h["logits_mode"].dtype).min / 2
        )
        if not torch.isfinite(logits).all():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            logits = logits.masked_fill(
                ~mode_mask.bool(), torch.finfo(logits.dtype).min / 2
            )
        mode_dist = Categorical(logits=logits)
        mode = caes_mode.long()
        lp_mode = mode_dist.log_prob(mode)

        def _lp_bounded(mu, ls, u, low, high):
            std = ls.exp()
            dist = Normal(mu, std)
            span = (high - low).clamp_min(1e-6)
            unit = ((u - low) / span).clamp(1e-6, 1.0 - 1e-6)
            z = _logit(unit)
            return self._squash_log_prob(dist, z, low, high), dist.entropy()

        zero = torch.zeros_like(u_tp_low)
        one = torch.ones_like(u_tp_low)
        lp_tp, e_tp = _lp_bounded(h["mu_tp"], h["ls_tp"], u_tp, u_tp_low, u_tp_high)
        lp_bat, e_bat = _lp_bounded(h["mu_bat"], h["ls_bat"], u_battery, u_bat_low, u_bat_high)
        lp_d, e_d = _lp_bounded(h["mu_d"], h["ls_d"], caes_magnitude, zero, one)
        lp_c, e_c = _lp_bounded(h["mu_c"], h["ls_c"], caes_magnitude, zero, one)
        lp_mag = torch.where(
            mode == int(CaesMode.DISCHARGE),
            lp_d,
            torch.where(mode == int(CaesMode.CHARGE), lp_c, torch.zeros_like(lp_d)),
        )
        e_mag = torch.where(
            mode == int(CaesMode.DISCHARGE),
            e_d,
            torch.where(mode == int(CaesMode.CHARGE), e_c, torch.zeros_like(e_d)),
        )
        log_prob = lp_tp + lp_bat + lp_mag + lp_mode
        entropy = e_tp + e_bat + e_mag + mode_dist.entropy()
        mode_oh = F.one_hot(mode, num_classes=3).float()
        return {
            "log_prob": log_prob,
            "entropy": entropy,
            "caes_mode_oh": mode_oh,
        }

    def act_numpy(self, obs, feasible, deterministic: bool = True, device="cpu"):
        self.eval()
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=device).view(1, -1)
            mask = torch.as_tensor(
                feasible.mode_mask.as_bool_array(), dtype=torch.bool, device=device
            ).view(1, 3)
            out = self.forward_action(
                o,
                torch.tensor([feasible.u_tp_low], device=device),
                torch.tensor([feasible.u_tp_high], device=device),
                torch.tensor([feasible.u_battery_low], device=device),
                torch.tensor([feasible.u_battery_high], device=device),
                mask,
                deterministic=deterministic,
            )
        return {
            "u_tp": np.asarray([float(out["u_tp"][0].cpu())], dtype=np.float32),
            "u_battery": np.asarray([float(out["u_battery"][0].cpu())], dtype=np.float32),
            "caes_mode": int(out["caes_mode"][0].cpu()),
            "caes_magnitude": np.asarray([float(out["caes_magnitude"][0].cpu())], dtype=np.float32),
            "log_prob": float(out["log_prob"][0].cpu()),
        }
