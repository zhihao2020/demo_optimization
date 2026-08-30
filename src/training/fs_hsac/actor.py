"""FS-HSAC actor: state-dependent hybrid support over CAES modes and magnitudes."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from actions.caes_u import physical_dict
from training.fs_hsac.action_support import (
    MODE_CHARGE,
    MODE_DISCHARGE,
    MODE_IDLE,
    one_hot_modes,
    interval_log_jacobian,
    sigmoid_log_jacobian,
    support_from_feasible_batch,
)


def _mlp(in_dim: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
    )


class FSHSACActor(nn.Module):
    """Feasible-support hybrid actor.

    Continuous heads map into state-dependent intervals with correct Jacobians.
    Mode head is a masked categorical. Idle is a point mass (no continuous mag entropy).
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.encoder = _mlp(obs_dim, hidden)
        self.tp_mean = nn.Linear(hidden, 1)
        self.bat_mean = nn.Linear(hidden, 1)
        self.tp_log_std = nn.Linear(hidden, 1)
        self.bat_log_std = nn.Linear(hidden, 1)
        self.mode_head = nn.Linear(hidden, 3)
        self.dis_mag_mean = nn.Linear(hidden, 1)
        self.dis_mag_log_std = nn.Linear(hidden, 1)
        self.chg_mag_mean = nn.Linear(hidden, 1)
        self.chg_mag_log_std = nn.Linear(hidden, 1)
        nn.init.zeros_(self.tp_mean.bias)
        nn.init.zeros_(self.bat_mean.bias)
        nn.init.zeros_(self.mode_head.weight)
        nn.init.zeros_(self.mode_head.bias)
        for layer in (
            self.dis_mag_mean,
            self.chg_mag_mean,
            self.dis_mag_log_std,
            self.chg_mag_log_std,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _heads(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(obs)
        return {
            "h": h,
            "mu_tp": self.tp_mean(h).squeeze(-1),
            "ls_tp": self.tp_log_std(h).squeeze(-1).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX),
            "mu_bat": self.bat_mean(h).squeeze(-1),
            "ls_bat": self.bat_log_std(h).squeeze(-1).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX),
            "mode_logits": self.mode_head(h),
            "mu_dis": self.dis_mag_mean(h).squeeze(-1),
            "ls_dis": self.dis_mag_log_std(h).squeeze(-1).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX),
            "mu_chg": self.chg_mag_mean(h).squeeze(-1),
            "ls_chg": self.chg_mag_log_std(h).squeeze(-1).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX),
        }

    def _mask_logits(self, logits: torch.Tensor, mode_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        legal = mode_mask.to(dtype=torch.bool)
        if legal.dim() == 1:
            legal = legal.view(1, -1)
        if legal.size(0) == 1 and logits.size(0) > 1:
            legal = legal.expand(logits.size(0), -1)
        empty = ~legal.any(dim=-1, keepdim=True)
        legal = torch.where(empty, torch.ones_like(legal), legal)
        return logits.masked_fill(~legal, -1.0e9), legal

    def masked_probs(
        self, logits: torch.Tensor, mode_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Softmax over legal modes only; illegal coordinates are exact zeros."""
        logits, legal = self._mask_logits(logits, mode_mask)
        probs = torch.softmax(logits, dim=-1)
        legal_f = legal.to(dtype=probs.dtype)
        probs = probs * legal_f
        denom = probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return probs / denom, legal

    def mode_probs(self, obs: torch.Tensor, mode_mask: torch.Tensor) -> torch.Tensor:
        h = self._heads(obs)
        probs, _ = self.masked_probs(h["mode_logits"], mode_mask)
        return probs

    def _sample_bounded(
        self,
        mu: torch.Tensor,
        log_std: torch.Tensor,
        lo: torch.Tensor,
        hi: torch.Tensor,
        *,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = Normal(mu, log_std.exp())
        z = mu if deterministic else dist.rsample()
        y = torch.sigmoid(z)
        u = lo + y * (hi - lo)
        log_prob = dist.log_prob(z) - sigmoid_log_jacobian(z) - interval_log_jacobian(lo, hi)
        entropy = dist.entropy() + sigmoid_log_jacobian(z) + interval_log_jacobian(lo, hi)
        return u, log_prob, entropy

    def sample_mode_action(
        self,
        obs: torch.Tensor,
        support: dict[str, torch.Tensor],
        mode_idx: int,
        *,
        deterministic: bool = False,
        heads: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample continuous components conditioned on a fixed mode index (0/1/2).

        Pass ``heads`` from a single ``_heads(obs)`` when enumerating all three
        modes in the SAC update so the encoder is not run three extra times.
        """
        h = self._heads(obs) if heads is None else heads
        b = obs.size(0)
        device = obs.device
        mode = torch.full((b,), int(mode_idx), dtype=torch.long, device=device)
        u_tp, lp_tp, ent_tp = self._sample_bounded(
            h["mu_tp"],
            h["ls_tp"],
            support["u_tp_low"],
            support["u_tp_high"],
            deterministic=deterministic,
        )
        u_bat, lp_bat, ent_bat = self._sample_bounded(
            h["mu_bat"],
            h["ls_bat"],
            support["u_bat_low"],
            support["u_bat_high"],
            deterministic=deterministic,
        )
        if mode_idx == MODE_IDLE:
            mag = torch.zeros(b, device=device)
            u_caes = torch.zeros(b, device=device)
            lp_mag = torch.zeros(b, device=device)
            ent_mag = torch.zeros(b, device=device)
            cont_dim = torch.full((b,), 2.0, device=device)
        elif mode_idx == MODE_DISCHARGE:
            dist = Normal(h["mu_dis"], h["ls_dis"].exp())
            z = h["mu_dis"] if deterministic else dist.rsample()
            y = torch.sigmoid(z)
            u_caes = support["dis_lo"] + y * (support["dis_hi"] - support["dis_lo"])
            mag = y
            lp_mag = (
                dist.log_prob(z)
                - sigmoid_log_jacobian(z)
                - interval_log_jacobian(support["dis_lo"], support["dis_hi"])
            )
            ent_mag = (
                dist.entropy()
                + sigmoid_log_jacobian(z)
                + interval_log_jacobian(support["dis_lo"], support["dis_hi"])
            )
            cont_dim = torch.full((b,), 3.0, device=device)
        else:
            dist = Normal(h["mu_chg"], h["ls_chg"].exp())
            z = h["mu_chg"] if deterministic else dist.rsample()
            y = torch.sigmoid(z)
            u_caes = support["chg_lo"] + y * (support["chg_hi"] - support["chg_lo"])
            mag = y
            lp_mag = (
                dist.log_prob(z)
                - sigmoid_log_jacobian(z)
                - interval_log_jacobian(support["chg_lo"], support["chg_hi"])
            )
            ent_mag = (
                dist.entropy()
                + sigmoid_log_jacobian(z)
                + interval_log_jacobian(support["chg_lo"], support["chg_hi"])
            )
            cont_dim = torch.full((b,), 3.0, device=device)

        return {
            "u_tp": u_tp,
            "u_battery": u_bat,
            "u_caes": u_caes,
            "mode_idx": mode,
            "mode_onehot": one_hot_modes(mode).to(device=device, dtype=obs.dtype),
            "mag": mag,
            "log_prob_cont": lp_tp + lp_bat + lp_mag,
            "entropy_cont": ent_tp + ent_bat + ent_mag,
            "cont_dim": cont_dim,
        }

    def act(
        self,
        obs: torch.Tensor,
        support: dict[str, torch.Tensor],
        *,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Sample one hybrid action on A(s): mode first, then magnitude for that mode.

        Exact three-mode enumeration is only for the SAC update
        (`sample_mode_action` + `_exact_soft_value`), not for environment steps.
        """
        h = self._heads(obs)
        b = obs.size(0)
        device = obs.device
        probs, legal = self.masked_probs(h["mode_logits"], support["mode_mask"])
        if deterministic:
            mode = probs.argmax(dim=-1)
        else:
            mode = torch.distributions.Categorical(probs=probs).sample()
        log_mode = torch.log(probs.gather(-1, mode.unsqueeze(-1)).clamp_min(1e-8)).squeeze(-1)
        ent_mode = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)

        u_tp, lp_tp, ent_tp = self._sample_bounded(
            h["mu_tp"],
            h["ls_tp"],
            support["u_tp_low"],
            support["u_tp_high"],
            deterministic=deterministic,
        )
        u_bat, lp_bat, ent_bat = self._sample_bounded(
            h["mu_bat"],
            h["ls_bat"],
            support["u_bat_low"],
            support["u_bat_high"],
            deterministic=deterministic,
        )

        dist_dis = Normal(h["mu_dis"], h["ls_dis"].exp())
        dist_chg = Normal(h["mu_chg"], h["ls_chg"].exp())
        z_dis = h["mu_dis"] if deterministic else dist_dis.rsample()
        z_chg = h["mu_chg"] if deterministic else dist_chg.rsample()
        y_dis = torch.sigmoid(z_dis)
        y_chg = torch.sigmoid(z_chg)
        u_dis = support["dis_lo"] + y_dis * (support["dis_hi"] - support["dis_lo"])
        u_chg = support["chg_lo"] + y_chg * (support["chg_hi"] - support["chg_lo"])
        lp_dis = (
            dist_dis.log_prob(z_dis)
            - sigmoid_log_jacobian(z_dis)
            - interval_log_jacobian(support["dis_lo"], support["dis_hi"])
        )
        lp_chg = (
            dist_chg.log_prob(z_chg)
            - sigmoid_log_jacobian(z_chg)
            - interval_log_jacobian(support["chg_lo"], support["chg_hi"])
        )
        ent_dis = (
            dist_dis.entropy()
            + sigmoid_log_jacobian(z_dis)
            + interval_log_jacobian(support["dis_lo"], support["dis_hi"])
        )
        ent_chg = (
            dist_chg.entropy()
            + sigmoid_log_jacobian(z_chg)
            + interval_log_jacobian(support["chg_lo"], support["chg_hi"])
        )

        zeros = torch.zeros(b, device=device)
        is_dis = mode == MODE_DISCHARGE
        is_chg = mode == MODE_CHARGE
        mag = torch.where(is_dis, y_dis, torch.where(is_chg, y_chg, zeros))
        u_caes = torch.where(is_dis, u_dis, torch.where(is_chg, u_chg, zeros))
        lp_mag = torch.where(is_dis, lp_dis, torch.where(is_chg, lp_chg, zeros))
        ent_mag = torch.where(is_dis, ent_dis, torch.where(is_chg, ent_chg, zeros))
        cont_dim = torch.where(is_dis | is_chg, torch.full((b,), 3.0, device=device), torch.full((b,), 2.0, device=device))
        lp_cont = lp_tp + lp_bat + lp_mag
        ent_cont = ent_tp + ent_bat + ent_mag

        return {
            "u_tp": u_tp,
            "u_battery": u_bat,
            "u_caes": u_caes,
            "mode_idx": mode,
            "mode_onehot": one_hot_modes(mode).to(device=device, dtype=obs.dtype),
            "mag": mag,
            "mode_probs": probs,
            "mode_mask": legal,
            "log_prob_mode": log_mode,
            "log_prob_cont": lp_cont,
            "log_prob": log_mode + lp_cont,
            "entropy_mode": ent_mode,
            "entropy_cont": ent_cont,
            "entropy": ent_mode + ent_cont,
            "cont_dim": cont_dim,
            "n_modes": legal.sum(dim=-1).to(dtype=torch.float32),
        }

    def act_numpy(self, obs, feasible, deterministic: bool = True, device="cpu"):
        self.eval()
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=device).view(1, -1)
            support = support_from_feasible_batch([feasible], device=device)
            out = self.act(o, support, deterministic=deterministic)
        return physical_dict(
            float(out["u_tp"][0].cpu()),
            float(out["u_battery"][0].cpu()),
            float(out["u_caes"][0].cpu()),
        )
