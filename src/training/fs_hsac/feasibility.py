"""Residual twin-feasibility network for FS-HSAC (PyTorch adapter)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.fs_hsac.action_support import MODE_CHARGE, MODE_DISCHARGE, one_hot_modes


class ResidualFeasibilityNet(nn.Module):
    """Cψ(s,a) = sigmoid(f(obs, u_tp, u_bat, mode_onehot, mag))."""

    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        in_dim = int(obs_dim) + 2 + 3 + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def _col(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) if x.ndim == 1 else x

    def forward(self, obs, u_tp, u_bat, mode_onehot, mag) -> torch.Tensor:
        x = torch.cat([obs, self._col(u_tp), self._col(u_bat), mode_onehot, self._col(mag)], dim=-1)
        return self.net(x).squeeze(-1)

    def prob(self, obs, u_tp, u_bat, mode_onehot, mag) -> torch.Tensor:
        return torch.sigmoid(self.forward(obs, u_tp, u_bat, mode_onehot, mag))


def physical_to_hybrid_features(
    u_tp: torch.Tensor,
    u_bat: torch.Tensor,
    u_caes: torch.Tensor,
    dis_lo: torch.Tensor,
    dis_hi: torch.Tensor,
    chg_lo: torch.Tensor,
    chg_hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mode = torch.ones_like(u_caes, dtype=torch.long)
    mode = torch.where(u_caes < -1e-6, torch.zeros_like(mode), mode)
    mode = torch.where(u_caes > 1e-6, torch.full_like(mode, 2), mode)
    mag = torch.zeros_like(u_caes)
    mag = torch.where(
        mode == MODE_DISCHARGE,
        ((u_caes - dis_lo) / (dis_hi - dis_lo).clamp_min(1e-4)).clamp(0.0, 1.0),
        mag,
    )
    mag = torch.where(
        mode == MODE_CHARGE,
        ((u_caes - chg_lo) / (chg_hi - chg_lo).clamp_min(1e-4)).clamp(0.0, 1.0),
        mag,
    )
    return one_hot_modes(mode).to(dtype=u_caes.dtype, device=u_caes.device), mag


class FeasibilityTrainer:
    """Train ResidualFeasibilityNet from FeasibilityReplay with class balance."""

    def __init__(
        self,
        net: ResidualFeasibilityNet,
        *,
        lr: float = 3e-4,
        device: torch.device | str = "cpu",
        min_unsafe: int = 32,
        min_safe: int = 32,
    ):
        self.net = net.to(device)
        self.device = torch.device(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.min_unsafe = int(min_unsafe)
        self.min_safe = int(min_safe)
        self.enabled = False
        self.last_metrics: dict[str, float] = {}

    def maybe_enable(self, n_safe: int, n_unsafe: int) -> bool:
        self.enabled = n_safe >= self.min_safe and n_unsafe >= self.min_unsafe
        return self.enabled

    def update(self, batch: dict, batch_size: int | None = None) -> dict[str, float]:
        labels = torch.as_tensor(batch["feasibility_label"], device=self.device)
        n_safe = int((labels > 0.5).sum().item())
        n_unsafe = int((labels <= 0.5).sum().item())
        if not self.maybe_enable(n_safe, n_unsafe) and (n_safe < 1 or n_unsafe < 1):
            return {"feas_loss": 0.0, "enabled": 0.0, "n_safe": float(n_safe), "n_unsafe": float(n_unsafe)}

        obs = torch.as_tensor(batch["obs"], device=self.device)
        u_tp = torch.as_tensor(batch["u_tp"], device=self.device)
        u_bat = torch.as_tensor(batch["u_battery"], device=self.device)
        u_caes = torch.as_tensor(batch["u_caes"], device=self.device)
        onehot, mag = physical_to_hybrid_features(
            u_tp,
            u_bat,
            u_caes,
            torch.as_tensor(batch["dis_lo"], device=self.device),
            torch.as_tensor(batch["dis_hi"], device=self.device),
            torch.as_tensor(batch["chg_lo"], device=self.device),
            torch.as_tensor(batch["chg_hi"], device=self.device),
        )
        logits = self.net(obs, u_tp, u_bat, onehot, mag)
        # pos_weight for unsafe (label 0) rarity: weight unsafe more
        # BCE with logits expects label 1 = positive class = safe here
        pos = max(n_safe, 1)
        neg = max(n_unsafe, 1)
        pos_weight = torch.tensor([neg / pos], device=self.device)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.opt.step()

        with torch.no_grad():
            prob = torch.sigmoid(logits)
            pred_safe = prob >= 0.5
            true_safe = labels > 0.5
            true_unsafe = ~true_safe
            false_safe = (pred_safe & true_unsafe).sum().float()
            false_safe_rate = float(false_safe / max(true_unsafe.sum().item(), 1))
            acc = float((pred_safe == true_safe).float().mean().item())
        metrics = {
            "feas_loss": float(loss.item()),
            "enabled": float(self.enabled),
            "n_safe": float(n_safe),
            "n_unsafe": float(n_unsafe),
            "false_safe_rate": false_safe_rate,
            "feas_acc": acc,
        }
        self.last_metrics = metrics
        return metrics
