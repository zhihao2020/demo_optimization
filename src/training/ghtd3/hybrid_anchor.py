"""Hybrid 锚定：冻结执行 / 权重移植到 goal-conditioned 底层。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from training.hybrid_td3.algorithm import HybridTD3


class HybridAnchor:
    """加载 Hybrid TD3；可冻结执行 a_H(s)，或把权重移植到 goal-conditioned LowLevelActor。"""

    def __init__(
        self,
        obs_dim: int,
        checkpoint: str | Path,
        *,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = int(obs_dim)
        self.agent = HybridTD3(obs_dim=obs_dim, device=self.device, explore_noise=0.0)
        self.agent.load(checkpoint)
        self.agent.actor.eval()
        self.agent.actor_target.eval()
        for p in self.agent.actor.parameters():
            p.requires_grad_(False)
        self.checkpoint = str(checkpoint)

    @torch.no_grad()
    def act(self, obs: np.ndarray, feasible: Any, *, deterministic: bool = True) -> dict[str, Any]:
        return self.agent.select_action(obs, feasible, deterministic=deterministic)

    @staticmethod
    def _f(x: Any) -> float:
        return float(np.asarray(x, dtype=np.float64).reshape(-1)[0])

    def act_scalars(self, obs: np.ndarray, feasible: Any, *, deterministic: bool = True) -> dict[str, float]:
        a = self.act(obs, feasible, deterministic=deterministic)
        return {
            "u_tp": self._f(a["u_tp"]),
            "u_battery": self._f(a["u_battery"]),
            "caes_mode": int(np.asarray(a["caes_mode"]).reshape(-1)[0]),
            "caes_magnitude": self._f(a.get("caes_magnitude", 0.0)),
        }

    @torch.no_grad()
    def act_tensors(
        self,
        obs: torch.Tensor,
        u_tp_low: torch.Tensor,
        u_tp_high: torch.Tensor,
        u_bat_low: torch.Tensor,
        u_bat_high: torch.Tensor,
        mode_mask: torch.Tensor,
        *,
        deterministic: bool = True,
    ) -> dict[str, torch.Tensor]:
        """批量 Hybrid 动作（raw obs，与训练 Hybrid 一致）。"""
        self.agent.actor.eval()
        out = self.agent.actor.act(
            obs,
            u_tp_low,
            u_tp_high,
            u_bat_low,
            u_bat_high,
            mode_mask,
            deterministic=deterministic,
            explore_noise_std=0.0,
        )
        return {
            "u_tp": out["u_tp"],
            "u_battery": out["u_battery"],
            "caes_mode": out["caes_mode"],
            "caes_mode_oh": out["caes_mode_oh"],
            "caes_magnitude": out["caes_magnitude"],
        }

    def transplant_into_goal_actor(self, lo_actor: nn.Module, goal_dim: int) -> dict[str, Any]:
        """把 Hybrid actor 权重写入 LowLevelActor，使 π(s, g=0) ≈ π_H(s)。

        LowLevelActor 输入为 concat(s, g)：
        - encoder 第一层：obs 列复制 Hybrid，goal 列置 0
        - 其余层与动作头直接复制
        """
        ha = self.agent.actor
        sd_h = ha.state_dict()
        sd_l = lo_actor.state_dict()
        report = {"copied": [], "skipped": [], "goal_dim": int(goal_dim), "obs_dim": self.obs_dim}

        # encoder.0: Linear(obs+goal, hidden) vs Linear(obs, hidden)
        w0_h = sd_h["encoder.0.weight"]  # [H, obs]
        b0_h = sd_h["encoder.0.bias"]
        w0_l = sd_l["encoder.0.weight"]  # [H, obs+goal]
        if w0_h.shape[0] != w0_l.shape[0] or w0_h.shape[1] != self.obs_dim:
            raise RuntimeError(
                f"Hybrid/Low encoder mismatch: hybrid {tuple(w0_h.shape)} low {tuple(w0_l.shape)} obs_dim={self.obs_dim}"
            )
        with torch.no_grad():
            w0_l.zero_()
            w0_l[:, : self.obs_dim].copy_(w0_h)
            # goal 列保持 0 → g=0 时与 Hybrid 一致
            sd_l["encoder.0.bias"].copy_(b0_h)
            report["copied"].append("encoder.0 (obs cols + zero goal cols)")

            # encoder.2 同形
            for key in ("encoder.2.weight", "encoder.2.bias"):
                if key in sd_h and key in sd_l and sd_h[key].shape == sd_l[key].shape:
                    sd_l[key].copy_(sd_h[key])
                    report["copied"].append(key)
                else:
                    report["skipped"].append(key)

            for key in (
                "thermal_head.weight",
                "thermal_head.bias",
                "battery_head.weight",
                "battery_head.bias",
                "mode_head.weight",
                "mode_head.bias",
                "discharge_mag_head.weight",
                "discharge_mag_head.bias",
                "charge_mag_head.weight",
                "charge_mag_head.bias",
            ):
                if key in sd_h and key in sd_l and sd_h[key].shape == sd_l[key].shape:
                    sd_l[key].copy_(sd_h[key])
                    report["copied"].append(key)
                else:
                    report["skipped"].append(key)

        lo_actor.load_state_dict(sd_l)
        return report
