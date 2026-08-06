"""GHTD3 双层 TD3 agent。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .buffers import HighReplayBuffer, HighTransition, LowReplayBuffer
from .goals import (
    actual_delta_soc,
    clip_goal,
    default_goal_boxes,
    extract_soc_from_obs,
    residual_scale_from_goal,
)
from .networks import HighLevelActor, HighLevelCritic, LowLevelActor, LowLevelCritic


class GHTD3Agent:
    def __init__(self, obs_dim: int, cfg: dict[str, Any], device: str | None = None):
        self.cfg = dict(cfg)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dlow, dhigh = default_goal_boxes()
        self.goal_dim = int(cfg.get("goal_dim", len(dlow)))
        # 若配置仍是 2 维而 goal_dim=5，用默认 5 维盒
        gl_cfg = cfg.get("goal_low")
        gh_cfg = cfg.get("goal_high")
        if gl_cfg is None or len(np.asarray(gl_cfg).ravel()) != self.goal_dim:
            self.goal_low = dlow[: self.goal_dim].astype(np.float32)
            self.goal_high = dhigh[: self.goal_dim].astype(np.float32)
            if self.goal_dim > len(dlow):
                # 扩展默认
                extra = self.goal_dim - len(dlow)
                self.goal_low = np.concatenate([self.goal_low, -0.1 * np.ones(extra, np.float32)])
                self.goal_high = np.concatenate([self.goal_high, 0.1 * np.ones(extra, np.float32)])
        else:
            self.goal_low = np.asarray(gl_cfg, dtype=np.float32).ravel()[: self.goal_dim]
            self.goal_high = np.asarray(gh_cfg, dtype=np.float32).ravel()[: self.goal_dim]
        self.gamma = float(cfg.get("gamma", 0.99))
        # SMDP：高层一步跨越 c 个环境步，折扣 γ^c（Cui et al. GHTD3）
        self.subgoal_interval = max(int(cfg.get("subgoal_interval", 8)), 1)
        self.gamma_high = float(self.gamma ** self.subgoal_interval)
        self.tau = float(cfg.get("tau", 0.005))
        self.policy_delay = int(cfg.get("policy_delay", 2))
        self.target_noise = float(cfg.get("target_noise", 0.2))
        self.noise_clip = float(cfg.get("noise_clip", 0.5))
        rc = cfg.get("reward_clip") or [-20.0, 20.0]
        self.reward_clip = (float(rc[0]), float(rc[1]))
        # 高层 reward 为 c 步均值后尺度接近底层，q_clip 可收紧
        self.q_clip = float(cfg.get("q_clip", 200.0))
        self.q_clip_high = float(cfg.get("q_clip_high", min(self.q_clip, 50.0)))
        self.high_noise = float(cfg.get("high_explore_noise", 0.05))
        self.low_noise = float(cfg.get("low_explore_noise", 0.08))
        # 高层 reward = mean_ext over cycle（稳定 critic；相对论文 sum 的实现改进）
        self.high_reward_normalize = bool(cfg.get("high_reward_normalize", True))
        # 默认 false：绝对 GC 主线无教师；残差路径 yaml 显式 hybrid_anchor: true
        self.hybrid_anchor_enabled = bool(cfg.get("hybrid_anchor", False))
        # action_residual: a=a_H+β tanh Δ(s,g)（推荐）；goal_conditioned: 绝对头；blend: 旧混合
        self.execution_mode = str(cfg.get("execution_mode", "action_residual")).lower()
        self.residual_alpha0 = float(cfg.get("residual_alpha0", 0.0))
        self._hybrid_anchor = None  # set by train via attach_hybrid_anchor
        # action_residual 用 norm obs；仅绝对头移植 Hybrid 时用 raw
        self.low_use_raw_obs = bool(cfg.get("low_use_raw_obs", False))
        self.beta_tp = float(cfg.get("residual_beta_tp", 0.15))
        self.beta_bat = float(cfg.get("residual_beta_bat", 0.30))
        self.beta_mag = float(cfg.get("residual_beta_mag", 0.20))
        self.mode_margin = float(cfg.get("residual_mode_margin", 1.5))
        self.mode_override = bool(cfg.get("residual_mode_override", False))
        self.logit_clip = float(cfg.get("residual_logit_clip", 8.0))
        self.beta_arb_scale = bool(cfg.get("residual_beta_from_arb", True))
        # TEA：教师锚定可扩张（progress∈[0,1] 由 train 注入）
        self._train_progress = 0.0
        self.tea_enabled = self.execution_mode in ("tea", "action_residual") and bool(
            cfg.get("tea_expandable", self.execution_mode == "tea")
        )
        # LTAR：Lagrangian 信任域（无 TEA 课程；λ 收紧 β）
        self.ltar_enabled = bool(cfg.get("ltar_enabled", False)) or str(
            cfg.get("execution_mode", "")
        ).lower() in ("ltar", "ltar_td3")
        # 高层-only λ-SoC（绝对 GC 可用；不启用动作信任域）
        self.high_lambda_soc = bool(cfg.get("high_lambda_soc", False))
        if self.ltar_enabled:
            # 与 TEA 课程互斥：扩张只靠 \(\bar\beta\cdot\sigma\)，不靠 ρ(t)
            self.tea_enabled = False
        self.tea_beta_max_tp = float(cfg.get("tea_beta_max_tp", cfg.get("residual_beta_tp", 0.15)))
        self.tea_beta_max_bat = float(cfg.get("tea_beta_max_bat", cfg.get("residual_beta_bat", 0.30)))
        self.tea_beta_max_mag = float(cfg.get("tea_beta_max_mag", cfg.get("residual_beta_mag", 0.20)))
        self.tea_rho_start = float(cfg.get("tea_rho_start", 0.15))
        self.tea_rho_end = float(cfg.get("tea_rho_end", 1.0))
        self.tea_mode_unlock = float(cfg.get("tea_mode_unlock_progress", 0.45))
        self.tea_adv_gate = bool(cfg.get("tea_adv_gate", True))
        self.tea_adv_temp = float(cfg.get("tea_adv_temp", 0.5))
        self.tea_bc0 = float(cfg.get("tea_teacher_bc_coef0", 0.35))
        self.tea_bc1 = float(cfg.get("tea_teacher_bc_coef1", 0.02))
        # 基底 β（扩张前的下限尺度）
        self.beta_tp0 = float(cfg.get("residual_beta_tp", 0.15))
        self.beta_bat0 = float(cfg.get("residual_beta_bat", 0.30))
        self.beta_mag0 = float(cfg.get("residual_beta_mag", 0.20))
        # Lagrangian dual for terminal inventory cost
        self.lambda_soc = float(cfg.get("ltar_lambda_init", 0.0))
        self.ltar_lambda_lr = float(cfg.get("ltar_lambda_lr", 0.15))
        self.ltar_lambda_max = float(cfg.get("ltar_lambda_max", 8.0))
        self.ltar_cost_tol = float(cfg.get("ltar_cost_tol", 0.0))
        self.ltar_adv_gate = bool(cfg.get("ltar_adv_gate", True))
        self.ltar_adv_temp = float(cfg.get("ltar_adv_temp", cfg.get("tea_adv_temp", 0.55)))

        gl = torch.as_tensor(self.goal_low, device=self.device)
        gh = torch.as_tensor(self.goal_high, device=self.device)
        self._goal_low_t = gl
        self._goal_high_t = gh

        # action_residual 默认 residual_init；goal_conditioned 移植时由 attach 覆盖
        _default_res_init = self.execution_mode == "action_residual" or bool(
            cfg.get("residual_init", self.hybrid_anchor_enabled)
        )
        residual_init = bool(cfg.get("residual_init", _default_res_init))
        # goal_conditioned / residual 均需放大 goal 通道，避免相对高维 obs_norm 被淹没
        _default_gscale = (
            4.0
            if self.execution_mode in ("action_residual", "tea", "goal_conditioned")
            else 1.0
        )
        goal_scale = float(cfg.get("goal_input_scale", _default_gscale))
        self.hi_actor = HighLevelActor(obs_dim, self.goal_dim).to(self.device)
        self.hi_actor_t = deepcopy(self.hi_actor).to(self.device)
        self.hi_critic = HighLevelCritic(obs_dim, self.goal_dim).to(self.device)
        self.hi_critic_t = deepcopy(self.hi_critic).to(self.device)
        self.lo_actor = LowLevelActor(
            obs_dim, self.goal_dim, residual_init=residual_init, goal_input_scale=goal_scale
        ).to(self.device)
        self.lo_actor_t = deepcopy(self.lo_actor).to(self.device)
        self.lo_critic = LowLevelCritic(obs_dim, self.goal_dim).to(self.device)
        self.lo_critic_t = deepcopy(self.lo_critic).to(self.device)

        alr = float(cfg.get("actor_lr", 3e-4))
        clr = float(cfg.get("critic_lr", 3e-4))
        self.hi_actor_opt = torch.optim.Adam(self.hi_actor.parameters(), lr=alr)
        self.hi_critic_opt = torch.optim.Adam(self.hi_critic.parameters(), lr=clr)
        self.lo_actor_opt = torch.optim.Adam(self.lo_actor.parameters(), lr=alr)
        self.lo_critic_opt = torch.optim.Adam(self.lo_critic.parameters(), lr=clr)

        self.hi_buffer = HighReplayBuffer(int(cfg.get("high_buffer_size", 20000)))
        self.lo_buffer = LowReplayBuffer(int(cfg.get("low_buffer_size", 100000)))
        self.hi_it = 0
        self.lo_it = 0
        self.last_metrics: dict[str, float] = {}
        # 功率等观测可达 1e8，不归一化则 actor 反传梯度数值上为 0
        self.obs_norm = bool(cfg.get("obs_norm", True))
        self._obs_scale = float(cfg.get("obs_scale", 1.0e6))

    def _prep_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.obs_norm:
            return obs
        # 压缩大动态范围，保留符号；SOC 等小量几乎不变
        return torch.tanh(obs / max(self._obs_scale, 1.0))

    def _expand_bounds(self, arr: np.ndarray, batch: int) -> torch.Tensor:
        t = torch.as_tensor(arr, device=self.device, dtype=torch.float32)
        if t.ndim == 1:
            t = t.unsqueeze(0).expand(batch, -1)
        return t

    def select_goal(self, obs: np.ndarray, *, deterministic: bool = False, random: bool = False) -> np.ndarray:
        if random:
            g = np.random.uniform(self.goal_low, self.goal_high).astype(np.float32)
            return clip_goal(g, self.goal_low, self.goal_high)
        self.hi_actor.eval()
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
            o = self._prep_obs(o)
            g = self.hi_actor(o, self._goal_low_t.unsqueeze(0), self._goal_high_t.unsqueeze(0))
            g = g.cpu().numpy().ravel()
        if not deterministic and self.high_noise > 0:
            g = g + np.random.randn(self.goal_dim).astype(np.float32) * self.high_noise
        return clip_goal(g, self.goal_low, self.goal_high)

    def attach_hybrid_anchor(self, anchor, *, transplant: bool | None = None) -> dict[str, Any]:
        """挂载 Hybrid。

        - action_residual：不移植，Hybrid 作 a_H，底层只学 Δ（obs_norm）。
        - goal_conditioned + hybrid_init_low：权重移植（旧路径，诊断显示易饱和）。
        """
        self._hybrid_anchor = anchor
        self.hybrid_anchor_enabled = anchor is not None
        report: dict[str, Any] = {"transplant": False, "execution_mode": self.execution_mode}
        if transplant is None:
            transplant = bool(self.cfg.get("hybrid_init_low", False)) and self.execution_mode not in (
                "action_residual",
                "tea",
            )
        if anchor is not None and transplant:
            report = anchor.transplant_into_goal_actor(self.lo_actor, self.goal_dim)
            self.lo_actor_t = deepcopy(self.lo_actor)
            self.low_use_raw_obs = True
            self.obs_norm = bool(self.cfg.get("obs_norm_high_only", True))
            alr = float(self.cfg.get("residual_actor_lr", self.cfg.get("actor_lr", 1e-4)))
            clr = float(self.cfg.get("critic_lr", 3e-4))
            self.lo_actor_opt = torch.optim.Adam(self.lo_actor.parameters(), lr=alr)
            self.lo_critic_opt = torch.optim.Adam(self.lo_critic.parameters(), lr=clr)
            report["transplant"] = True
            report["execution_mode"] = self.execution_mode
        elif anchor is not None and self.execution_mode in ("action_residual", "tea"):
            # 残差/TEA 底层：norm obs + 较小 lr
            self.low_use_raw_obs = bool(self.cfg.get("low_use_raw_obs", False))
            alr = float(self.cfg.get("residual_actor_lr", self.cfg.get("actor_lr", 1e-4)))
            clr = float(self.cfg.get("critic_lr", 3e-4))
            self.lo_actor_opt = torch.optim.Adam(self.lo_actor.parameters(), lr=alr)
            self.lo_critic_opt = torch.optim.Adam(self.lo_critic.parameters(), lr=clr)
            report["action_residual"] = True
            report["tea"] = bool(self.tea_enabled)
            report["beta0"] = {"tp": self.beta_tp0, "bat": self.beta_bat0, "mag": self.beta_mag0}
            report["beta_max"] = {
                "tp": self.tea_beta_max_tp,
                "bat": self.tea_beta_max_bat,
                "mag": self.tea_beta_max_mag,
            }
        return report

    def _prep_obs_low(self, obs: torch.Tensor) -> torch.Tensor:
        if self.low_use_raw_obs:
            return obs
        return self._prep_obs(obs)

    def set_progress(self, progress: float) -> None:
        """训练进度 ∈[0,1]：驱动 TEA 扩张课程与 mode 解锁。"""
        self._train_progress = float(np.clip(progress, 0.0, 1.0))
        # LTAR / STFR / 信任域残差：mode 始终锁教师（模式因子化）
        if (
            self.ltar_enabled
            or bool(self.cfg.get("stfr_enabled", False))
            or str(self.cfg.get("high_level_mode", "")).lower()
            in ("prior_only", "prior", "stfr_a", "stfr")
        ):
            self.mode_override = False
            return
        if self.tea_enabled:
            if bool(self.cfg.get("tea_force_mode_lock", False)):
                # 冬季安全：mode 全程跟教师，只扩连续维（防 CAES 炸吞吐）
                self.mode_override = False
            else:
                self.mode_override = self._train_progress >= self.tea_mode_unlock

    def update_lambda_soc(self, cost: float) -> float:
        """Lagrangian 对偶更新：cost = [‖z_T−z_0‖−ξ]_+ 或 1{SOC 失败}。"""
        if not (self.ltar_enabled or getattr(self, "high_lambda_soc", False)):
            return self.lambda_soc
        c = float(max(0.0, cost - self.ltar_cost_tol))
        self.lambda_soc = float(
            np.clip(
                self.lambda_soc + self.ltar_lambda_lr * c,
                0.0,
                self.ltar_lambda_max,
            )
        )
        # 若本回合满足约束（cost≈0），缓慢松弛 λ，允许再探索
        if c <= 1e-8:
            self.lambda_soc = float(max(0.0, self.lambda_soc * 0.97))
        self.last_metrics["lambda_soc"] = self.lambda_soc
        self.last_metrics["ltar_cost"] = c
        return self.lambda_soc

    def lambda_beta_scale(self) -> float:
        """σ_λ = 1/(1+λ)：违约越大，信任域越紧。"""
        if not self.ltar_enabled:
            return 1.0
        return float(1.0 / (1.0 + max(0.0, self.lambda_soc)))

    def set_episode_context(self, *, rem_steps: int | None = None, episode_steps: int | None = None) -> None:
        """注入回合上下文：剩余步数用于回收期收缩残差（安全可扩张）。"""
        if rem_steps is not None:
            self._rem_steps = int(rem_steps)
        if episode_steps is not None:
            self._episode_steps = int(episode_steps)

    def _tea_rho(self) -> float:
        """课程扩张系数 ρ：progress 从 tea_rho_start 线性爬到 tea_rho_end。"""
        p = self._train_progress
        t0 = float(np.clip(self.tea_rho_start, 0.0, 0.95))
        if p <= t0:
            return 0.05  # 极弱扩张，接近教师
        u = (p - t0) / max(1e-6, 1.0 - t0)
        return float(np.clip(0.05 + (self.tea_rho_end - 0.05) * u, 0.05, 1.0))

    def teacher_bc_weight(self) -> float:
        """actor 额外 ‖a−a_H‖² 权重：随进度退火；回收窗抬高（回教师保 SOC）。"""
        if self.ltar_enabled:
            w = float(self.cfg.get("actor_bc_weight", 0.20))
            # λ 大 → 更贴教师
            w = w + 0.08 * float(min(self.lambda_soc, self.ltar_lambda_max))
            rem = getattr(self, "_rem_steps", None)
            H = int(self.cfg.get("recovery_goal_horizon_steps", 40) or 40)
            if rem is not None and rem <= H:
                floor = float(self.cfg.get("tea_recovery_bc_floor", 0.40))
                w = max(w, floor)
            return float(min(w, 1.5))
        if not self.tea_enabled:
            return float(self.cfg.get("actor_bc_weight", 0.15))
        p = self._train_progress
        w = float(self.tea_bc0 + (self.tea_bc1 - self.tea_bc0) * p)
        rem = getattr(self, "_rem_steps", None)
        H = int(self.cfg.get("recovery_goal_horizon_steps", 40) or 40)
        if rem is not None and rem <= H:
            # 期末强制更贴教师，防止扩张导致终端 SOC 漂移
            floor = float(self.cfg.get("tea_recovery_bc_floor", 0.28))
            w = max(w, floor)
        return w

    def _safe_expansion_scales(self, obs: np.ndarray) -> tuple[float, float]:
        """参考调速器：仅在库存安全带内允许大残差；回收期收缩。

        Returns:
            (scale_tp, scale_bat) ∈ (0,1]
        """
        if not bool(self.cfg.get("tea_safe_expansion", True)):
            return 1.0, 1.0
        st = 1.0
        sb = 1.0
        try:
            soc = extract_soc_from_obs(np.asarray(obs, dtype=np.float32), 2)
            bat = float(soc[0]) if soc.size > 0 else 0.5
            gas = float(soc[1]) if soc.size > 1 else 0.8
        except Exception:
            bat, gas = 0.5, 0.8
        # 库存边界屏障：越靠边越禁止大动作修正
        lo_b = float(self.cfg.get("tea_soc_lo", 0.18))
        hi_b = float(self.cfg.get("tea_soc_hi", 0.92))
        for s in (bat, gas):
            if s <= lo_b or s >= hi_b:
                sb *= 0.20
                st *= 0.55
            elif s <= lo_b + 0.10 or s >= hi_b - 0.08:
                sb *= 0.45
                st *= 0.75
        # 回收期：剩余步数越少，残差越小（原理：终端约束 → 收缩可行修正集）
        rem = getattr(self, "_rem_steps", None)
        H = int(self.cfg.get("recovery_goal_horizon_steps", 40) or 40)
        if rem is not None and rem <= H:
            # rem=H → 1.0；rem=0 → tea_recovery_beta_floor
            floor = float(self.cfg.get("tea_recovery_beta_floor", 0.12))
            frac = float(np.clip(rem / max(H, 1), 0.0, 1.0))
            rec = floor + (1.0 - floor) * frac
            sb *= rec
            st *= max(rec, 0.35)
        return float(np.clip(st, 0.05, 1.0)), float(np.clip(sb, 0.05, 1.0))

    def _betas_from_goal(
        self,
        goal: torch.Tensor,
        *,
        buy_price: torch.Tensor | float | None = None,
        adv_scale: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """β 调度：TEA 扩张 + g_arb + 可选价差/优势门控。"""
        b = goal.shape[0]
        device = goal.device
        dtype = goal.dtype
        if self.tea_enabled:
            rho = self._tea_rho()
            # 在 [β0, β_max] 间按 ρ 插值
            bt0, bb0, bm0 = self.beta_tp0, self.beta_bat0, self.beta_mag0
            bt = bt0 + (self.tea_beta_max_tp - bt0) * rho
            bb = bb0 + (self.tea_beta_max_bat - bb0) * rho
            bm = bm0 + (self.tea_beta_max_mag - bm0) * rho
            bt = torch.full((b,), bt, device=device, dtype=dtype)
            bb = torch.full((b,), bb, device=device, dtype=dtype)
            bm = torch.full((b,), bm, device=device, dtype=dtype)
        else:
            bt = torch.full((b,), self.beta_tp, device=device, dtype=dtype)
            bb = torch.full((b,), self.beta_bat, device=device, dtype=dtype)
            bm = torch.full((b,), self.beta_mag, device=device, dtype=dtype)
        if self.beta_arb_scale and goal.shape[-1] > 4:
            arb = goal[:, 4].clamp(0.0, 1.0)
            scale = 0.55 + 0.45 * arb
            bt, bb, bm = bt * scale, bb * scale, bm * scale
        if bool(self.cfg.get("residual_beta_price_gate", False)) and buy_price is not None:
            thr = float(self.cfg.get("residual_price_gate_thr", 0.70))
            boost = float(self.cfg.get("residual_price_gate_boost", 1.35))
            if torch.is_tensor(buy_price):
                p = buy_price.reshape(-1).to(device=device, dtype=dtype)
                if p.numel() == 1:
                    p = p.expand(b)
            else:
                p = torch.full((b,), float(buy_price), device=device, dtype=dtype)
            gate = torch.where(p >= thr, torch.full_like(p, boost), torch.ones_like(p))
            bb = bb * gate
            bt = bt * (0.85 + 0.15 * gate)
        # 优势门控：adv_scale∈(0,1+] 放大 β（仅当残差 Q 更好）
        if adv_scale is not None:
            if not torch.is_tensor(adv_scale):
                adv_scale = torch.full((b,), float(adv_scale), device=device, dtype=dtype)
            else:
                adv_scale = adv_scale.reshape(-1).to(device=device, dtype=dtype)
                if adv_scale.numel() == 1:
                    adv_scale = adv_scale.expand(b)
            bt, bb, bm = bt * adv_scale, bb * adv_scale, bm * adv_scale
        # LTAR：σ_λ 收紧信任域
        if self.ltar_enabled:
            ls = self.lambda_beta_scale()
            bt, bb, bm = bt * ls, bb * ls, bm * ls
        return bt, bb, bm

    def _select_action_residual(
        self,
        obs: np.ndarray,
        goal: np.ndarray,
        feasible,
        *,
        deterministic: bool = False,
    ) -> dict:
        """a = clip(a_H + β tanh Δ(s_norm, g))。"""
        if self._hybrid_anchor is None:
            return self.select_low_action(obs, goal, feasible, deterministic=deterministic)
        a_h = self._hybrid_anchor.act_scalars(obs, feasible, deterministic=True)
        self.lo_actor.eval()
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
            o = self._prep_obs_low(o)
            g = torch.as_tensor(goal, dtype=torch.float32, device=self.device).view(1, -1)
            if g.shape[-1] != self.goal_dim:
                gg = torch.zeros(1, self.goal_dim, device=self.device)
                n = min(self.goal_dim, g.shape[-1])
                gg[:, :n] = g[:, :n]
                g = gg
            buy_p = None
            if bool(self.cfg.get("residual_beta_price_gate", False)):
                buy_p = getattr(self, "_last_buy_price", None)
            bt, bb, bm = self._betas_from_goal(g, buy_price=buy_p)
            # 安全可扩张：SOC 屏障 + 回收期收缩（先于优势门控）
            s_tp, s_bat = self._safe_expansion_scales(obs)
            bt = bt * s_tp
            bb = bb * s_bat
            bm = bm * min(s_bat, s_tp)
            mask = torch.as_tensor(feasible.mode_mask.as_bool_array(), dtype=torch.bool, device=self.device).view(1, 3)
            # 优势门控（TEA 或 LTAR）：残差 Q 更好才放大 β
            adv_scale = None
            use_adv = (self.tea_enabled and self.tea_adv_gate and self._train_progress > self.tea_rho_start) or (
                self.ltar_enabled and self.ltar_adv_gate
            )
            adv_temp = self.ltar_adv_temp if self.ltar_enabled else self.tea_adv_temp
            if use_adv:
                out_probe = self.lo_actor.residual_compose(
                    o,
                    g,
                    torch.tensor([a_h["u_tp"]], device=self.device),
                    torch.tensor([a_h["u_battery"]], device=self.device),
                    torch.tensor([a_h["caes_mode"]], device=self.device, dtype=torch.long),
                    torch.tensor([a_h["caes_magnitude"]], device=self.device),
                    torch.tensor([feasible.u_tp_low], device=self.device),
                    torch.tensor([feasible.u_tp_high], device=self.device),
                    torch.tensor([feasible.u_battery_low], device=self.device),
                    torch.tensor([feasible.u_battery_high], device=self.device),
                    mask,
                    beta_tp=bt,
                    beta_bat=bb,
                    beta_mag=bm,
                    mode_margin=self.mode_margin,
                    mode_override=self.mode_override,
                    logit_clip=self.logit_clip,
                    soft_mode_for_grad=True,
                    deterministic=True,
                    explore_noise_std=0.0,
                )
                mode_h_oh = F.one_hot(
                    torch.tensor([int(a_h["caes_mode"])], device=self.device, dtype=torch.long), 3
                ).float()
                q_h = self.lo_critic.q1_only(
                    o,
                    g,
                    torch.tensor([a_h["u_tp"]], device=self.device),
                    torch.tensor([a_h["u_battery"]], device=self.device),
                    mode_h_oh,
                    torch.tensor([a_h["caes_magnitude"]], device=self.device),
                )
                q_r = self.lo_critic.q1_only(
                    o,
                    g,
                    out_probe["u_tp"],
                    out_probe["u_battery"],
                    out_probe["caes_mode_oh"],
                    out_probe["caes_magnitude"],
                )
                # σ(ΔQ/T)：残差更好 → scale>1；更差 → scale≈0.7
                delta = (q_r - q_h).clamp(-5.0, 5.0)
                adv_scale = 0.7 + 0.6 * torch.sigmoid(delta / max(adv_temp, 1e-3))
                bt, bb, bm = self._betas_from_goal(g, buy_price=buy_p, adv_scale=adv_scale)
                # 安全尺度 + λ 已在 _betas_from_goal / 下方再次乘 soc scale
                s_tp, s_bat = self._safe_expansion_scales(obs)
                bt = bt * s_tp
                bb = bb * s_bat
                bm = bm * min(s_bat, s_tp)
            out = self.lo_actor.residual_compose(
                o,
                g,
                torch.tensor([a_h["u_tp"]], device=self.device),
                torch.tensor([a_h["u_battery"]], device=self.device),
                torch.tensor([a_h["caes_mode"]], device=self.device, dtype=torch.long),
                torch.tensor([a_h["caes_magnitude"]], device=self.device),
                torch.tensor([feasible.u_tp_low], device=self.device),
                torch.tensor([feasible.u_tp_high], device=self.device),
                torch.tensor([feasible.u_battery_low], device=self.device),
                torch.tensor([feasible.u_battery_high], device=self.device),
                mask,
                beta_tp=bt,
                beta_bat=bb,
                beta_mag=bm,
                mode_margin=self.mode_margin,
                mode_override=self.mode_override,
                logit_clip=self.logit_clip,
                soft_mode_for_grad=False,
                deterministic=deterministic,
                explore_noise_std=0.0 if deterministic else self.low_noise,
            )
        return {
            "u_tp": np.asarray([float(out["u_tp"][0].cpu())], dtype=np.float32),
            "u_battery": np.asarray([float(out["u_battery"][0].cpu())], dtype=np.float32),
            "caes_mode": int(out["caes_mode"][0].cpu()),
            "caes_magnitude": np.asarray([float(out["caes_magnitude"][0].cpu())], dtype=np.float32),
        }

    def select_low_action(
        self,
        obs: np.ndarray,
        goal: np.ndarray,
        feasible,
        *,
        deterministic: bool = False,
    ) -> dict:
        """Goal-conditioned 底层（移植 Hybrid 后：g=0 ≈ Hybrid）。"""
        self.lo_actor.eval()
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
            o = self._prep_obs_low(o)
            g = torch.as_tensor(goal, dtype=torch.float32, device=self.device).view(1, -1)
            if g.shape[-1] != self.goal_dim:
                gg = torch.zeros(1, self.goal_dim, device=self.device)
                n = min(self.goal_dim, g.shape[-1])
                gg[:, :n] = g[:, :n]
                g = gg
            mask = torch.as_tensor(feasible.mode_mask.as_bool_array(), dtype=torch.bool, device=self.device).view(1, 3)
            out = self.lo_actor.act(
                o,
                g,
                torch.tensor([feasible.u_tp_low], device=self.device),
                torch.tensor([feasible.u_tp_high], device=self.device),
                torch.tensor([feasible.u_battery_low], device=self.device),
                torch.tensor([feasible.u_battery_high], device=self.device),
                mask,
                deterministic=deterministic,
                explore_noise_std=0.0 if deterministic else self.low_noise,
            )
        return {
            "u_tp": np.asarray([float(out["u_tp"][0].cpu())], dtype=np.float32),
            "u_battery": np.asarray([float(out["u_battery"][0].cpu())], dtype=np.float32),
            "caes_mode": int(out["caes_mode"][0].cpu()),
            "caes_magnitude": np.asarray([float(out["caes_magnitude"][0].cpu())], dtype=np.float32),
        }

    def select_composed_action(
        self,
        obs: np.ndarray,
        goal: np.ndarray,
        feasible,
        *,
        deterministic: bool = False,
        residual_scale: float | None = None,
    ) -> dict:
        """执行入口。

        - tea / action_residual：a=clip(a_H+β(t) tanh Δ)；TEA 下 β 可扩张。
        - goal_conditioned：a=π_lo(s,g) 绝对头。
        - blend：旧版 (1-α)a_H+α a_res，仅作消融。
        """
        if self.execution_mode in ("action_residual", "tea"):
            return self._select_action_residual(obs, goal, feasible, deterministic=deterministic)
        if self.execution_mode != "blend" or self._hybrid_anchor is None:
            return self.select_low_action(obs, goal, feasible, deterministic=deterministic)

        # ---- 旧 blend 路径（消融）----
        a_h = self._hybrid_anchor.act_scalars(obs, feasible, deterministic=True)
        if residual_scale is None:
            residual_scale = residual_scale_from_goal(
                goal,
                alpha0=self.residual_alpha0,
                alpha_max=float(self.cfg.get("residual_alpha_max", 0.28)),
            )
        alpha = float(np.clip(residual_scale, 0.0, 1.0))
        if alpha <= 1e-6:
            return {
                "u_tp": np.asarray([a_h["u_tp"]], dtype=np.float32),
                "u_battery": np.asarray([a_h["u_battery"]], dtype=np.float32),
                "caes_mode": int(a_h["caes_mode"]),
                "caes_magnitude": np.asarray([a_h["caes_magnitude"]], dtype=np.float32),
            }
        a_r = self.select_low_action(obs, goal, feasible, deterministic=deterministic)
        u_tp = (1.0 - alpha) * a_h["u_tp"] + alpha * float(a_r["u_tp"][0])
        u_bat = (1.0 - alpha) * a_h["u_battery"] + alpha * float(a_r["u_battery"][0])
        if alpha >= 0.5:
            mode = int(a_r["caes_mode"])
            mag = float(a_r["caes_magnitude"][0])
        else:
            mode = int(a_h["caes_mode"])
            mag = float(a_h["caes_magnitude"])
        u_tp = float(np.clip(u_tp, feasible.u_tp_low, feasible.u_tp_high))
        u_bat = float(np.clip(u_bat, feasible.u_battery_low, feasible.u_battery_high))
        mag = float(np.clip(mag, 0.0, 1.0))
        return {
            "u_tp": np.asarray([u_tp], dtype=np.float32),
            "u_battery": np.asarray([u_bat], dtype=np.float32),
            "caes_mode": mode,
            "caes_magnitude": np.asarray([mag], dtype=np.float32),
        }

    def update_low(self, batch_size: int) -> dict[str, float]:
        if len(self.lo_buffer) < batch_size:
            return {}
        self.lo_it += 1
        b = self.lo_buffer.sample_batch(batch_size)
        obs = self._prep_obs_low(torch.as_tensor(b["obs"], device=self.device))
        goal = torch.as_tensor(b["goal"], device=self.device)
        next_obs = self._prep_obs_low(torch.as_tensor(b["next_obs"], device=self.device))
        next_goal = torch.as_tensor(b["next_goal"], device=self.device)
        # goal 维对齐
        if goal.shape[-1] != self.goal_dim:
            gg = torch.zeros(goal.shape[0], self.goal_dim, device=self.device)
            n = min(self.goal_dim, goal.shape[-1])
            gg[:, :n] = goal[:, :n]
            goal = gg
        if next_goal.shape[-1] != self.goal_dim:
            ng = torch.zeros(next_goal.shape[0], self.goal_dim, device=self.device)
            n = min(self.goal_dim, next_goal.shape[-1])
            ng[:, :n] = next_goal[:, :n]
            next_goal = ng
        u_tp = torch.as_tensor(b["u_tp"], device=self.device)
        u_bat = torch.as_tensor(b["u_battery"], device=self.device)
        mode = torch.as_tensor(b["caes_mode"], device=self.device, dtype=torch.int64)
        mode_oh = F.one_hot(mode, 3).float()
        mag = torch.as_tensor(b["caes_magnitude"], device=self.device)
        reward = torch.as_tensor(b["reward"], device=self.device).clamp(*self.reward_clip)
        done = torch.as_tensor(b["done"], device=self.device)
        next_mask = torch.as_tensor(b["next_mode_mask"], device=self.device)

        # 训练边界（保证 actor 反传有宽度）
        tp_lo_b = torch.as_tensor(b["u_tp_low"], device=self.device)
        tp_hi_b = torch.as_tensor(b["u_tp_high"], device=self.device)
        bat_lo_b = torch.as_tensor(b["u_bat_low"], device=self.device)
        bat_hi_b = torch.as_tensor(b["u_bat_high"], device=self.device)
        n_tp_lo = torch.as_tensor(b["next_u_tp_low"], device=self.device)
        n_tp_hi = torch.as_tensor(b["next_u_tp_high"], device=self.device)
        n_bat_lo = torch.as_tensor(b["next_u_bat_low"], device=self.device)
        n_bat_hi = torch.as_tensor(b["next_u_bat_high"], device=self.device)
        min_span_tp, min_span_bat = 0.15, 0.30
        tp_hi_b = torch.maximum(tp_hi_b, tp_lo_b + min_span_tp)
        tp_lo_b = torch.minimum(tp_lo_b, tp_hi_b - min_span_tp).clamp(0.0, 1.0)
        tp_hi_b = tp_hi_b.clamp(0.0, 1.0)
        bat_hi_b = torch.maximum(bat_hi_b, bat_lo_b + min_span_bat)
        bat_lo_b = torch.minimum(bat_lo_b, bat_hi_b - min_span_bat).clamp(-1.0, 1.0)
        bat_hi_b = bat_hi_b.clamp(-1.0, 1.0)
        n_tp_hi = torch.maximum(n_tp_hi, n_tp_lo + min_span_tp)
        n_tp_lo = torch.minimum(n_tp_lo, n_tp_hi - min_span_tp).clamp(0.0, 1.0)
        n_tp_hi = n_tp_hi.clamp(0.0, 1.0)
        n_bat_hi = torch.maximum(n_bat_hi, n_bat_lo + min_span_bat)
        n_bat_lo = torch.minimum(n_bat_lo, n_bat_hi - min_span_bat).clamp(-1.0, 1.0)
        n_bat_hi = n_bat_hi.clamp(-1.0, 1.0)
        mask = torch.as_tensor(b["mode_mask"], device=self.device)
        all_false = ~mask.any(dim=-1, keepdim=True)
        mask = torch.where(all_false, torch.ones_like(mask), mask)
        next_mask = torch.where(~next_mask.any(dim=-1, keepdim=True), torch.ones_like(next_mask), next_mask)

        use_ares = self.execution_mode in ("action_residual", "tea") and self._hybrid_anchor is not None
        # residual 路径：critic 输入用 norm obs；Hybrid base 用 raw obs
        obs_raw = torch.as_tensor(b["obs"], device=self.device)
        next_obs_raw = torch.as_tensor(b["next_obs"], device=self.device)

        with torch.no_grad():
            if use_ares:
                base_n = self._hybrid_anchor.act_tensors(
                    next_obs_raw, n_tp_lo, n_tp_hi, n_bat_lo, n_bat_hi, next_mask, deterministic=True
                )
                bt_n, bb_n, bm_n = self._betas_from_goal(next_goal)
                na = self.lo_actor_t.residual_compose(
                    next_obs,
                    next_goal,
                    base_n["u_tp"],
                    base_n["u_battery"],
                    base_n["caes_mode"],
                    base_n["caes_magnitude"],
                    n_tp_lo,
                    n_tp_hi,
                    n_bat_lo,
                    n_bat_hi,
                    next_mask,
                    beta_tp=bt_n,
                    beta_bat=bb_n,
                    beta_mag=bm_n,
                    mode_margin=self.mode_margin,
                    mode_override=self.mode_override,
                    logit_clip=self.logit_clip,
                    soft_mode_for_grad=True,
                    deterministic=True,
                    explore_noise_std=0.0,
                )
            else:
                na = self.lo_actor_t.act(
                    next_obs,
                    next_goal,
                    n_tp_lo,
                    n_tp_hi,
                    n_bat_lo,
                    n_bat_hi,
                    next_mask,
                    deterministic=False,
                    explore_noise_std=0.0,
                )
            n_tp = torch.clamp(
                na["u_tp"] + (torch.randn_like(na["u_tp"]) * self.target_noise).clamp(-self.noise_clip, self.noise_clip),
                n_tp_lo,
                n_tp_hi,
            )
            n_bat = torch.clamp(
                na["u_battery"]
                + (torch.randn_like(na["u_battery"]) * self.target_noise).clamp(-self.noise_clip, self.noise_clip),
                n_bat_lo,
                n_bat_hi,
            )
            n_mag = torch.clamp(
                na["caes_magnitude"]
                + (torch.randn_like(na["caes_magnitude"]) * self.target_noise).clamp(-self.noise_clip, self.noise_clip),
                0.0,
                1.0,
            )
            q1t, q2t = self.lo_critic_t(next_obs, next_goal, n_tp, n_bat, na["caes_mode_oh"], n_mag)
            target = reward + (1.0 - done) * self.gamma * torch.min(q1t, q2t).clamp(-self.q_clip, self.q_clip)

        self.lo_actor.train()
        self.lo_critic.train()
        q1, q2 = self.lo_critic(obs, goal, u_tp, u_bat, mode_oh, mag)
        # 不 clamp 预测 Q：硬夹会切断梯度，反而让 critic 卡在爆炸区
        loss = F.smooth_l1_loss(q1, target) + F.smooth_l1_loss(q2, target)
        self.lo_critic_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.lo_critic.parameters(), 10.0)
        self.lo_critic_opt.step()

        actor_loss_v = 0.0
        if self.lo_it % self.policy_delay == 0:
            if use_ares:
                with torch.no_grad():
                    base = self._hybrid_anchor.act_tensors(
                        obs_raw, tp_lo_b, tp_hi_b, bat_lo_b, bat_hi_b, mask, deterministic=True
                    )
                bt, bb, bm = self._betas_from_goal(goal)
                cur = self.lo_actor.residual_compose(
                    obs,
                    goal,
                    base["u_tp"],
                    base["u_battery"],
                    base["caes_mode"],
                    base["caes_magnitude"],
                    tp_lo_b,
                    tp_hi_b,
                    bat_lo_b,
                    bat_hi_b,
                    mask,
                    beta_tp=bt,
                    beta_bat=bb,
                    beta_mag=bm,
                    mode_margin=self.mode_margin,
                    mode_override=self.mode_override,
                    logit_clip=self.logit_clip,
                    soft_mode_for_grad=True,
                    deterministic=True,
                    explore_noise_std=0.0,
                )
            else:
                cur = self.lo_actor.act(
                    obs,
                    goal,
                    tp_lo_b,
                    tp_hi_b,
                    bat_lo_b,
                    bat_hi_b,
                    mask,
                    deterministic=False,
                    explore_noise_std=0.0,
                    soft_mode_for_grad=True,
                )
            q_pi = self.lo_critic.q1_only(
                obs, goal, cur["u_tp"], cur["u_battery"], cur["caes_mode_oh"], cur["caes_magnitude"]
            )
            # buffer BC + TEA 教师退火 BC
            bc_w = float(self.cfg.get("actor_bc_weight", 0.15))
            bc = (
                F.mse_loss(cur["u_tp"], u_tp.detach())
                + F.mse_loss(cur["u_battery"], u_bat.detach())
                + F.mse_loss(cur["caes_magnitude"], mag.detach())
                + F.mse_loss(cur["caes_mode_oh"], mode_oh.detach())
            )
            teach_w = self.teacher_bc_weight() if use_ares else 0.0
            if teach_w > 1e-8 and use_ares:
                teach = (
                    F.mse_loss(cur["u_tp"], base["u_tp"].detach())
                    + F.mse_loss(cur["u_battery"], base["u_battery"].detach())
                    + 0.5 * F.mse_loss(cur["caes_magnitude"], base["caes_magnitude"].detach())
                )
            else:
                teach = cur["u_tp"].new_zeros(())
            q_term = q_pi.mean() / (q_pi.detach().abs().mean().clamp_min(1.0))
            aloss = -q_term + bc_w * bc + teach_w * teach
            self.lo_actor_opt.zero_grad()
            aloss.backward()
            torch.nn.utils.clip_grad_norm_(self.lo_actor.parameters(), 10.0)
            self.lo_actor_opt.step()
            self._soft(self.lo_actor, self.lo_actor_t)
            self._soft(self.lo_critic, self.lo_critic_t)
            actor_loss_v = float(aloss.detach().item())

        metrics = {
            "lo_critic_loss": float(loss.item()),
            "lo_actor_loss": actor_loss_v,
            "lo_q1_mean": float(q1.mean().item()),
        }
        self.last_metrics.update(metrics)
        return metrics

    def _achieved_delta(self, tr: HighTransition) -> np.ndarray | None:
        if tr.achieved_delta is not None:
            d = np.asarray(tr.achieved_delta, dtype=np.float32).ravel()
            if d.size == self.goal_dim:
                return d
            out = np.zeros(self.goal_dim, dtype=np.float32)
            out[: min(self.goal_dim, d.size)] = d[: self.goal_dim]
            return out
        if tr.soc_seq and len(tr.soc_seq) >= 2:
            d = actual_delta_soc(tr.soc_seq[0], tr.soc_seq[-1]).ravel()
            out = np.zeros(self.goal_dim, dtype=np.float32)
            out[: min(self.goal_dim, d.size)] = d[: self.goal_dim]
            return out
        return None

    def _sample_future_achieved(self, tr: HighTransition) -> np.ndarray | None:
        """从同 episode 后续 high 转移（若可得）或 buffer 中其它 achieved Δ 采样。"""
        storage = getattr(self.hi_buffer, "_storage", []) or []
        same_ep = [
            t
            for t in storage
            if isinstance(t, HighTransition)
            and int(getattr(t, "episode_id", -1)) == int(tr.episode_id)
            and int(getattr(t, "cycle_idx", -1)) > int(tr.cycle_idx)
        ]
        pool = same_ep if same_ep else [
            t for t in storage if isinstance(t, HighTransition) and self._achieved_delta(t) is not None
        ]
        if not pool:
            return self._achieved_delta(tr)
        other = pool[int(np.random.randint(0, len(pool)))]
        return self._achieved_delta(other)

    def _ms_her_weights(self, tr: HighTransition) -> tuple[float, float]:
        """MS-HER：价区/经济权重（相对均匀 HER-mix 的升级）。

        Returns:
            (w_tou, w_econ) 均 ≥ min_w，用于抬高 ach / future 采样。
        """
        min_w = float(self.cfg.get("ms_her_min_weight", 0.35))
        max_w = float(self.cfg.get("ms_her_max_weight", 2.5))
        # 经济：外在回报越高，future/ach 更可信
        r = float(getattr(tr, "reward_ext_sum", 0.0) or 0.0)
        w_econ = float(np.clip(0.6 + 0.15 * r, min_w, max_w))
        # 价区：若窗内吞吐（代理套利活跃）高，抬高 ach
        thr = 0.0
        for ha in getattr(tr, "action_seq", None) or []:
            thr += abs(float(ha.get("u_battery", 0.0) or 0.0)) + abs(
                float(ha.get("caes_magnitude", 0.0) or 0.0)
            )
        n_act = max(len(getattr(tr, "action_seq", None) or []), 1)
        thr_mean = thr / float(n_act)
        w_tou = float(np.clip(0.5 + 1.2 * thr_mean, min_w, max_w))
        return w_tou, w_econ

    def _relabel_goals(self, transitions: list[HighTransition]) -> np.ndarray:
        """历史目标重放：HER-mix / MS-HER，避免「恒等于实际 Δ」的平凡 relabel。

        - her_mix: 原 goal / 本窗达成 / future 均匀三类概率
        - ms_her: 同上，但对 ach/future 按价区吞吐与外在回报加权；可选注入市场 prior 候选
        """
        mode = str(self.cfg.get("goal_relabel_mode", "her_mix")).lower()
        use_ms = mode in ("ms_her", "market_her", "ms-her") or bool(
            self.cfg.get("ms_her_weighting", False)
        )
        if mode in ("ms_her", "market_her", "ms-her"):
            mode = "her_mix"  # 分支逻辑同 her_mix，仅权重不同
        goals = []
        p_orig = float(self.cfg.get("relabel_p_orig", 0.4))
        p_ach = float(self.cfg.get("relabel_p_ach", 0.4))
        p_mkt = float(self.cfg.get("relabel_p_mkt", 0.0))
        if use_ms and p_mkt <= 0.0:
            p_mkt = float(self.cfg.get("ms_her_p_mkt", 0.10))
        # remaining → future
        for tr in transitions:
            delta = self._achieved_delta(tr)
            if mode in ("off", "none", "false"):
                goals.append(np.asarray(tr.goal, dtype=np.float32))
                continue
            if mode == "legacy":
                # 旧实现：候选含实际 Δ 再取最近 → 近似恒为 Δ；保留作消融
                if delta is None:
                    goals.append(np.asarray(tr.goal, dtype=np.float32))
                else:
                    cands = [tr.goal, delta]
                    for _ in range(max(int(self.cfg.get("relabel_candidates", 8)) - 2, 0)):
                        cands.append(delta + np.random.randn(self.goal_dim).astype(np.float32) * 0.02)
                    best = min(cands, key=lambda g: float(np.linalg.norm(np.asarray(g) - delta)))
                    goals.append(clip_goal(np.asarray(best, dtype=np.float32), self.goal_low, self.goal_high))
                continue
            # her_mix / ms_her
            if use_ms:
                w_tou, w_econ = self._ms_her_weights(tr)
                # 归一化四类：orig, ach, fut, mkt
                w_orig = max(p_orig, 1e-6)
                w_a = max(p_ach * w_tou, 1e-6)
                w_f = max((1.0 - p_orig - p_ach) * w_econ, 1e-6)
                w_k = max(p_mkt, 0.0)
                z = w_orig + w_a + w_f + w_k
                u = float(np.random.rand()) * z
                if u < w_orig or delta is None:
                    g = np.asarray(tr.goal, dtype=np.float32)
                elif u < w_orig + w_a:
                    g = delta
                elif u < w_orig + w_a + w_f:
                    fut = self._sample_future_achieved(tr)
                    g = fut if fut is not None else delta
                else:
                    # 市场 prior 候选：用原始 goal 与达成目标的凸组合近似 g^mkt
                    base = np.asarray(tr.goal, dtype=np.float32)
                    if delta is not None:
                        g = 0.5 * base + 0.5 * np.asarray(delta, dtype=np.float32)
                    else:
                        g = base
            else:
                u = float(np.random.rand())
                if u < p_orig or delta is None:
                    g = np.asarray(tr.goal, dtype=np.float32)
                elif u < p_orig + p_ach:
                    g = delta
                else:
                    fut = self._sample_future_achieved(tr)
                    g = fut if fut is not None else delta
            goals.append(clip_goal(np.asarray(g, dtype=np.float32), self.goal_low, self.goal_high))
        return np.stack(goals).astype(np.float32)

    def update_high(self, batch_size: int) -> dict[str, float]:
        if len(self.hi_buffer) < batch_size:
            return {}
        self.hi_it += 1
        b = self.hi_buffer.sample_batch(batch_size)
        obs = self._prep_obs(torch.as_tensor(b["obs"], device=self.device))
        next_obs = self._prep_obs(torch.as_tensor(b["next_obs"], device=self.device))
        reward = torch.as_tensor(b["reward"], device=self.device).clamp(*self.reward_clip)
        done = torch.as_tensor(b["done"], device=self.device)
        if self.cfg.get("goal_relabel", True):
            goal_np = self._relabel_goals(b["transitions"])
        else:
            goal_np = b["goal"]
        goal = torch.as_tensor(goal_np, device=self.device)

        self.hi_actor.train()
        self.hi_critic.train()
        with torch.no_grad():
            ng = self.hi_actor_t(next_obs, self._goal_low_t.expand(next_obs.size(0), -1), self._goal_high_t.expand(next_obs.size(0), -1))
            ng = ng + (torch.randn_like(ng) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
            ng = torch.maximum(torch.minimum(ng, self._goal_high_t.expand_as(ng)), self._goal_low_t.expand_as(ng))
            q1t, q2t = self.hi_critic_t(next_obs, ng)
            # SMDP 折扣 γ^c；bootstrap 硬夹 + 整体 target 再夹，防止目标网络滞后爆炸
            boot = torch.min(q1t, q2t).clamp(-self.q_clip_high, self.q_clip_high)
            target = (reward + (1.0 - done) * self.gamma_high * boot).clamp(
                -self.q_clip_high, self.q_clip_high
            )

        q1, q2 = self.hi_critic(obs, goal)
        loss = F.smooth_l1_loss(q1, target) + F.smooth_l1_loss(q2, target)
        self.hi_critic_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.hi_critic.parameters(), 5.0)
        self.hi_critic_opt.step()

        actor_loss_v = 0.0
        if self.hi_it % self.policy_delay == 0:
            g_pi = self.hi_actor(obs, self._goal_low_t.expand(obs.size(0), -1), self._goal_high_t.expand(obs.size(0), -1))
            q_pi = self.hi_critic.q1_only(obs, g_pi)
            aloss = -q_pi.clamp(-self.q_clip_high, self.q_clip_high).mean()
            self.hi_actor_opt.zero_grad()
            aloss.backward()
            torch.nn.utils.clip_grad_norm_(self.hi_actor.parameters(), 5.0)
            self.hi_actor_opt.step()
            self._soft(self.hi_actor, self.hi_actor_t)
            self._soft(self.hi_critic, self.hi_critic_t)
            actor_loss_v = float(aloss.item())

        metrics = {
            "hi_critic_loss": float(loss.item()),
            "hi_actor_loss": actor_loss_v,
            "hi_q1_mean": float(q1.mean().item()),
            "hi_gamma_c": float(self.gamma_high),
        }
        self.last_metrics.update(metrics)
        return metrics

    def _soft(self, src: torch.nn.Module, tgt: torch.nn.Module) -> None:
        for sp, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "hi_actor": self.hi_actor.state_dict(),
                "hi_critic": self.hi_critic.state_dict(),
                "lo_actor": self.lo_actor.state_dict(),
                "lo_critic": self.lo_critic.state_dict(),
                "hi_it": self.hi_it,
                "lo_it": self.lo_it,
                "cfg": self.cfg,
            },
            path,
        )

    def load(self, path: str | Path, *, strict: bool = True) -> None:
        data = torch.load(path, map_location=self.device, weights_only=False)
        # 底层优先严格加载；高层允许 partial（输入维扩展时）
        self.lo_actor.load_state_dict(data["lo_actor"], strict=strict)
        self.lo_critic.load_state_dict(data["lo_critic"], strict=strict)
        try:
            self.hi_actor.load_state_dict(data["hi_actor"], strict=strict)
            self.hi_critic.load_state_dict(data["hi_critic"], strict=strict)
        except RuntimeError:
            self.hi_actor.load_state_dict(data["hi_actor"], strict=False)
            self.hi_critic.load_state_dict(data["hi_critic"], strict=False)
        self.hi_actor_t = deepcopy(self.hi_actor)
        self.hi_critic_t = deepcopy(self.hi_critic)
        self.lo_actor_t = deepcopy(self.lo_actor)
        self.lo_critic_t = deepcopy(self.lo_critic)
        self.hi_it = int(data.get("hi_it", 0))
        self.lo_it = int(data.get("lo_it", 0))
        # 关键优化器绑定当前参数（避免 resume 后 Adam 状态/引用异常导致 actor 不更新）
        alr = float(self.cfg.get("actor_lr", 3e-4))
        clr = float(self.cfg.get("critic_lr", 3e-4))
        self.hi_actor_opt = torch.optim.Adam(self.hi_actor.parameters(), lr=alr)
        self.hi_critic_opt = torch.optim.Adam(self.hi_critic.parameters(), lr=clr)
        self.lo_actor_opt = torch.optim.Adam(self.lo_actor.parameters(), lr=alr)
        self.lo_critic_opt = torch.optim.Adam(self.lo_critic.parameters(), lr=clr)
