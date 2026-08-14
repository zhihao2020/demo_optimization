"""HMSD / GHTD3: absolute goal-conditioned hierarchical TD3 (no Hybrid teacher)."""

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
    battery_soc_discharge,
    clip_goal,
    default_goal_boxes,
    enforce_budget_on_action,
    goal_budget_layout,
)
from .networks import (
    HighLevelActor,
    HighLevelCritic,
    LowLevelActor,
    LowLevelCritic,
)


class GHTD3Agent:
    def __init__(self, obs_dim: int, cfg: dict[str, Any], device: str | None = None):
        self.cfg = dict(cfg)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.goal_dim = int(cfg.get("goal_dim", 2))
        self.wear_budget = bool(cfg.get("wear_budget", False))
        self.thermal_budget = bool(cfg.get("thermal_budget", False))
        self.caes_budget = bool(cfg.get("caes_budget", False))
        self.hybrid_caes = bool(cfg.get("hybrid_caes", False))
        self.goal_layout = goal_budget_layout(
            wear_budget=self.wear_budget,
            caes_budget=self.caes_budget,
            thermal_budget=self.thermal_budget,
        )
        dlow, dhigh = default_goal_boxes(
            self.goal_dim,
            wear_budget=self.wear_budget,
            thermal_budget=self.thermal_budget,
            caes_budget=self.caes_budget,
        )
        gl_cfg = cfg.get("goal_low")
        gh_cfg = cfg.get("goal_high")
        if gl_cfg is None or len(np.asarray(gl_cfg).ravel()) != self.goal_dim:
            self.goal_low = dlow.astype(np.float32)
            self.goal_high = dhigh.astype(np.float32)
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
        # 高层 reward 为 c 步均值后尺度接近底层，q_clip 可收紧
        self.q_clip = float(cfg.get("q_clip", 200.0))
        self.q_clip_high = float(cfg.get("q_clip_high", min(self.q_clip, 50.0)))
        # Training-time feasible span floor. Large values invent fake action range.
        # 0 disables widening (prefer true oracle bounds).
        self.min_span_tp = float(cfg.get("min_span_tp", 0.02))
        self.min_span_bat = float(cfg.get("min_span_bat", 0.05))
        self.clamp_high_actor_q = bool(cfg.get("clamp_high_actor_q", False))
        self.high_noise = float(cfg.get("high_explore_noise", 0.05))
        self.low_noise = float(cfg.get("low_explore_noise", 0.08))
        # 高层 reward = mean_ext over cycle（稳定 critic；相对论文 sum 的实现改进）
        self.high_reward_normalize = bool(cfg.get("high_reward_normalize", True))
        # HMSD mainline: absolute goal-conditioned (no Hybrid teacher / residual)
        self.execution_mode = str(cfg.get("execution_mode", "goal_conditioned")).lower()
        if self.execution_mode not in ("goal_conditioned", "gc", "absolute", ""):
            raise ValueError(
                f"HMSD mainline only supports execution_mode=goal_conditioned; got {self.execution_mode!r}."
            )
        self.execution_mode = "goal_conditioned"
        self.hybrid_anchor_enabled = False
        self._hybrid_anchor = None
        self.low_level_algo = "td3"
        self.low_use_raw_obs = bool(cfg.get("low_use_raw_obs", False))
        self._train_progress = 0.0
        self.high_lambda_soc = bool(cfg.get("high_lambda_soc", False))
        self.ltar_enabled = False
        self.lambda_soc = float(cfg.get("ltar_lambda_init", 0.0))
        self.ltar_lambda_lr = float(cfg.get("ltar_lambda_lr", 0.15))
        self.ltar_lambda_max = float(cfg.get("ltar_lambda_max", 8.0))
        self.ltar_cost_tol = float(cfg.get("ltar_cost_tol", 0.0))

        gl = torch.as_tensor(self.goal_low, device=self.device)
        gh = torch.as_tensor(self.goal_high, device=self.device)
        self._goal_low_t = gl
        self._goal_high_t = gh

        residual_init = bool(cfg.get("residual_init", False))
        goal_scale = float(cfg.get("goal_input_scale", 4.0))
        self.hi_actor = HighLevelActor(obs_dim, self.goal_dim).to(self.device)
        self.hi_actor_t = deepcopy(self.hi_actor).to(self.device)
        self.hi_critic = HighLevelCritic(obs_dim, self.goal_dim).to(self.device)
        self.hi_critic_t = deepcopy(self.hi_critic).to(self.device)
        self.lo_actor = LowLevelActor(
            obs_dim,
            self.goal_dim,
            residual_init=residual_init,
            goal_input_scale=goal_scale,
            hybrid_caes=self.hybrid_caes,
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

    def _prep_obs_low(self, obs: torch.Tensor) -> torch.Tensor:
        if self.low_use_raw_obs:
            return obs
        return self._prep_obs(obs)

    def set_progress(self, progress: float) -> None:
        """训练进度 ∈[0,1]（日志 / prior 退火；无教师扩张逻辑）。"""
        self._train_progress = float(np.clip(progress, 0.0, 1.0))

    def update_lambda_soc(self, cost: float) -> float:
        """高层 λ-SoC 对偶更新：cost = [‖z_T−z_0‖−ξ]_+ 或 1{SOC 失败}。"""
        if not getattr(self, "high_lambda_soc", False):
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


    def select_low_action(
        self,
        obs: np.ndarray,
        goal: np.ndarray,
        feasible,
        *,
        deterministic: bool = False,
    ) -> dict:
        """Goal-conditioned low level: TD3 deterministic head or SAC stochastic hybrid."""
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
            mask = torch.as_tensor(
                feasible.mode_mask.as_bool_array(), dtype=torch.bool, device=self.device
            ).view(1, 3)
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
        from actions.caes_u import physical_dict
        return physical_dict(
            float(out["u_tp"][0].cpu()),
            float(out["u_battery"][0].cpu()),
            float(out["u_caes"][0].cpu()),
        )

    def select_composed_action(
        self,
        obs: np.ndarray,
        goal: np.ndarray,
        feasible,
        *,
        deterministic: bool = False,
        residual_scale: float | None = None,
    ) -> dict:
        """Absolute goal-conditioned action (HMSD mainline). residual_scale ignored."""
        _ = residual_scale
        action = self.select_low_action(obs, goal, feasible, deterministic=deterministic)
        return enforce_budget_on_action(
            action,
            goal,
            wear_budget=self.wear_budget,
            thermal_budget=self.thermal_budget,
            caes_budget=self.caes_budget,
            wear_enforce=bool(self.cfg.get("wear_enforce", True)),
            thermal_enforce=bool(self.cfg.get("thermal_enforce", True)),
            caes_enforce=bool(self.cfg.get("caes_enforce", True)),
            c=self.subgoal_interval,
            layout=self.goal_layout,
        )


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
        u_caes = torch.as_tensor(b["u_caes"], device=self.device)
        reward = torch.as_tensor(b["reward"], device=self.device)
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
        min_span_tp, min_span_bat = float(self.min_span_tp), float(self.min_span_bat)
        if min_span_tp > 0.0:
            tp_hi_b = torch.maximum(tp_hi_b, tp_lo_b + min_span_tp)
            tp_lo_b = torch.minimum(tp_lo_b, tp_hi_b - min_span_tp).clamp(0.0, 1.0)
            tp_hi_b = tp_hi_b.clamp(0.0, 1.0)
            n_tp_hi = torch.maximum(n_tp_hi, n_tp_lo + min_span_tp)
            n_tp_lo = torch.minimum(n_tp_lo, n_tp_hi - min_span_tp).clamp(0.0, 1.0)
            n_tp_hi = n_tp_hi.clamp(0.0, 1.0)
        if min_span_bat > 0.0:
            bat_hi_b = torch.maximum(bat_hi_b, bat_lo_b + min_span_bat)
            bat_lo_b = torch.minimum(bat_lo_b, bat_hi_b - min_span_bat).clamp(-1.0, 1.0)
            bat_hi_b = bat_hi_b.clamp(-1.0, 1.0)
            n_bat_hi = torch.maximum(n_bat_hi, n_bat_lo + min_span_bat)
            n_bat_lo = torch.minimum(n_bat_lo, n_bat_hi - min_span_bat).clamp(-1.0, 1.0)
            n_bat_hi = n_bat_hi.clamp(-1.0, 1.0)
        mask = torch.as_tensor(b["mode_mask"], device=self.device)
        all_false = ~mask.any(dim=-1, keepdim=True)
        mask = torch.where(all_false, torch.ones_like(mask), mask)
        next_mask = torch.where(~next_mask.any(dim=-1, keepdim=True), torch.ones_like(next_mask), next_mask)

        with torch.no_grad():
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
            from actions.caes_u import project_u_caes_torch

            n_caes = project_u_caes_torch(
                torch.clamp(
                    na["u_caes"]
                    + (torch.randn_like(na["u_caes"]) * self.target_noise).clamp(
                        -self.noise_clip, self.noise_clip
                    ),
                    -1.0,
                    1.0,
                )
            )
            q1t, q2t = self.lo_critic_t(next_obs, next_goal, n_tp, n_bat, n_caes)
            target = reward + (1.0 - done) * self.gamma * torch.min(q1t, q2t).clamp(-self.q_clip, self.q_clip)

        self.lo_actor.train()
        self.lo_critic.train()
        q1, q2 = self.lo_critic(obs, goal, u_tp, u_bat, u_caes)
        # 不 clamp 预测 Q：硬夹会切断梯度，反而让 critic 卡在爆炸区
        loss = F.smooth_l1_loss(q1, target) + F.smooth_l1_loss(q2, target)
        self.lo_critic_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.lo_critic.parameters(), 10.0)
        self.lo_critic_opt.step()

        actor_loss_v = 0.0
        if self.lo_it % self.policy_delay == 0:
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
                obs, goal, cur["u_tp"], cur["u_battery"], cur["u_caes"]
            )
            bc_w = float(self.cfg.get("actor_bc_weight", 0.15))
            bc = (
                F.mse_loss(cur["u_tp"], u_tp.detach())
                + F.mse_loss(cur["u_battery"], u_bat.detach())
                + F.mse_loss(cur["u_caes"], u_caes.detach())
            )
            q_term = q_pi.mean() / (q_pi.detach().abs().mean().clamp_min(1.0))
            aloss = -q_term + bc_w * bc
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


    def _wear_from_soc_seq(self, soc_seq: list) -> float:
        wear = 0.0
        for a, b in zip(soc_seq[:-1], soc_seq[1:]):
            wear += battery_soc_discharge(a, b)
        return float(wear)

    def _copy_budget_dim(self, out: np.ndarray, d: np.ndarray, name: str, *, clip01: bool = False) -> None:
        i = self.goal_layout.get(name)
        if i is None or out.size <= i or d.size <= i:
            return
        val = max(0.0, float(d[i]))
        out[i] = min(1.0, val) if clip01 else val

    def _achieved_delta(self, tr: HighTransition) -> np.ndarray | None:
        """Inventory Δ on dims 0–1; budget dims stay quotas, never process-inventory residual."""
        out = np.zeros(self.goal_dim, dtype=np.float32)
        have = False
        if tr.achieved_delta is not None:
            d = np.asarray(tr.achieved_delta, dtype=np.float32).ravel()
            n_inv = min(2, self.goal_dim, d.size)
            if n_inv > 0:
                out[:n_inv] = d[:n_inv]
            self._copy_budget_dim(out, d, "wear")
            self._copy_budget_dim(out, d, "caes", clip01=True)
            self._copy_budget_dim(out, d, "thermal", clip01=True)
            have = True
        elif tr.soc_seq and len(tr.soc_seq) >= 2:
            d = actual_delta_soc(tr.soc_seq[0], tr.soc_seq[-1]).ravel()
            n_inv = min(2, self.goal_dim, d.size)
            if n_inv > 0:
                out[:n_inv] = d[:n_inv]
            have = True
        if not have:
            return None
        iw = self.goal_layout.get("wear")
        if iw is not None and self.goal_dim > iw and float(out[iw]) == 0.0 and tr.soc_seq:
            out[iw] = self._wear_from_soc_seq(tr.soc_seq)
        return out

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

    def _relabel_goals(self, transitions: list[HighTransition]) -> np.ndarray:
        """Historical goal relabeling (plain HER).

        Modes (``goal_relabel_mode``):
        - ``her_mix`` (default): original / achieved / future with p_orig, p_ach, rest→future
        - ``none`` / ``off``: keep original goals
        - ``legacy``: nearest-to-delta candidates (ablation only; nearly collapses to Δ)
        """
        mode = str(self.cfg.get("goal_relabel_mode", "her_mix")).lower()
        # ms_her / market_her names kept as aliases → plain her_mix (no market weights)
        if mode in ("ms_her", "market_her", "ms-her"):
            mode = "her_mix"
        goals = []
        p_orig = float(self.cfg.get("relabel_p_orig", 0.4))
        p_ach = float(self.cfg.get("relabel_p_ach", 0.4))
        for tr in transitions:
            delta = self._achieved_delta(tr)
            if mode in ("off", "none", "false"):
                goals.append(np.asarray(tr.goal, dtype=np.float32))
                continue
            if mode == "legacy":
                if delta is None:
                    goals.append(np.asarray(tr.goal, dtype=np.float32))
                else:
                    cands = [tr.goal, delta]
                    for _ in range(max(int(self.cfg.get("relabel_candidates", 8)) - 2, 0)):
                        cands.append(delta + np.random.randn(self.goal_dim).astype(np.float32) * 0.02)
                    best = min(cands, key=lambda g: float(np.linalg.norm(np.asarray(g) - delta)))
                    goals.append(clip_goal(np.asarray(best, dtype=np.float32), self.goal_low, self.goal_high))
                continue
            # her_mix (default): original / achieved / future
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
        reward = torch.as_tensor(b["reward"], device=self.device)
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
            # Do not clamp actor Q by default: hard clip flattens high-level preference.
            if self.clamp_high_actor_q:
                q_pi = q_pi.clamp(-self.q_clip_high, self.q_clip_high)
            aloss = -q_pi.mean()
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
        payload = {
            "hi_actor": self.hi_actor.state_dict(),
            "hi_critic": self.hi_critic.state_dict(),
            "lo_actor": self.lo_actor.state_dict(),
            "lo_critic": self.lo_critic.state_dict(),
            "hi_it": self.hi_it,
            "lo_it": self.lo_it,
            "cfg": self.cfg,
            "low_level_algo": self.low_level_algo,
        }
        torch.save(payload, path)

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
        # 重建优化器绑定当前参数（避免 resume 后 Adam 状态/引用异常导致 actor 不更新）
        alr = float(self.cfg.get("actor_lr", 3e-4))
        clr = float(self.cfg.get("critic_lr", 3e-4))
        self.hi_actor_opt = torch.optim.Adam(self.hi_actor.parameters(), lr=alr)
        self.hi_critic_opt = torch.optim.Adam(self.hi_critic.parameters(), lr=clr)
        self.lo_actor_opt = torch.optim.Adam(self.lo_actor.parameters(), lr=alr)
        self.lo_critic_opt = torch.optim.Adam(self.lo_critic.parameters(), lr=clr)
