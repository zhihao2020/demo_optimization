"""GHTD3 双层 TD3 agent。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .buffers import HighReplayBuffer, HighTransition, LowReplayBuffer
from .goals import actual_delta_soc, clip_goal, extract_soc_from_obs
from .networks import HighLevelActor, HighLevelCritic, LowLevelActor, LowLevelCritic


class GHTD3Agent:
    def __init__(self, obs_dim: int, cfg: dict[str, Any], device: str | None = None):
        self.cfg = dict(cfg)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.goal_dim = int(cfg.get("goal_dim", 2))
        self.goal_low = np.asarray(cfg.get("goal_low", [-0.25, -0.15]), dtype=np.float32)
        self.goal_high = np.asarray(cfg.get("goal_high", [0.25, 0.15]), dtype=np.float32)
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

        gl = torch.as_tensor(self.goal_low, device=self.device)
        gh = torch.as_tensor(self.goal_high, device=self.device)
        self._goal_low_t = gl
        self._goal_high_t = gh

        self.hi_actor = HighLevelActor(obs_dim, self.goal_dim).to(self.device)
        self.hi_actor_t = deepcopy(self.hi_actor).to(self.device)
        self.hi_critic = HighLevelCritic(obs_dim, self.goal_dim).to(self.device)
        self.hi_critic_t = deepcopy(self.hi_critic).to(self.device)
        self.lo_actor = LowLevelActor(obs_dim, self.goal_dim).to(self.device)
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
            g = self.hi_actor(o, self._goal_low_t.unsqueeze(0), self._goal_high_t.unsqueeze(0))
            g = g.cpu().numpy().ravel()
        if not deterministic and self.high_noise > 0:
            g = g + np.random.randn(self.goal_dim).astype(np.float32) * self.high_noise
        return clip_goal(g, self.goal_low, self.goal_high)

    def select_low_action(
        self,
        obs: np.ndarray,
        goal: np.ndarray,
        feasible,
        *,
        deterministic: bool = False,
    ) -> dict:
        self.lo_actor.eval()
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
            g = torch.as_tensor(goal, dtype=torch.float32, device=self.device).view(1, -1)
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

    def update_low(self, batch_size: int) -> dict[str, float]:
        if len(self.lo_buffer) < batch_size:
            return {}
        self.lo_it += 1
        b = self.lo_buffer.sample_batch(batch_size)
        obs = torch.as_tensor(b["obs"], device=self.device)
        goal = torch.as_tensor(b["goal"], device=self.device)
        next_obs = torch.as_tensor(b["next_obs"], device=self.device)
        next_goal = torch.as_tensor(b["next_goal"], device=self.device)
        u_tp = torch.as_tensor(b["u_tp"], device=self.device)
        u_bat = torch.as_tensor(b["u_battery"], device=self.device)
        mode = torch.as_tensor(b["caes_mode"], device=self.device, dtype=torch.int64)
        mode_oh = F.one_hot(mode, 3).float()
        mag = torch.as_tensor(b["caes_magnitude"], device=self.device)
        reward = torch.as_tensor(b["reward"], device=self.device).clamp(*self.reward_clip)
        done = torch.as_tensor(b["done"], device=self.device)
        next_mask = torch.as_tensor(b["next_mode_mask"], device=self.device)

        with torch.no_grad():
            na = self.lo_actor_t.act(
                next_obs,
                next_goal,
                torch.as_tensor(b["next_u_tp_low"], device=self.device),
                torch.as_tensor(b["next_u_tp_high"], device=self.device),
                torch.as_tensor(b["next_u_bat_low"], device=self.device),
                torch.as_tensor(b["next_u_bat_high"], device=self.device),
                next_mask,
                deterministic=False,
                explore_noise_std=0.0,
            )
            n_tp = torch.clamp(
                na["u_tp"] + (torch.randn_like(na["u_tp"]) * self.target_noise).clamp(-self.noise_clip, self.noise_clip),
                torch.as_tensor(b["next_u_tp_low"], device=self.device),
                torch.as_tensor(b["next_u_tp_high"], device=self.device),
            )
            n_bat = torch.clamp(
                na["u_battery"]
                + (torch.randn_like(na["u_battery"]) * self.target_noise).clamp(-self.noise_clip, self.noise_clip),
                torch.as_tensor(b["next_u_bat_low"], device=self.device),
                torch.as_tensor(b["next_u_bat_high"], device=self.device),
            )
            n_mag = torch.clamp(
                na["caes_magnitude"]
                + (torch.randn_like(na["caes_magnitude"]) * self.target_noise).clamp(-self.noise_clip, self.noise_clip),
                0.0,
                1.0,
            )
            q1t, q2t = self.lo_critic_t(next_obs, next_goal, n_tp, n_bat, na["caes_mode_oh"], n_mag)
            target = reward + (1.0 - done) * self.gamma * torch.min(q1t, q2t).clamp(-self.q_clip, self.q_clip)

        q1, q2 = self.lo_critic(obs, goal, u_tp, u_bat, mode_oh, mag)
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
                torch.as_tensor(b["u_tp_low"], device=self.device),
                torch.as_tensor(b["u_tp_high"], device=self.device),
                torch.as_tensor(b["u_bat_low"], device=self.device),
                torch.as_tensor(b["u_bat_high"], device=self.device),
                torch.as_tensor(b["mode_mask"], device=self.device),
                deterministic=False,
                explore_noise_std=0.0,
            )
            q_pi = self.lo_critic.q1_only(
                obs, goal, cur["u_tp"], cur["u_battery"], cur["caes_mode_oh"], cur["caes_magnitude"]
            )
            # tanh 压缩后最大化，避免 actor 追逐无界 Q
            aloss = -torch.tanh(q_pi / max(self.q_clip, 1.0)).mean()
            self.lo_actor_opt.zero_grad()
            aloss.backward()
            torch.nn.utils.clip_grad_norm_(self.lo_actor.parameters(), 10.0)
            self.lo_actor_opt.step()
            self._soft(self.lo_actor, self.lo_actor_t)
            self._soft(self.lo_critic, self.lo_critic_t)
            actor_loss_v = float(aloss.item())

        metrics = {
            "lo_critic_loss": float(loss.item()),
            "lo_actor_loss": actor_loss_v,
            "lo_q1_mean": float(q1.mean().item()),
        }
        self.last_metrics.update(metrics)
        return metrics

    def _relabel_goals(self, transitions: list[HighTransition]) -> np.ndarray:
        """用实际累计 ΔSoC 作为主候选，贴近论文 relabel 的“与低层行为一致”思想。"""
        goals = []
        for tr in transitions:
            if tr.soc_seq and len(tr.soc_seq) >= 2:
                delta = actual_delta_soc(tr.soc_seq[0], tr.soc_seq[-1])
                # 候选：原 goal、实际增量、加噪实际增量
                cands = [tr.goal, delta]
                for _ in range(max(int(self.cfg.get("relabel_candidates", 8)) - 2, 0)):
                    cands.append(delta + np.random.randn(self.goal_dim).astype(np.float32) * 0.02)
                # 选与实际 delta 最近的（最大似然高斯近似）
                best = min(cands, key=lambda g: float(np.linalg.norm(np.asarray(g) - delta)))
                goals.append(clip_goal(np.asarray(best, dtype=np.float32), self.goal_low, self.goal_high))
            else:
                goals.append(tr.goal)
        return np.stack(goals).astype(np.float32)

    def update_high(self, batch_size: int) -> dict[str, float]:
        if len(self.hi_buffer) < batch_size:
            return {}
        self.hi_it += 1
        b = self.hi_buffer.sample_batch(batch_size)
        obs = torch.as_tensor(b["obs"], device=self.device)
        next_obs = torch.as_tensor(b["next_obs"], device=self.device)
        reward = torch.as_tensor(b["reward"], device=self.device).clamp(*self.reward_clip)
        done = torch.as_tensor(b["done"], device=self.device)
        if self.cfg.get("goal_relabel", True):
            goal_np = self._relabel_goals(b["transitions"])
        else:
            goal_np = b["goal"]
        goal = torch.as_tensor(goal_np, device=self.device)

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
            aloss = -torch.tanh(q_pi / max(self.q_clip_high, 1.0)).mean()
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

    def load(self, path: str | Path) -> None:
        data = torch.load(path, map_location=self.device)
        self.hi_actor.load_state_dict(data["hi_actor"])
        self.hi_critic.load_state_dict(data["hi_critic"])
        self.lo_actor.load_state_dict(data["lo_actor"])
        self.lo_critic.load_state_dict(data["lo_critic"])
        self.hi_actor_t = deepcopy(self.hi_actor)
        self.hi_critic_t = deepcopy(self.hi_critic)
        self.lo_actor_t = deepcopy(self.lo_actor)
        self.lo_critic_t = deepcopy(self.lo_critic)
        self.hi_it = int(data.get("hi_it", 0))
        self.lo_it = int(data.get("lo_it", 0))
