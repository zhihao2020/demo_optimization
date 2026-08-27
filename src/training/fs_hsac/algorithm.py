"""FS-HSAC: exact hybrid-entropy SAC with dual temperatures."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.fs_hsac.action_support import MODE_CHARGE, MODE_DISCHARGE, MODE_IDLE
from training.fs_hsac.actor import FSHSACActor
from training.fs_hsac.critic import FSHSACCritic

ALGORITHM_VERSION = "fs_hsac_v2"


class FSHSAC:
    """Feasible-support hybrid SAC with exact mode enumeration and dual alpha."""

    def __init__(
        self,
        obs_dim: int,
        *,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        device: str | None = None,
        alpha_min: float = 1e-4,
        alpha_max: float = 10.0,
        q_clip: float = 200.0,
        skip_nonfinite_update: bool = True,
        use_feasibility_penalty: bool = False,
        feasibility_beta: float = 0.1,
        target_entropy_cont: float | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.q_clip = float(q_clip)
        self.skip_nonfinite_update = bool(skip_nonfinite_update)
        self.use_feasibility_penalty = bool(use_feasibility_penalty)
        self.feasibility_beta = float(feasibility_beta)
        self.obs_dim = int(obs_dim)
        self._actor_lr = float(actor_lr)
        self._critic_lr = float(critic_lr)
        self._alpha_lr = float(alpha_lr)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.log_alpha_min = float(np.log(self.alpha_min))
        self.log_alpha_max = float(np.log(self.alpha_max))
        # continuous target entropy: default -3 (tp, bat, mag); idle uses -2 internally via cont_dim
        self.target_entropy_cont = float(-3.0 if target_entropy_cont is None else target_entropy_cont)

        self.actor = FSHSACActor(obs_dim).to(self.device)
        self.critic = FSHSACCritic(obs_dim).to(self.device)
        self.critic_target = deepcopy(self.critic).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.log_alpha_d = torch.zeros(1, requires_grad=True, device=self.device)
        self.log_alpha_c = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_d_opt = torch.optim.Adam([self.log_alpha_d], lr=alpha_lr)
        self.alpha_c_opt = torch.optim.Adam([self.log_alpha_c], lr=alpha_lr)
        self.total_it = 0
        self.nonfinite_skips = 0
        self.last_metrics: dict[str, float] = {}
        self.feasibility_net = None  # optional nn.Module set by train loop

    @property
    def alpha_d(self) -> torch.Tensor:
        return self.log_alpha_d.clamp(self.log_alpha_min, self.log_alpha_max).exp()

    @property
    def alpha_c(self) -> torch.Tensor:
        return self.log_alpha_c.clamp(self.log_alpha_min, self.log_alpha_max).exp()

    def _clamp_log_alpha_(self) -> None:
        with torch.no_grad():
            self.log_alpha_d.clamp_(self.log_alpha_min, self.log_alpha_max)
            self.log_alpha_c.clamp_(self.log_alpha_min, self.log_alpha_max)

    def select_action(self, obs, feasible, deterministic: bool = False) -> dict:
        return self.actor.act_numpy(obs, feasible, deterministic=deterministic, device=self.device)

    def _support_from_batch(self, batch: dict, *, next_state: bool = False) -> dict[str, torch.Tensor]:
        p = "next_" if next_state else ""
        return {
            "mode_mask": torch.as_tensor(batch[f"{p}mode_mask"], device=self.device),
            "u_tp_low": torch.as_tensor(batch[f"{p}u_tp_low"], device=self.device),
            "u_tp_high": torch.as_tensor(batch[f"{p}u_tp_high"], device=self.device),
            "u_bat_low": torch.as_tensor(batch[f"{p}u_bat_low"], device=self.device),
            "u_bat_high": torch.as_tensor(batch[f"{p}u_bat_high"], device=self.device),
            "dis_lo": torch.as_tensor(batch[f"{p}dis_lo"], device=self.device),
            "dis_hi": torch.as_tensor(batch[f"{p}dis_hi"], device=self.device),
            "chg_lo": torch.as_tensor(batch[f"{p}chg_lo"], device=self.device),
            "chg_hi": torch.as_tensor(batch[f"{p}chg_hi"], device=self.device),
        }

    def _exact_soft_value(
        self,
        obs: torch.Tensor,
        support: dict[str, torch.Tensor],
        *,
        target: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (soft_q, entropy_mode, entropy_cont) with exact mode sum."""
        critic = self.critic_target if target else self.critic
        # recompute with actor heads for consistency under no_grad/grad
        h = self.actor._heads(obs)
        logits, legal = self.actor._mask_logits(h["mode_logits"], support["mode_mask"])
        probs = torch.softmax(logits, dim=-1)
        soft_terms = []
        ent_c_terms = []
        for mode_idx in (MODE_DISCHARGE, MODE_IDLE, MODE_CHARGE):
            samp = self.actor.sample_mode_action(
                obs, support, mode_idx, deterministic=False, heads=h
            )
            q1, q2 = critic(
                obs,
                samp["u_tp"],
                samp["u_battery"],
                samp["mode_onehot"],
                samp["mag"],
            )
            min_q = torch.min(q1, q2)
            if self.q_clip > 0:
                min_q = min_q.clamp(-self.q_clip, self.q_clip)
            # soft Q contribution for this mode
            term = min_q - self.alpha_c.detach() * samp["log_prob_cont"]
            soft_terms.append(probs[:, mode_idx] * term)
            ent_c_terms.append(probs[:, mode_idx] * samp["entropy_cont"])
        soft_q = soft_terms[0] + soft_terms[1] + soft_terms[2]
        # subtract discrete entropy bonus: E[Q - α_d log π_d - α_c log π_c]
        log_probs = probs.clamp_min(1e-8).log()
        soft_q = soft_q - self.alpha_d.detach() * (probs * log_probs).sum(dim=-1)
        ent_mode = -(probs * log_probs).sum(dim=-1)
        ent_cont = ent_c_terms[0] + ent_c_terms[1] + ent_c_terms[2]
        # zero illegal modes already handled by masked softmax
        soft_q = soft_q * legal.any(dim=-1).float()
        return soft_q, ent_mode, ent_cont

    def _feasibility_penalty(self, obs, u_tp, u_bat, mode_onehot, mag) -> torch.Tensor:
        if self.feasibility_net is None or not self.use_feasibility_penalty:
            return torch.zeros(obs.size(0), device=obs.device)
        # Cψ -> -log Cψ
        logit = self.feasibility_net(obs, u_tp, u_bat, mode_onehot, mag)
        prob = torch.sigmoid(logit).clamp(1e-4, 1.0 - 1e-4)
        return -torch.log(prob)

    def update(self, buffer, batch_size: int = 256) -> dict[str, float]:
        if len(buffer) < batch_size:
            return {}
        self.total_it += 1
        batch = buffer.sample_bellman(batch_size)
        obs = torch.as_tensor(batch["obs"], device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], device=self.device)
        reward = torch.as_tensor(batch["reward"], device=self.device)
        done = torch.as_tensor(batch["done"], device=self.device)
        support = self._support_from_batch(batch, next_state=False)
        next_support = self._support_from_batch(batch, next_state=True)

        # Critic: only physical FMU transitions
        with torch.no_grad():
            soft_next, _, _ = self._exact_soft_value(next_obs, next_support, target=True)
            if self.q_clip > 0:
                soft_next = soft_next.clamp(-self.q_clip, self.q_clip)
            target_q = reward + (1.0 - done) * self.gamma * soft_next
            if self.q_clip > 0:
                target_q = target_q.clamp(-self.q_clip, self.q_clip)

        q1, q2 = self.critic.q_from_physical(
            obs,
            torch.as_tensor(batch["u_tp"], device=self.device),
            torch.as_tensor(batch["u_battery"], device=self.device),
            torch.as_tensor(batch["u_caes"], device=self.device),
            dis_lo=support["dis_lo"],
            dis_hi=support["dis_hi"],
            chg_lo=support["chg_lo"],
            chg_hi=support["chg_hi"],
        )
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        if not torch.isfinite(critic_loss):
            self.nonfinite_skips += 1
            if self.skip_nonfinite_update:
                self._clamp_log_alpha_()
                return {"critic_loss": float("nan"), "skipped": 1.0, "nonfinite_skips": float(self.nonfinite_skips)}
            raise RuntimeError("FS-HSAC critic_loss non-finite")
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_opt.step()

        # Actor: exact mode expectation
        h = self.actor._heads(obs)
        logits, legal = self.actor._mask_logits(h["mode_logits"], support["mode_mask"])
        probs = torch.softmax(logits, dim=-1)
        actor_terms = []
        ent_mode = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
        ent_cont_acc = torch.zeros(obs.size(0), device=self.device)
        log_cont_acc = torch.zeros(obs.size(0), device=self.device)
        for mode_idx in (MODE_DISCHARGE, MODE_IDLE, MODE_CHARGE):
            samp = self.actor.sample_mode_action(
                obs, support, mode_idx, deterministic=False, heads=h
            )
            q1_pi, q2_pi = self.critic(
                obs, samp["u_tp"], samp["u_battery"], samp["mode_onehot"], samp["mag"]
            )
            min_q = torch.min(q1_pi, q2_pi)
            if self.q_clip > 0:
                min_q = min_q.clamp(-self.q_clip, self.q_clip)
            pen = self._feasibility_penalty(
                obs, samp["u_tp"], samp["u_battery"], samp["mode_onehot"], samp["mag"]
            )
            # J = E_k[ α_d log π_d + α_c log π_c - Q + β (-log C) ]
            # with exact sum over modes
            term = (
                self.alpha_d.detach() * probs[:, mode_idx].clamp_min(1e-8).log()
                + self.alpha_c.detach() * samp["log_prob_cont"]
                - min_q
                + self.feasibility_beta * pen
            )
            actor_terms.append(probs[:, mode_idx] * term)
            ent_cont_acc = ent_cont_acc + probs[:, mode_idx] * samp["entropy_cont"]
            log_cont_acc = log_cont_acc + probs[:, mode_idx] * samp["log_prob_cont"]
        actor_loss = (actor_terms[0] + actor_terms[1] + actor_terms[2]).mean()
        if not torch.isfinite(actor_loss):
            self.nonfinite_skips += 1
            self._clamp_log_alpha_()
            if self.skip_nonfinite_update:
                return {
                    "critic_loss": float(critic_loss.item()),
                    "actor_loss": float("nan"),
                    "skipped": 1.0,
                    "nonfinite_skips": float(self.nonfinite_skips),
                }
            raise RuntimeError("FS-HSAC actor_loss non-finite")
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()

        # Dual temperatures
        # discrete target entropy: -H_uniform = -log(|K|)
        n_modes = legal.sum(dim=-1).clamp_min(1).to(dtype=torch.float32)
        target_ent_d = -torch.log(n_modes)
        # stop-grad on entropy estimates from fresh forward after actor step
        with torch.no_grad():
            h2 = self.actor._heads(obs)
            logits2, legal2 = self.actor._mask_logits(h2["mode_logits"], support["mode_mask"])
            probs2 = torch.softmax(logits2, dim=-1)
            ent_mode2 = -(probs2 * probs2.clamp_min(1e-8).log()).sum(dim=-1)
            ent_cont2 = torch.zeros(obs.size(0), device=self.device)
            for mode_idx in (MODE_DISCHARGE, MODE_IDLE, MODE_CHARGE):
                samp = self.actor.sample_mode_action(
                    obs, support, mode_idx, deterministic=False, heads=h2
                )
                ent_cont2 = ent_cont2 + probs2[:, mode_idx] * samp["entropy_cont"]
            # continuous target scales with expected cont dim ≈ 2 + P(non-idle)
            p_cont = (probs2[:, MODE_DISCHARGE] + probs2[:, MODE_CHARGE]).clamp(0.0, 1.0)
            target_ent_c = -2.0 - p_cont  # approx -cont_dim

        alpha_d_loss = -(self.log_alpha_d * (ent_mode2 + target_ent_d).detach()).mean()
        alpha_c_loss = -(self.log_alpha_c * (ent_cont2 + target_ent_c).detach()).mean()
        self.alpha_d_opt.zero_grad()
        alpha_d_loss.backward()
        self.alpha_d_opt.step()
        self.alpha_c_opt.zero_grad()
        alpha_c_loss.backward()
        self.alpha_c_opt.step()
        self._clamp_log_alpha_()

        self._soft_update(self.critic, self.critic_target)
        metrics = {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_d": float(self.alpha_d.item()),
            "alpha_c": float(self.alpha_c.item()),
            "alpha_d_loss": float(alpha_d_loss.item()),
            "alpha_c_loss": float(alpha_c_loss.item()),
            "entropy_mode": float(ent_mode2.mean().item()),
            "entropy_cont": float(ent_cont2.mean().item()),
            "q1_mean": float(q1.mean().item()),
            "nonfinite_skips": float(self.nonfinite_skips),
            "skipped": 0.0,
        }
        self.last_metrics = metrics
        return metrics

    def _soft_update(self, src: torch.nn.Module, tgt: torch.nn.Module) -> None:
        for sp, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "algorithm_version": ALGORITHM_VERSION,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha_d": self.log_alpha_d.detach().cpu(),
            "log_alpha_c": self.log_alpha_c.detach().cpu(),
            "total_it": self.total_it,
            "obs_dim": self.obs_dim,
            "use_feasibility_penalty": self.use_feasibility_penalty,
            "feasibility_beta": self.feasibility_beta,
        }
        if self.feasibility_net is not None:
            payload["feasibility_net"] = self.feasibility_net.state_dict()
        torch.save(payload, path)

    def load(self, path: str | Path) -> None:
        data = torch.load(path, map_location=self.device, weights_only=False)
        ver = data.get("algorithm_version")
        if ver != ALGORITHM_VERSION:
            raise RuntimeError(
                f"Incompatible checkpoint algorithm_version={ver!r}; expected {ALGORITHM_VERSION!r}"
            )
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
        self.critic_target.load_state_dict(data["critic_target"])
        self.log_alpha_d = data["log_alpha_d"].to(self.device).clone().detach().requires_grad_(True)
        self.log_alpha_c = data["log_alpha_c"].to(self.device).clone().detach().requires_grad_(True)
        self.alpha_d_opt = torch.optim.Adam([self.log_alpha_d], lr=self._alpha_lr)
        self.alpha_c_opt = torch.optim.Adam([self.log_alpha_c], lr=self._alpha_lr)
        self.total_it = int(data.get("total_it", 0))
        self.use_feasibility_penalty = bool(data.get("use_feasibility_penalty", False))
        self.feasibility_beta = float(data.get("feasibility_beta", self.feasibility_beta))
        if "feasibility_net" in data and self.feasibility_net is not None:
            self.feasibility_net.load_state_dict(data["feasibility_net"])
