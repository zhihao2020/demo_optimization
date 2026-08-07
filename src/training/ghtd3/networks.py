"""GHTD3 高/低层 Actor-Critic。"""

from __future__ import annotations

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


class HighLevelActor(nn.Module):
    """s -> goal（有界 SoC 增量）。"""

    def __init__(self, obs_dim: int, goal_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.goal_dim = goal_dim
        self.encoder = _mlp(obs_dim, hidden)
        self.head = nn.Linear(hidden, goal_dim)
        # 小目标先验
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs: torch.Tensor, goal_low: torch.Tensor, goal_high: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.head(self.encoder(obs)))
        # map [-1,1] -> [low, high]
        mid = 0.5 * (goal_high + goal_low)
        half = 0.5 * (goal_high - goal_low)
        return mid + z * half


class HighLevelCritic(nn.Module):
    def __init__(self, obs_dim: int, goal_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.q1 = _mlp(obs_dim + goal_dim, hidden)
        self.q1_out = nn.Linear(hidden, 1)
        self.q2 = _mlp(obs_dim + goal_dim, hidden)
        self.q2_out = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor, goal: torch.Tensor):
        x = torch.cat([obs, goal], dim=-1)
        return self.q1_out(self.q1(x)).squeeze(-1), self.q2_out(self.q2(x)).squeeze(-1)

    def q1_only(self, obs: torch.Tensor, goal: torch.Tensor):
        x = torch.cat([obs, goal], dim=-1)
        return self.q1_out(self.q1(x)).squeeze(-1)


class LowLevelActor(nn.Module):
    """[s, g] -> hybrid action（有界火电/电池 + CAES 模式）。

    residual_init=True 时近零输出，便于 Hybrid 锚定残差起步。
    """

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int = 2,
        hidden: int = 256,
        *,
        residual_init: bool = False,
        goal_input_scale: float = 1.0,
    ):
        super().__init__()
        self.goal_dim = int(goal_dim)
        # goal 放大后再 concat，避免相对 163 维 norm-obs 被淹没
        self.goal_input_scale = float(goal_input_scale)
        self.encoder = _mlp(obs_dim + goal_dim, hidden)
        self.thermal_head = nn.Linear(hidden, 1)
        self.battery_head = nn.Linear(hidden, 1)
        self.mode_head = nn.Linear(hidden, 3)
        self.discharge_mag_head = nn.Linear(hidden, 1)
        self.charge_mag_head = nn.Linear(hidden, 1)
        if residual_init:
            # 残差：中等头增益，使扫 g 时 Δ 达 O(0.1)，g=0 附近仍接近 0（无偏置）
            for m in (self.thermal_head, self.battery_head, self.discharge_mag_head, self.charge_mag_head):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                nn.init.zeros_(m.bias)
            nn.init.xavier_uniform_(self.mode_head.weight, gain=0.3)
            with torch.no_grad():
                self.mode_head.bias.zero_()
                self.mode_head.bias[int(CaesMode.IDLE)] = 2.5
                self.mode_head.bias[int(CaesMode.DISCHARGE)] = -1.5
                self.mode_head.bias[int(CaesMode.CHARGE)] = -1.5
            for layer in self.encoder:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=1.0)
                    nn.init.zeros_(layer.bias)
        else:
            nn.init.constant_(self.thermal_head.bias, 2.0)
            nn.init.constant_(self.battery_head.bias, 0.0)
            with torch.no_grad():
                self.mode_head.bias.zero_()
                self.mode_head.bias[int(CaesMode.IDLE)] = 2.0
                self.mode_head.bias[int(CaesMode.DISCHARGE)] = -1.0
                self.mode_head.bias[int(CaesMode.CHARGE)] = -1.0

    @staticmethod
    def map_bounded(z: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        mapped = low + torch.sigmoid(z) * (high - low)
        return torch.minimum(torch.maximum(mapped, low), high)

    def _pack(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        if self.goal_input_scale != 1.0:
            goal = goal * self.goal_input_scale
        return torch.cat([obs, goal], dim=-1)

    def forward_logits(self, obs: torch.Tensor, goal: torch.Tensor) -> dict[str, torch.Tensor]:
        """BC 用：返回 pre-squash logit。"""
        h = self.encoder(self._pack(obs, goal))
        return {
            "z_tp": self.thermal_head(h).squeeze(-1),
            "z_bat": self.battery_head(h).squeeze(-1),
            "logits_mode": self.mode_head(h),
            "z_discharge": self.discharge_mag_head(h).squeeze(-1),
            "z_charge": self.charge_mag_head(h).squeeze(-1),
        }

    def act(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
        u_tp_low: torch.Tensor,
        u_tp_high: torch.Tensor,
        u_bat_low: torch.Tensor,
        u_bat_high: torch.Tensor,
        mode_mask: torch.Tensor,
        *,
        deterministic: bool = False,
        explore_noise_std: float = 0.0,
        gumbel_tau: float = 1.0,
        soft_mode_for_grad: bool = False,
    ) -> dict[str, torch.Tensor]:
        h = self.encoder(self._pack(obs, goal))
        logits = self.mode_head(h).masked_fill(~mode_mask.bool(), -1e9)
        if soft_mode_for_grad:
            # 可微 soft mode，保证 actor 反传不经 hard argmax/Gumbel 断裂
            mode_oh = F.softmax(logits, dim=-1)
            mode = torch.argmax(mode_oh, dim=-1)
        elif deterministic:
            mode = torch.argmax(logits, dim=-1)
            mode_oh = F.one_hot(mode, num_classes=3).float()
        else:
            mode_oh = F.gumbel_softmax(logits, tau=gumbel_tau, hard=True, dim=-1)
            mode = torch.argmax(mode_oh, dim=-1)
        u_tp = self.map_bounded(self.thermal_head(h).squeeze(-1), u_tp_low, u_tp_high)
        u_bat = self.map_bounded(self.battery_head(h).squeeze(-1), u_bat_low, u_bat_high)
        mag_d = torch.sigmoid(self.discharge_mag_head(h).squeeze(-1))
        mag_c = torch.sigmoid(self.charge_mag_head(h).squeeze(-1))
        mag = mode_oh[:, 0] * mag_d + mode_oh[:, 2] * mag_c
        if explore_noise_std > 0 and not deterministic:
            u_tp = torch.clamp(u_tp + explore_noise_std * torch.randn_like(u_tp), u_tp_low, u_tp_high)
            u_bat = torch.clamp(u_bat + explore_noise_std * torch.randn_like(u_bat), u_bat_low, u_bat_high)
            mag = torch.clamp(mag + explore_noise_std * torch.randn_like(mag), 0.0, 1.0)
            mag = torch.where(mode == int(CaesMode.IDLE), torch.zeros_like(mag), mag)
        return {
            "u_tp": u_tp,
            "u_battery": u_bat,
            "caes_mode": mode,
            "caes_mode_oh": mode_oh,
            "caes_magnitude": mag,
            "logits_mode": logits,
        }

    @staticmethod
    def _beta_vec(beta: float | torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """将 β 广播到与 ref 同 batch 的 1D 向量。"""
        if not torch.is_tensor(beta):
            return torch.full((ref.shape[0],), float(beta), device=ref.device, dtype=ref.dtype)
        b = beta.to(device=ref.device, dtype=ref.dtype).reshape(-1)
        if b.numel() == 1:
            return b.expand(ref.shape[0])
        if b.numel() != ref.shape[0]:
            raise RuntimeError(f"beta batch {b.numel()} != ref batch {ref.shape[0]}")
        return b

    def residual_compose(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
        base_u_tp: torch.Tensor,
        base_u_bat: torch.Tensor,
        base_mode: torch.Tensor,
        base_mag: torch.Tensor,
        u_tp_low: torch.Tensor,
        u_tp_high: torch.Tensor,
        u_bat_low: torch.Tensor,
        u_bat_high: torch.Tensor,
        mode_mask: torch.Tensor,
        *,
        beta_tp: float | torch.Tensor = 0.15,
        beta_bat: float | torch.Tensor = 0.30,
        beta_mag: float | torch.Tensor = 0.20,
        mode_margin: float = 1.5,
        mode_override: bool = False,
        logit_clip: float = 8.0,
        soft_mode_for_grad: bool = False,
        explore_noise_std: float = 0.0,
        deterministic: bool = True,
    ) -> dict[str, torch.Tensor]:
        """动作空间残差：a = clip(a_H + β·tanh(clip(Δ(s,g)-Δ(s,0))))。

        - 差分中心化保证 g=0 ⇒ a≡a_H
        - 默认 **模式锁定 Hybrid**（mode_override=False），只改连续量，避免 CAES 模式乱切
        - mag 始终用残差形式 clip(mag_H + β tanh z)，不用 sigmoid(z_res) 当绝对值
        - logit_clip 抑制残差头数值爆炸
        """
        clip_v = float(max(logit_clip, 1.0))
        h = self.encoder(self._pack(obs, goal))
        with torch.no_grad():
            h0 = self.encoder(self._pack(obs, torch.zeros_like(goal)))

        z_tp = (self.thermal_head(h).squeeze(-1) - self.thermal_head(h0).squeeze(-1).detach()).clamp(
            -clip_v, clip_v
        )
        z_bat = (self.battery_head(h).squeeze(-1) - self.battery_head(h0).squeeze(-1).detach()).clamp(
            -clip_v, clip_v
        )
        z_d = (
            self.discharge_mag_head(h).squeeze(-1) - self.discharge_mag_head(h0).squeeze(-1).detach()
        ).clamp(-clip_v, clip_v)
        z_c = (
            self.charge_mag_head(h).squeeze(-1) - self.charge_mag_head(h0).squeeze(-1).detach()
        ).clamp(-clip_v, clip_v)
        d_tp = torch.tanh(z_tp)
        d_bat = torch.tanh(z_bat)
        d_mag = torch.tanh(0.5 * (z_d + z_c))

        beta_tp_t = self._beta_vec(beta_tp, d_tp)
        beta_bat_t = self._beta_vec(beta_bat, d_bat)
        beta_mag_t = self._beta_vec(beta_mag, d_mag)

        base_u_tp = base_u_tp.reshape(-1).detach()
        base_u_bat = base_u_bat.reshape(-1).detach()
        base_mag = base_mag.reshape(-1).detach()
        base_mode_i = base_mode.long().view(-1)
        base_oh = F.one_hot(base_mode_i, num_classes=3).float()

        u_tp = torch.clamp(base_u_tp + beta_tp_t * d_tp, u_tp_low, u_tp_high)
        u_bat = torch.clamp(base_u_bat + beta_bat_t * d_bat, u_bat_low, u_bat_high)
        # mag：纯残差；IDLE 时强制 0
        mag = torch.clamp(base_mag + beta_mag_t * d_mag, 0.0, 1.0)
        mag = torch.where(base_mode_i == int(CaesMode.IDLE), torch.zeros_like(mag), mag)

        logits_g = self.mode_head(h)
        logits_0 = self.mode_head(h0).detach()
        logits = (logits_g - logits_0).clamp(-clip_v, clip_v)
        logits = logits.masked_fill(~mode_mask.bool(), -1e9)

        # 默认：模式 = Hybrid（安全）
        mode = base_mode_i
        mode_oh = base_oh
        # mode 未解锁时：幅度也锁教师，避免“同 mode 拉满 mag”造成 CAES 吞吐爆炸
        if not mode_override:
            mag = base_mag.detach().clamp(0.0, 1.0)
            mag = torch.where(base_mode_i == int(CaesMode.IDLE), torch.zeros_like(mag), mag)
        if mode_override:
            g_strength = goal.abs().mean(dim=-1).clamp(0.0, 1.0)
            if soft_mode_for_grad:
                conf = F.softmax(logits, dim=-1)
                idle = conf[:, int(CaesMode.IDLE)]
                non_idle = conf[:, int(CaesMode.DISCHARGE)] + conf[:, int(CaesMode.CHARGE)]
                gate = torch.sigmoid(4.0 * (non_idle - idle - 0.15 * mode_margin)) * torch.tanh(
                    6.0 * g_strength
                )
                mode_oh = (1.0 - gate.unsqueeze(-1)) * base_oh + gate.unsqueeze(-1) * conf
                mode = torch.argmax(mode_oh, dim=-1)
            else:
                mode_res = torch.argmax(logits, dim=-1)
                gather_res = logits.gather(1, mode_res.view(-1, 1)).squeeze(-1)
                idle_logit = logits[:, int(CaesMode.IDLE)]
                override = (
                    (mode_res != int(CaesMode.IDLE))
                    & ((gather_res - idle_logit) > mode_margin)
                    & (g_strength > 0.05)
                )
                mode = torch.where(override, mode_res, base_mode_i)
                mode_oh = F.one_hot(mode, num_classes=3).float()
            mag = torch.where(mode == int(CaesMode.IDLE), torch.zeros_like(mag), mag)

        if explore_noise_std > 0 and not deterministic:
            u_tp = torch.clamp(u_tp + explore_noise_std * torch.randn_like(u_tp), u_tp_low, u_tp_high)
            u_bat = torch.clamp(u_bat + explore_noise_std * torch.randn_like(u_bat), u_bat_low, u_bat_high)
            mag = torch.clamp(mag + explore_noise_std * torch.randn_like(mag), 0.0, 1.0)
            mag = torch.where(mode == int(CaesMode.IDLE), torch.zeros_like(mag), mag)

        return {
            "u_tp": u_tp,
            "u_battery": u_bat,
            "caes_mode": mode,
            "caes_mode_oh": mode_oh,
            "caes_magnitude": mag,
            "logits_mode": logits,
            "delta_tp": d_tp,
            "delta_bat": d_bat,
            "delta_mag": d_mag,
        }


class LowLevelStochasticActor(nn.Module):
    """Goal-conditioned hybrid stochastic actor for hierarchical SAC.

    Continuous channels: squashed diagonal Gaussians on dynamic bounds.
    CAES mode: masked Categorical. Returns log_prob for max-entropy updates.
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int = 5,
        hidden: int = 256,
        *,
        goal_input_scale: float = 4.0,
    ):
        super().__init__()
        self.goal_dim = int(goal_dim)
        self.goal_input_scale = float(goal_input_scale)
        self.encoder = _mlp(obs_dim + goal_dim, hidden)
        self.tp_mean = nn.Linear(hidden, 1)
        self.bat_mean = nn.Linear(hidden, 1)
        self.mode_head = nn.Linear(hidden, 3)
        self.d_mag_mean = nn.Linear(hidden, 1)
        self.c_mag_mean = nn.Linear(hidden, 1)
        self.tp_log_std = nn.Linear(hidden, 1)
        self.bat_log_std = nn.Linear(hidden, 1)
        self.d_mag_log_std = nn.Linear(hidden, 1)
        self.c_mag_log_std = nn.Linear(hidden, 1)
        # Prefer idle initially; avoid battery thrashing bias
        nn.init.constant_(self.tp_mean.bias, 0.5)
        nn.init.constant_(self.bat_mean.bias, 0.0)
        with torch.no_grad():
            self.mode_head.bias.zero_()
            self.mode_head.bias[int(CaesMode.IDLE)] = 1.5
            self.mode_head.bias[int(CaesMode.DISCHARGE)] = -0.5
            self.mode_head.bias[int(CaesMode.CHARGE)] = -0.5

    def _pack(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        if goal.shape[-1] != self.goal_dim:
            g = torch.zeros(goal.shape[0], self.goal_dim, device=goal.device, dtype=goal.dtype)
            n = min(self.goal_dim, goal.shape[-1])
            g[:, :n] = goal[:, :n]
            goal = g
        if self.goal_input_scale != 1.0:
            goal = goal * self.goal_input_scale
        return torch.cat([obs, goal], dim=-1)

    def _heads(self, obs: torch.Tensor, goal: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(self._pack(obs, goal))
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

    def forward_logits(self, obs: torch.Tensor, goal: torch.Tensor) -> dict[str, torch.Tensor]:
        """F-MLE / BC interface: expose pre-squash means as logits."""
        h = self._heads(obs, goal)
        return {
            "z_tp": h["mu_tp"],
            "z_bat": h["mu_bat"],
            "logits_mode": h["logits_mode"],
            "z_discharge": h["mu_d"],
            "z_charge": h["mu_c"],
        }

    @staticmethod
    def map_bounded(z: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        mapped = low + torch.sigmoid(z) * (high - low)
        return torch.minimum(torch.maximum(mapped, low), high)

    @staticmethod
    def _squash_log_prob(
        dist: Normal, z: torch.Tensor, low: torch.Tensor, high: torch.Tensor
    ) -> torch.Tensor:
        s = torch.sigmoid(z)
        log_det = torch.log(s * (1.0 - s) + 1e-6) + torch.log((high - low).clamp_min(1e-6))
        return dist.log_prob(z) - log_det

    def forward_action(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
        u_tp_low: torch.Tensor,
        u_tp_high: torch.Tensor,
        u_bat_low: torch.Tensor,
        u_bat_high: torch.Tensor,
        mode_mask: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        goal = torch.nan_to_num(goal, nan=0.0, posinf=0.0, neginf=0.0)
        h = self._heads(obs, goal)
        logits = h["logits_mode"].masked_fill(
            ~mode_mask.bool(), torch.finfo(h["logits_mode"].dtype).min / 2
        )
        if not torch.isfinite(logits).all():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            logits = logits.masked_fill(
                ~mode_mask.bool(), torch.finfo(logits.dtype).min / 2
            )
        mode_dist = Categorical(logits=logits)
        mode = torch.argmax(logits, dim=-1) if deterministic else mode_dist.sample()
        mode_oh = F.one_hot(mode, num_classes=3).float()

        def _cont(mu, ls, low, high):
            mu = torch.nan_to_num(mu, nan=0.0)
            ls = torch.nan_to_num(ls, nan=-1.0).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
            dist = Normal(mu, ls.exp())
            z = mu if deterministic else dist.rsample()
            u = self.map_bounded(z, low, high)
            lp = self._squash_log_prob(dist, z, low, high)
            return u, lp, dist.entropy()

        zero = torch.zeros_like(u_tp_low)
        one = torch.ones_like(u_tp_low)
        u_tp, lp_tp, e_tp = _cont(h["mu_tp"], h["ls_tp"], u_tp_low, u_tp_high)
        u_bat, lp_bat, e_bat = _cont(h["mu_bat"], h["ls_bat"], u_bat_low, u_bat_high)
        mag_d, lp_d, e_d = _cont(h["mu_d"], h["ls_d"], zero, one)
        mag_c, lp_c, e_c = _cont(h["mu_c"], h["ls_c"], zero, one)
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

    def act(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
        u_tp_low: torch.Tensor,
        u_tp_high: torch.Tensor,
        u_bat_low: torch.Tensor,
        u_bat_high: torch.Tensor,
        mode_mask: torch.Tensor,
        *,
        deterministic: bool = False,
        explore_noise_std: float = 0.0,
        gumbel_tau: float = 1.0,
        soft_mode_for_grad: bool = False,
    ) -> dict[str, torch.Tensor]:
        """TD3/BC-compatible interface; ignores explore_noise when stochastic."""
        _ = (explore_noise_std, gumbel_tau, soft_mode_for_grad)
        return self.forward_action(
            obs,
            goal,
            u_tp_low,
            u_tp_high,
            u_bat_low,
            u_bat_high,
            mode_mask,
            deterministic=deterministic,
        )


class LowLevelCritic(nn.Module):
    def __init__(self, obs_dim: int, goal_dim: int = 2, hidden: int = 256):
        super().__init__()
        act_dim = 1 + 1 + 3 + 1
        in_dim = obs_dim + goal_dim + act_dim
        self.q1 = self._net(in_dim, hidden)
        self.q2 = self._net(in_dim, hidden)

    @staticmethod
    def _net(in_dim: int, hidden: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _pack(self, obs, goal, u_tp, u_bat, mode_oh, mag):
        return torch.cat(
            [
                obs,
                goal,
                u_tp.unsqueeze(-1) if u_tp.ndim == 1 else u_tp,
                u_bat.unsqueeze(-1) if u_bat.ndim == 1 else u_bat,
                mode_oh,
                mag.unsqueeze(-1) if mag.ndim == 1 else mag,
            ],
            dim=-1,
        )

    def forward(self, obs, goal, u_tp, u_bat, mode_oh, mag):
        x = self._pack(obs, goal, u_tp, u_bat, mode_oh, mag)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_only(self, obs, goal, u_tp, u_bat, mode_oh, mag):
        x = self._pack(obs, goal, u_tp, u_bat, mode_oh, mag)
        return self.q1(x).squeeze(-1)
