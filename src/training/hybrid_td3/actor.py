"""有界参数化混合 Actor(HybridActor)。连续动作经 sigmoid 映射到动态可行区间；模式经 mask 与 Gumbel-Softmax。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from actions import CaesMode, ModeMask


def _mlp(in_dim: int, hidden: int = 256) -> nn.Sequential:
    """构建两层 ReLU 全连接编码器。

    Args:
        in_dim: 输入维度。
        hidden: 隐层宽度。

    Returns:
        Sequential MLP 模块。
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
    )


class HybridActor(nn.Module):
    """混合 Actor：火电/电池 sigmoid 有界连续 + CAES 三模式离散头。"""

    def __init__(self, obs_dim: int, hidden: int = 256):
        """初始化编码器与各动作头。

        Args:
            obs_dim: 观测维度。
            hidden: 隐层宽度。
        """
        super().__init__()
        self.encoder = _mlp(obs_dim, hidden)
        self.thermal_head = nn.Linear(hidden, 1)
        self.battery_head = nn.Linear(hidden, 1)
        self.mode_head = nn.Linear(hidden, 3)
        self.discharge_mag_head = nn.Linear(hidden, 1)
        self.charge_mag_head = nn.Linear(hidden, 1)

    def forward_logits(self, obs: torch.Tensor):
        """计算各动作潜变量与模式 logits。

        Args:
            obs: 形状 (B, obs_dim) 的观测。

        Returns:
            含 z_tp、z_bat、logits_mode、z_discharge、z_charge 的字典。
        """
        h = self.encoder(obs)
        return {
            "z_tp": self.thermal_head(h).squeeze(-1),
            "z_bat": self.battery_head(h).squeeze(-1),
            "logits_mode": self.mode_head(h),
            "z_discharge": self.discharge_mag_head(h).squeeze(-1),
            "z_charge": self.charge_mag_head(h).squeeze(-1),
        }

    @staticmethod
    def map_bounded(z: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        """sigmoid 映射到 [low, high] 并显式 clamp，避免 float32 越界。

        Args:
            z: 无界潜变量。
            low: 下界。
            high: 上界。

        Returns:
            有界动作张量。
        """
        # 映射后显式 clamp，避免 float32 越出动态边界导致校验失败
        mapped = low + torch.sigmoid(z) * (high - low)
        return torch.minimum(torch.maximum(mapped, low), high)

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
        gumbel_tau: float = 1.0,
        explore_noise_std: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """采样混合动作；连续量在动态边界内，模式受 mask 约束。

        Args:
            obs: 观测张量。
            u_tp_low: 火电下界。
            u_tp_high: 火电上界。
            u_bat_low: 电池下界。
            u_bat_high: 电池上界。
            mode_mask: 形状 (B,3) bool，True 表示 CAES 模式允许。
            deterministic: True 时 argmax 模式，否则 Gumbel-Softmax 硬采样。
            gumbel_tau: Gumbel-Softmax 温度。
            explore_noise_std: 非确定性时在连续动作上加高斯噪声并 clamp。

        Returns:
            含 u_tp、u_battery、caes_mode、caes_mode_oh、caes_magnitude 等的字典。
        """
        out = self.forward_logits(obs)
        logits = out["logits_mode"].masked_fill(~mode_mask.bool(), -1e9)
        if deterministic:
            mode = torch.argmax(logits, dim=-1)
            mode_oh = F.one_hot(mode, num_classes=3).float()
        else:
            mode_oh = F.gumbel_softmax(logits, tau=gumbel_tau, hard=True, dim=-1)
            mode = torch.argmax(mode_oh, dim=-1)

        u_tp = self.map_bounded(out["z_tp"], u_tp_low, u_tp_high)
        u_bat = self.map_bounded(out["z_bat"], u_bat_low, u_bat_high)
        mag_d = torch.sigmoid(out["z_discharge"])
        mag_c = torch.sigmoid(out["z_charge"])
        # 按模式选择幅值；IDLE -> 0
        mag = mode_oh[:, 0] * mag_d + mode_oh[:, 2] * mag_c

        if explore_noise_std > 0 and not deterministic:
            # 噪声保持在动态范围内：对连续变量在区间内扰动后重新夹紧到 [low,high]
            u_tp = torch.clamp(u_tp + explore_noise_std * torch.randn_like(u_tp), u_tp_low, u_tp_high)
            u_bat = torch.clamp(u_bat + explore_noise_std * torch.randn_like(u_bat), u_bat_low, u_bat_high)
            mag = torch.clamp(mag + explore_noise_std * torch.randn_like(mag), 0.0, 1.0)
            # 噪声后 IDLE 幅值仍记 0
            mag = torch.where(mode == int(CaesMode.IDLE), torch.zeros_like(mag), mag)

        return {
            "u_tp": u_tp,
            "u_battery": u_bat,
            "caes_mode": mode,
            "caes_mode_oh": mode_oh,
            "caes_magnitude": mag,
            "logits_mode": logits,
        }

    def act_numpy(self, obs, feasible, deterministic: bool = True, device="cpu", explore_noise_std: float = 0.0):
        """NumPy 单步接口：返回环境可 step 的混合动作字典。

        Args:
            obs: 一维观测。
            feasible: 可行动作规格。
            deterministic: 是否确定性。
            device: PyTorch 设备。
            explore_noise_std: 探索噪声标准差。

        Returns:
            含 u_tp、u_battery、caes_mode、caes_magnitude 的字典。
        """
        import numpy as np
        self.eval()
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=device).view(1, -1)
            mask = torch.as_tensor(feasible.mode_mask.as_bool_array(), dtype=torch.bool, device=device).view(1, 3)
            out = self.act(
                o,
                torch.tensor([feasible.u_tp_low], device=device),
                torch.tensor([feasible.u_tp_high], device=device),
                torch.tensor([feasible.u_battery_low], device=device),
                torch.tensor([feasible.u_battery_high], device=device),
                mask,
                deterministic=deterministic,
                explore_noise_std=0.0 if deterministic else explore_noise_std,
            )
        return {
            "u_tp": np.asarray([float(out["u_tp"][0].cpu())], dtype=np.float32),
            "u_battery": np.asarray([float(out["u_battery"][0].cpu())], dtype=np.float32),
            "caes_mode": int(out["caes_mode"][0].cpu()),
            "caes_magnitude": np.asarray([float(out["caes_magnitude"][0].cpu())], dtype=np.float32),
        }
