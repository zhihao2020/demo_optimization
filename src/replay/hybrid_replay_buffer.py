"""Hybrid GiveSafe Replay：Physical 与 GiveSafe 分区按配置比例混合采样。"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch.nn.functional as F

from actions.caes_u import (
    CHARGE_HI,
    CHARGE_LO,
    DISCHARGE_HI,
    DISCHARGE_LO,
    mag_from_u,
    mode_from_u,
)
from training.hybrid_td3.buffer import Transition

from .givesafe_partition import GiveSafeReplayPartition
from .physical_partition import PhysicalReplayPartition


def _pack_batch(batch: list[Transition]) -> dict[str, np.ndarray]:
    """将转移列表打包为训练用 NumPy 字典批次。

    Args:
        batch: 转移(Transition) 列表。

    Returns:
        含 obs、next_obs、混合动作分量、奖励、终止标志、模式掩码、
        动态动作界及 transition_type 等键的字典。
    """
    obs = np.stack([t.observation for t in batch]).astype(np.float32)
    next_obs = np.stack([t.next_observation for t in batch]).astype(np.float32)
    u_tp = np.asarray([t.hybrid_action["u_tp"] for t in batch], dtype=np.float32)
    u_bat = np.asarray([t.hybrid_action["u_battery"] for t in batch], dtype=np.float32)
    u_caes = np.asarray([t.hybrid_action["u_caes"] for t in batch], dtype=np.float32)
    caes_mode = np.asarray(
        [
            int(t.hybrid_action.get("caes_mode", int(mode_from_u(float(t.hybrid_action["u_caes"])))))
            for t in batch
        ],
        dtype=np.int64,
    )
    caes_magnitude = np.asarray(
        [
            float(t.hybrid_action.get("caes_magnitude", mag_from_u(float(t.hybrid_action["u_caes"]))))
            for t in batch
        ],
        dtype=np.float32,
    )
    reward = np.asarray([t.reward for t in batch], dtype=np.float32)
    done = np.asarray([float(t.terminated) for t in batch], dtype=np.float32)
    mask = np.stack([t.valid_mode_mask for t in batch]).astype(np.bool_)
    next_mask = np.stack(
        [(t.next_valid_mode_mask if t.next_valid_mode_mask is not None else t.valid_mode_mask) for t in batch]
    ).astype(np.bool_)
    ttypes = np.asarray([1 if getattr(t, "transition_type", "") == "givesafe_rejection" else 0 for t in batch], dtype=np.int64)

    _FALLBACK = {
        "u_caes_discharge_low": DISCHARGE_LO,
        "u_caes_discharge_high": DISCHARGE_HI,
        "u_caes_charge_low": CHARGE_LO,
        "u_caes_charge_high": CHARGE_HI,
    }

    def bounds_arr(key: str, next_b: bool = False):
        """从批次中提取指定动态动作界键的一维数组。"""
        vals = []
        for t in batch:
            src = t.next_dynamic_action_bounds if next_b and t.next_dynamic_action_bounds else t.dynamic_action_bounds
            vals.append(float(src.get(key, _FALLBACK.get(key, 0.0))))
        return np.asarray(vals, dtype=np.float32)

    return {
        "obs": obs,
        "next_obs": next_obs,
        "u_tp": u_tp,
        "u_battery": u_bat,
        "u_caes": u_caes,
        "caes_mode": caes_mode,
        "caes_magnitude": caes_magnitude,
        "reward": reward,
        "done": done,
        "mode_mask": mask,
        "next_mode_mask": next_mask,
        "u_tp_low": bounds_arr("u_tp_low"),
        "u_tp_high": bounds_arr("u_tp_high"),
        "u_bat_low": bounds_arr("u_battery_low"),
        "u_bat_high": bounds_arr("u_battery_high"),
        "dis_lo": bounds_arr("u_caes_discharge_low"),
        "dis_hi": bounds_arr("u_caes_discharge_high"),
        "chg_lo": bounds_arr("u_caes_charge_low"),
        "chg_hi": bounds_arr("u_caes_charge_high"),
        "next_u_tp_low": bounds_arr("u_tp_low", True),
        "next_u_tp_high": bounds_arr("u_tp_high", True),
        "next_u_bat_low": bounds_arr("u_battery_low", True),
        "next_u_bat_high": bounds_arr("u_battery_high", True),
        "next_dis_lo": bounds_arr("u_caes_discharge_low", True),
        "next_dis_hi": bounds_arr("u_caes_discharge_high", True),
        "next_chg_lo": bounds_arr("u_caes_charge_low", True),
        "next_chg_hi": bounds_arr("u_caes_charge_high", True),
        "transition_type": ttypes,
    }


class HybridGiveSafeReplayBuffer:
    """混合 GiveSafe 回放缓冲(HybridGiveSafeReplayBuffer)：物理样本与 GiveSafe 自环样本按配置比例混合采样。"""

    def __init__(
        self,
        capacity: int = 100_000,
        *,
        physical_fraction: float = 1.0,
        givesafe_fraction: float = 0.0,
    ):
        """初始化混合回放缓冲及两个子分区。

        Args:
            capacity: 各子分区最大容量(capacity)，默认 100_000。
            physical_fraction: 物理样本采样占比(physical_fraction)，默认 0.7。
            givesafe_fraction: GiveSafe 样本采样占比(givesafe_fraction)，默认 0.3。

        Raises:
            AssertionError: physical_fraction 与 givesafe_fraction 之和不为 1。
        """
        assert abs(physical_fraction + givesafe_fraction - 1.0) < 1e-6
        self.physical = PhysicalReplayPartition(capacity)
        self.givesafe = GiveSafeReplayPartition(capacity)
        self.physical_fraction = float(physical_fraction)
        self.givesafe_fraction = float(givesafe_fraction)
        self.rejected_count = 0
        self.invalid_attempt_count = 0

    def __len__(self) -> int:
        """返回两个分区存储的转移总数。

        Returns:
            物理分区与 GiveSafe 分区条数之和。
        """
        return len(self.physical) + len(self.givesafe)

    @property
    def physical_size(self) -> int:
        """物理分区当前样本数。

        Returns:
            物理分区(PhysicalReplayPartition) 内转移条数。
        """
        return len(self.physical)

    @property
    def givesafe_size(self) -> int:
        """GiveSafe 分区当前样本数。

        Returns:
            GiveSafe 分区(GiveSafeReplayPartition) 内转移条数。
        """
        return len(self.givesafe)

    def add_physical(self, transition: Transition) -> bool:
        """向物理分区写入一条转移。

        Args:
            transition: 待写入的转移(Transition)。

        Returns:
            写入成功为 True；被拒绝时递增 rejected_count 与 invalid_attempt_count 并返回 False。
        """
        ok = self.physical.add(transition)
        if not ok:
            self.rejected_count += 1
            self.invalid_attempt_count += 1
        return ok

    def add_givesafe_rejection(self, transition: Transition) -> bool:
        """向 GiveSafe 分区写入一条拒绝自环转移。

        Args:
            transition: 待写入的转移(Transition)。

        Returns:
            写入结果，由 GiveSafe 分区 add 方法决定。
        """
        return self.givesafe.add(transition)

    # 兼容旧接口名
    def add(self, transition: Transition) -> bool:
        """按 transition_type 自动路由到物理或 GiveSafe 分区。

        Args:
            transition: 待写入的转移(Transition)。

        Returns:
            对应分区 add 方法的返回值。
        """
        ttype = getattr(transition, "transition_type", None)
        if ttype == "givesafe_rejection":
            return self.add_givesafe_rejection(transition)
        return self.add_physical(transition)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """按配置比例从两分区混合采样并打包为训练批次。

        若一侧为空，则用另一侧补齐；仍不足则循环采样直至达到 batch_size 或无法再补。

        Args:
            batch_size: 目标批次大小(batch_size)。

        Returns:
            由 _pack_batch 生成的 NumPy 字典批次。

        Raises:
            RuntimeError: 两个分区均为空，无法采样。
        """
        n_phys = int(round(batch_size * self.physical_fraction))
        n_gs = batch_size - n_phys
        batch: list[Transition] = []
        if len(self.physical) > 0 and n_phys > 0:
            batch.extend(self.physical.sample(n_phys))
        elif n_phys > 0 and len(self.givesafe) > 0:
            batch.extend(self.givesafe.sample(n_phys))
        if len(self.givesafe) > 0 and n_gs > 0:
            batch.extend(self.givesafe.sample(n_gs))
        elif n_gs > 0 and len(self.physical) > 0:
            batch.extend(self.physical.sample(n_gs))
        if not batch:
            raise RuntimeError("replay buffer 为空")
        # 若因一侧为空导致数量不足，用已有侧补齐
        while len(batch) < batch_size:
            src = self.physical if len(self.physical) >= len(self.givesafe) else self.givesafe
            if len(src) == 0:
                break
            batch.extend(src.sample(1))
        return _pack_batch(batch[:batch_size])
