"""Hybrid GiveSafe Replay：Physical + GiveSafe 分区混合采样。"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch.nn.functional as F

from training.hybrid_td3.buffer import Transition

from .givesafe_partition import GiveSafeReplayPartition
from .physical_partition import PhysicalReplayPartition


def _pack_batch(batch: list[Transition]) -> dict[str, np.ndarray]:
    obs = np.stack([t.observation for t in batch]).astype(np.float32)
    next_obs = np.stack([t.next_observation for t in batch]).astype(np.float32)
    u_tp = np.asarray([t.hybrid_action["u_tp"] for t in batch], dtype=np.float32)
    u_bat = np.asarray([t.hybrid_action["u_battery"] for t in batch], dtype=np.float32)
    mode = np.asarray([t.hybrid_action["caes_mode"] for t in batch], dtype=np.int64)
    mag = np.asarray([t.hybrid_action["caes_magnitude"] for t in batch], dtype=np.float32)
    reward = np.asarray([t.reward for t in batch], dtype=np.float32)
    done = np.asarray([float(t.terminated) for t in batch], dtype=np.float32)
    mask = np.stack([t.valid_mode_mask for t in batch]).astype(np.bool_)
    next_mask = np.stack(
        [(t.next_valid_mode_mask if t.next_valid_mode_mask is not None else t.valid_mode_mask) for t in batch]
    ).astype(np.bool_)
    ttypes = np.asarray([1 if getattr(t, "transition_type", "") == "givesafe_rejection" else 0 for t in batch], dtype=np.int64)

    def bounds_arr(key: str, next_b: bool = False):
        vals = []
        for t in batch:
            src = t.next_dynamic_action_bounds if next_b and t.next_dynamic_action_bounds else t.dynamic_action_bounds
            vals.append(float(src.get(key, 0.0)))
        return np.asarray(vals, dtype=np.float32)

    return {
        "obs": obs,
        "next_obs": next_obs,
        "u_tp": u_tp,
        "u_battery": u_bat,
        "caes_mode": mode,
        "caes_magnitude": mag,
        "reward": reward,
        "done": done,
        "mode_mask": mask,
        "next_mode_mask": next_mask,
        "u_tp_low": bounds_arr("u_tp_low"),
        "u_tp_high": bounds_arr("u_tp_high"),
        "u_bat_low": bounds_arr("u_battery_low"),
        "u_bat_high": bounds_arr("u_battery_high"),
        "next_u_tp_low": bounds_arr("u_tp_low", True),
        "next_u_tp_high": bounds_arr("u_tp_high", True),
        "next_u_bat_low": bounds_arr("u_battery_low", True),
        "next_u_bat_high": bounds_arr("u_battery_high", True),
        "transition_type": ttypes,
    }


class HybridGiveSafeReplayBuffer:
    """物理样本 + GiveSafe 自环样本；按配置比例混合采样。"""

    def __init__(
        self,
        capacity: int = 100_000,
        *,
        physical_fraction: float = 0.7,
        givesafe_fraction: float = 0.3,
    ):
        assert abs(physical_fraction + givesafe_fraction - 1.0) < 1e-6
        self.physical = PhysicalReplayPartition(capacity)
        self.givesafe = GiveSafeReplayPartition(capacity)
        self.physical_fraction = float(physical_fraction)
        self.givesafe_fraction = float(givesafe_fraction)
        self.rejected_count = 0
        self.invalid_attempt_count = 0

    def __len__(self) -> int:
        return len(self.physical) + len(self.givesafe)

    @property
    def physical_size(self) -> int:
        return len(self.physical)

    @property
    def givesafe_size(self) -> int:
        return len(self.givesafe)

    def add_physical(self, transition: Transition) -> bool:
        ok = self.physical.add(transition)
        if not ok:
            self.rejected_count += 1
            self.invalid_attempt_count += 1
        return ok

    def add_givesafe_rejection(self, transition: Transition) -> bool:
        return self.givesafe.add(transition)

    # 兼容旧接口名
    def add(self, transition: Transition) -> bool:
        ttype = getattr(transition, "transition_type", None)
        if ttype == "givesafe_rejection":
            return self.add_givesafe_rejection(transition)
        return self.add_physical(transition)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
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
