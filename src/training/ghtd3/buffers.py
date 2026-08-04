"""高/低层 replay。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LowTransition:
    observation: np.ndarray
    goal: np.ndarray
    hybrid_action: dict[str, Any]
    reward_int: float
    next_observation: np.ndarray
    next_goal: np.ndarray
    terminated: bool
    valid_mode_mask: np.ndarray
    next_valid_mode_mask: np.ndarray
    dynamic_action_bounds: dict[str, float]
    next_dynamic_action_bounds: dict[str, float]
    reward_terms: dict[str, float] = field(default_factory=dict)


@dataclass
class HighTransition:
    observation: np.ndarray
    goal: np.ndarray
    reward_ext_sum: float
    next_observation: np.ndarray
    terminated: bool
    # relabel 用：周期内逐步 SoC 与动作
    soc_seq: list[np.ndarray] = field(default_factory=list)
    action_seq: list[dict[str, Any]] = field(default_factory=list)


class RingBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._storage: list[Any] = []
        self._pos = 0

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, item: Any) -> None:
        if len(self._storage) < self.capacity:
            self._storage.append(item)
        else:
            self._storage[self._pos] = item
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Any]:
        n = len(self._storage)
        if n == 0:
            raise RuntimeError("buffer 为空")
        idx = np.random.randint(0, n, size=min(batch_size, n))
        return [self._storage[i] for i in idx]


class LowReplayBuffer(RingBuffer):
    def sample_batch(self, batch_size: int) -> dict[str, np.ndarray]:
        batch = self.sample(batch_size)
        obs = np.stack([t.observation for t in batch]).astype(np.float32)
        goal = np.stack([t.goal for t in batch]).astype(np.float32)
        next_obs = np.stack([t.next_observation for t in batch]).astype(np.float32)
        next_goal = np.stack([t.next_goal for t in batch]).astype(np.float32)
        u_tp = np.asarray([t.hybrid_action["u_tp"] for t in batch], dtype=np.float32)
        u_bat = np.asarray([t.hybrid_action["u_battery"] for t in batch], dtype=np.float32)
        mode = np.asarray([t.hybrid_action["caes_mode"] for t in batch], dtype=np.int64)
        mag = np.asarray([t.hybrid_action["caes_magnitude"] for t in batch], dtype=np.float32)
        reward = np.asarray([t.reward_int for t in batch], dtype=np.float32)
        done = np.asarray([float(t.terminated) for t in batch], dtype=np.float32)
        mask = np.stack([t.valid_mode_mask for t in batch]).astype(np.bool_)
        next_mask = np.stack([t.next_valid_mode_mask for t in batch]).astype(np.bool_)

        def bnds(key: str, nxt: bool = False):
            vals = []
            for t in batch:
                src = t.next_dynamic_action_bounds if nxt else t.dynamic_action_bounds
                vals.append(float(src[key]))
            return np.asarray(vals, dtype=np.float32)

        return {
            "obs": obs,
            "goal": goal,
            "next_obs": next_obs,
            "next_goal": next_goal,
            "u_tp": u_tp,
            "u_battery": u_bat,
            "caes_mode": mode,
            "caes_magnitude": mag,
            "reward": reward,
            "done": done,
            "mode_mask": mask,
            "next_mode_mask": next_mask,
            "u_tp_low": bnds("u_tp_low"),
            "u_tp_high": bnds("u_tp_high"),
            "u_bat_low": bnds("u_battery_low"),
            "u_bat_high": bnds("u_battery_high"),
            "next_u_tp_low": bnds("u_tp_low", True),
            "next_u_tp_high": bnds("u_tp_high", True),
            "next_u_bat_low": bnds("u_battery_low", True),
            "next_u_bat_high": bnds("u_battery_high", True),
        }


class HighReplayBuffer(RingBuffer):
    def sample_batch(self, batch_size: int) -> dict[str, np.ndarray]:
        batch = self.sample(batch_size)
        return {
            "obs": np.stack([t.observation for t in batch]).astype(np.float32),
            "goal": np.stack([t.goal for t in batch]).astype(np.float32),
            "reward": np.asarray([t.reward_ext_sum for t in batch], dtype=np.float32),
            "next_obs": np.stack([t.next_observation for t in batch]).astype(np.float32),
            "done": np.asarray([float(t.terminated) for t in batch], dtype=np.float32),
            "transitions": batch,  # for relabel
        }
