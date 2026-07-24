"""物理有效转移分区：仅存储 physically_valid 为真的环境转移样本。"""

from __future__ import annotations

from typing import Any

import numpy as np

from training.hybrid_td3.buffer import Transition


class PhysicalReplayPartition:
    """物理回放分区(PhysicalReplayPartition)：环形缓冲，只接纳物理有效的转移。"""

    TRANSITION_TYPE = "physical"

    def __init__(self, capacity: int = 100_000):
        """初始化物理回放分区。

        Args:
            capacity: 最大存储容量(capacity)，默认 100_000。
        """
        self.capacity = capacity
        self._storage: list[Transition] = []
        self._pos = 0
        self.rejected_count = 0

    def __len__(self) -> int:
        """返回当前已存储的转移数量。

        Returns:
            缓冲区内转移条数。
        """
        return len(self._storage)

    def add(self, transition: Transition) -> bool:
        """尝试写入一条转移；非物理有效或类型不符则拒绝。

        Args:
            transition: 待写入的转移(Transition)。

        Returns:
            写入成功为 True，被拒绝为 False。
        """
        if not transition.physically_valid:
            self.rejected_count += 1
            return False
        if getattr(transition, "transition_type", self.TRANSITION_TYPE) not in (self.TRANSITION_TYPE, None):
            # 允许未标注时视为物理
            if getattr(transition, "transition_type", None) == "givesafe_rejection":
                self.rejected_count += 1
                return False
        transition.transition_type = self.TRANSITION_TYPE
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
        else:
            self._storage[self._pos] = transition
        self._pos = (self._pos + 1) % self.capacity
        return True

    def sample(self, batch_size: int) -> list[Transition]:
        """有放回随机采样一批转移。

        Args:
            batch_size: 采样条数(batch_size)。

        Returns:
            转移(Transition) 列表；缓冲为空时返回空列表。
        """
        n = len(self._storage)
        if n == 0:
            return []
        idx = np.random.randint(0, n, size=batch_size)
        return [self._storage[i] for i in idx]
