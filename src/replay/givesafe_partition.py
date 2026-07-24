"""GiveSafe 拒绝自环样本分区：存储安全层拒绝后的自环转移用于离线学习。"""

from __future__ import annotations

from typing import Any

import numpy as np

from training.hybrid_td3.buffer import Transition


class GiveSafeReplayPartition:
    """GiveSafe 回放分区(GiveSafeReplayPartition)：存储 givesafe_rejection 类型的自环样本。"""

    TRANSITION_TYPE = "givesafe_rejection"

    def __init__(self, capacity: int = 100_000):
        """初始化 GiveSafe 回放分区。

        Args:
            capacity: 最大存储容量(capacity)，默认 100_000。
        """
        self.capacity = capacity
        self._storage: list[Transition] = []
        self._pos = 0

    def __len__(self) -> int:
        """返回当前已存储的转移数量。

        Returns:
            缓冲区内转移条数。
        """
        return len(self._storage)

    def add(self, transition: Transition) -> bool:
        """写入一条 GiveSafe 拒绝自环转移。

        调用方应保证 next_observation 与 observation 相同（自环约束）。

        Args:
            transition: 待写入的转移(Transition)。

        Returns:
            始终为 True（本分区不拒绝写入）。
        """
        transition.transition_type = self.TRANSITION_TYPE
        transition.physically_valid = False  # 语义上非物理前进，但允许训练
        # 自环约束：next == obs 由调用方保证
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
        else:
            self._storage[self._pos] = transition
        self._pos = (self._pos + 1) % self.capacity
        return True

    def sample(self, batch_size: int) -> list[Transition]:
        """有放回随机采样一批 GiveSafe 转移。

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
