"""GiveSafe 拒绝自环样本分区。"""

from __future__ import annotations

from typing import Any

import numpy as np

from training.hybrid_td3.buffer import Transition


class GiveSafeReplayPartition:
    TRANSITION_TYPE = "givesafe_rejection"

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self._storage: list[Transition] = []
        self._pos = 0

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, transition: Transition) -> bool:
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
        n = len(self._storage)
        if n == 0:
            return []
        idx = np.random.randint(0, n, size=batch_size)
        return [self._storage[i] for i in idx]
