"""物理有效转移分区。"""

from __future__ import annotations

from typing import Any

import numpy as np

from training.hybrid_td3.buffer import Transition


class PhysicalReplayPartition:
    TRANSITION_TYPE = "physical"

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self._storage: list[Transition] = []
        self._pos = 0
        self.rejected_count = 0

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, transition: Transition) -> bool:
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
        n = len(self._storage)
        if n == 0:
            return []
        idx = np.random.randint(0, n, size=batch_size)
        return [self._storage[i] for i in idx]
