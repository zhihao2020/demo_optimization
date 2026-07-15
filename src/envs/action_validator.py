"""动作合法性检查；绝不剪切、饱和或投影。"""

from __future__ import annotations

import numpy as np


class InvalidActionError(ValueError):
    pass


class ActionOutOfBoundsError(InvalidActionError):
    pass


class NonFiniteActionError(InvalidActionError):
    pass


class ActionConstraintError(InvalidActionError):
    pass


class ActionValidator:
    def __init__(self, names: tuple[str, ...], units: tuple[str, ...], low: np.ndarray, high: np.ndarray,
                 valid_intervals: dict[str, tuple[tuple[float, float], ...]] | None = None) -> None:
        self.names, self.units = names, units
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.valid_intervals = valid_intervals or {}

    def validate(self, action: np.ndarray) -> np.ndarray:
        if not isinstance(action, np.ndarray):
            raise InvalidActionError("action 必须是固定形状的 numpy.ndarray")
        if action.shape != self.low.shape:
            raise InvalidActionError(f"action shape={action.shape}; 期望 {self.low.shape}")
        if not np.issubdtype(action.dtype, np.number):
            raise InvalidActionError(f"action dtype={action.dtype} 不是数值型")
        if not np.all(np.isfinite(action)):
            raise NonFiniteActionError(f"action 含 NaN/Inf: {action!r}")
        values = action.astype(np.float64, copy=False)
        invalid = np.flatnonzero((values < self.low) | (values > self.high))
        if invalid.size:
            detail = "; ".join(
                f"{self.names[i]}={values[i]} {self.units[i]}，范围 [{self.low[i]}, {self.high[i]}]"
                for i in invalid
            )
            raise ActionOutOfBoundsError(f"动作越界（维度 {invalid.tolist()}）：{detail}")
        for index, name in enumerate(self.names):
            intervals = self.valid_intervals.get(name)
            if intervals and not any(lo <= values[index] <= hi for lo, hi in intervals):
                raise ActionConstraintError(
                    f"{name}={values[index]} {self.units[index]} 不在允许集合 {intervals}；未执行任何投影"
                )
        return action
