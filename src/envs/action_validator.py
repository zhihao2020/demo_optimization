"""连续动作合法性检查；绝不剪切、饱和或投影。"""

from __future__ import annotations

import numpy as np


class InvalidActionError(ValueError):
    """无效动作基类(InvalidActionError)。"""


class ActionOutOfBoundsError(InvalidActionError):
    """动作越界(ActionOutOfBoundsError)。"""


class NonFiniteActionError(InvalidActionError):
    """动作含 NaN/Inf(NonFiniteActionError)。"""


class ActionConstraintError(InvalidActionError):
    """动作不满足离散/区间约束(ActionConstraintError)。"""


class ActionValidator:
    """连续动作校验器(ActionValidator)。

    按名称、单位与上下界校验 numpy 动作向量；越界直接抛错，不做投影。
    """

    def __init__(
        self,
        names: tuple[str, ...],
        units: tuple[str, ...],
        low: np.ndarray,
        high: np.ndarray,
        valid_intervals: dict[str, tuple[tuple[float, float], ...]] | None = None,
    ) -> None:
        """初始化校验器。

        Args:
            names: 动作维度名称元组。
            units: 各维度物理单位元组。
            low: 各维度下界数组。
            high: 各维度上界数组。
            valid_intervals: 可选，指定维度的允许区间并集（如 CAES 分段集合）。
        """
        self.names, self.units = names, units
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.valid_intervals = valid_intervals or {}

    def validate(self, action: np.ndarray) -> np.ndarray:
        """校验动作形状、数值有限性与边界。

        Args:
            action: 待校验的动作向量。

        Returns:
            校验通过的 ``float64`` 视图（不拷贝时为零拷贝）。

        Raises:
            InvalidActionError: 类型、形状或 dtype 非法。
            NonFiniteActionError: 含 NaN/Inf。
            ActionOutOfBoundsError: 超出 ``[low, high]``。
            ActionConstraintError: 不在 ``valid_intervals`` 允许集合内。
        """
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
