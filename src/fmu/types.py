"""FMU 会话层用的最小数据类型。

仅描述调度计划与仿真轨迹，不含市场结算 / reward。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DispatchPlan:
    """按时序排列的调度指令。

    Attributes:
        time: 通信点时间戳，单位 s；长度通常为 horizon+1。
        u_tp / u_battery / u_caes: 各小时动作，长度通常为 horizon。
    """

    time: np.ndarray
    u_tp: np.ndarray
    u_battery: np.ndarray
    u_caes: np.ndarray


@dataclass
class SimulationResult:
    """一次 rollout / 仿真的输出轨迹。

    Attributes:
        time: 各采样时刻，单位 s。
        variables: 物理输出名 -> 与 time 对齐的一维数组。
        metadata: 执行信息（是否失败、错误信息、完成小时数等）。
    """

    time: np.ndarray
    variables: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, name: str) -> np.ndarray | None:
        """按 FMU 顶层输出名取值；不存在则返回 None。"""
        return self.variables.get(name)
