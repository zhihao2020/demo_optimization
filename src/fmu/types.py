"""FMU 会话层用的最小数据类型。

仅描述调度计划与仿真轨迹，不含市场结算 / reward。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DispatchPlan:
    """调度计划(DispatchPlan)：按时序排列的 FMU 输入指令。

    Attributes:
        time: 通信点时间戳（秒）；长度通常为 horizon+1。
        u_tp: 火电负荷率序列，长度通常为 horizon。
        u_battery: 电池功率指令序列（归一化）。
        u_caes: CAES 功率指令序列（归一化）。
    """

    time: np.ndarray
    u_tp: np.ndarray
    u_battery: np.ndarray
    u_caes: np.ndarray


@dataclass
class SimulationResult:
    """仿真结果(SimulationResult)：一次 rollout 的输出轨迹。

    Attributes:
        time: 各采样时刻（秒）。
        variables: 物理输出名 -> 与 time 对齐的一维数组。
        metadata: 执行信息（是否失败、错误信息、完成小时数等）。
    """

    time: np.ndarray
    variables: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, name: str) -> np.ndarray | None:
        """按 FMU 顶层输出名取值。

        Args:
            name: FMU 变量名。

        Returns:
            与 ``time`` 对齐的一维数组；不存在则 ``None``。
        """
        return self.variables.get(name)
