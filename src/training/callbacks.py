"""Stable-Baselines3 训练回调：逐步审计 FMU 状态与奖励分项。"""

from __future__ import annotations

import csv
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


class EpisodeAuditCallback(BaseCallback):
    """回合审计回调(EpisodeAuditCallback)：逐步 CSV 记录 FMU 状态、动作与奖励分项，不改写动作或奖励。"""

    def __init__(self, path: str | Path):
        """初始化审计回调。

        Args:
            path: 审计 CSV 输出路径。
        """
        super().__init__()
        self.path = Path(path)
        self.handle = None
        self.writer = None
        self.fmu_failure_count = 0

    def _on_training_start(self) -> None:
        """训练开始时创建 CSV 并写入表头。

        Returns:
            无。
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=["timesteps", "time", "fmu_status", "reward", "total_cost", "solver_failure_cost"])
        self.writer.writeheader()

    def _on_step(self) -> bool:
        """每步写入 info 与 reward 审计行。

        Returns:
            始终为 True，继续训练。
        """
        for info, reward in zip(self.locals.get("infos", []), self.locals.get("rewards", [])):
            terms = info.get("reward_terms", {})
            status = info.get("fmu_status", "unknown")
            self.fmu_failure_count += int(status == "failure")
            self.writer.writerow({"timesteps": self.num_timesteps, "time": info.get("time"), "fmu_status": status,
                                  "reward": float(reward), "total_cost": terms.get("total_cost"), "solver_failure_cost": terms.get("solver_failure_cost")})
        if self.handle:
            self.handle.flush()
        return True

    def _on_training_end(self) -> None:
        """训练结束时关闭 CSV 文件句柄。

        Returns:
            无。
        """
        if self.handle:
            self.handle.close()
