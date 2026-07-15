from __future__ import annotations

import csv
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


class EpisodeAuditCallback(BaseCallback):
    """逐步 CSV 审计 FMU 状态、动作与 reward 分项；不改写动作或奖励。"""
    def __init__(self, path: str | Path):
        super().__init__()
        self.path = Path(path)
        self.handle = None
        self.writer = None
        self.fmu_failure_count = 0

    def _on_training_start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=["timesteps", "time", "fmu_status", "reward", "total_cost", "solver_failure_cost"])
        self.writer.writeheader()

    def _on_step(self) -> bool:
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
        if self.handle:
            self.handle.close()
