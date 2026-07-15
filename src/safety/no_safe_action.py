"""无安全动作时的失败类型。"""

from __future__ import annotations

from envs.failures import ConstraintFailure


class NoSafeActionFoundError(ConstraintFailure):
    failure_type = "NoSafeActionFound"

    def __init__(self, reason: str, *, attempts: int = 0, rejected: list | None = None):
        self.attempts = attempts
        self.rejected = rejected or []
        super().__init__(reason)
