"""未找到安全动作时的失败类型定义。"""

from __future__ import annotations

from envs.failures import ConstraintFailure


class NoSafeActionFoundError(ConstraintFailure):
    """在最大重采样次数内仍无法找到安全动作时抛出的约束失败异常。"""

    failure_type = "NoSafeActionFound"

    def __init__(
        self,
        reason: str,
        *,
        attempts: int = 0,
        rejected: list | None = None,
        first_check=None,
        reasons: list | None = None,
    ):
        """构造未找到安全动作异常。

        Args:
            reason: 人类可读失败原因。
            attempts: 已尝试的采样次数。
            rejected: 被拒绝的候选动作列表。

        Raises:
            无：本方法为异常构造函数。
        """
        self.attempts = attempts
        self.rejected = rejected or []
        self.first_check = first_check
        self.reasons = reasons or []
        super().__init__(reason)
