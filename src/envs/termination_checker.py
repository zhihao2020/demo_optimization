"""Episode 终止判定：本阶段无经济 terminated，异常一律 truncated。"""

from __future__ import annotations


class TerminationChecker:
    """终止检查器(TerminationChecker)。

    当前阶段不触发经济意义上的 ``terminated``；时限与异常由环境标为 ``truncated``。
    """

    def terminated(self, _outputs: dict[str, float]) -> tuple[bool, str | None]:
        """判断 FMU 输出是否触发 episode 终止。

        Args:
            _outputs: FMU 物理输出字典（本阶段未使用）。

        Returns:
            ``(False, None)``；本阶段永不经济终止。
        """
        return False, None
