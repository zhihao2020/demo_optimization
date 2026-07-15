from __future__ import annotations


class TerminationChecker:
    """本阶段没有经济 terminated；异常/时限一律由环境标为 truncated。"""
    def terminated(self, _outputs: dict[str, float]) -> tuple[bool, str | None]:
        return False, None
