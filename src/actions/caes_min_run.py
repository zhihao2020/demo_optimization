"""CAES 充/放最短连续运行约束控制器（由 Python 执行，不改写动作）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .mode_mask import ModeMask
from .types import CaesMode


# 1 = no dwell lock. Cui 2024 uses start-up cost only (no MUT/MDT).
# Pass min_steps>=2 to restore a consecutive-run lock (tests use 4).
MIN_CAES_RUN_STEPS = 1


@dataclass
class CaesMinimumRunController:
    """CAES 最短运行控制器(CaesMinimumRunController)：锁定已启动的充/放方向直至连续成功运行 min_steps 次。

    安全可行域优先于最短运行：若锁定模式下一步已被 Oracle 禁止，当前段立即审计为 interrupted，
    随后由调用方重新选择普通合法动作；本类不会把动作替换成 idle 或任何其他值。
    """

    min_steps: int = MIN_CAES_RUN_STEPS
    active_mode: CaesMode | None = None
    completed_steps: int = 0
    segments: list[dict[str, Any]] = field(default_factory=list)
    interruptions: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        """重置锁定状态与段/中断审计记录。"""
        self.active_mode = None
        self.completed_steps = 0
        self.segments.clear()
        self.interruptions.clear()

    def constrain(
        self,
        physical_mask: ModeMask,
        *,
        steps_remaining: int,
        step: int | None = None,
    ) -> tuple[ModeMask, dict[str, Any]]:
        """在物理模式掩码(ModeMask)上叠加最短运行规则，返回约束后的掩码与审计状态。

        Args:
            physical_mask: Oracle 给出的物理可行模式掩码。
            steps_remaining: 当前 episode 剩余决策步数。
            step: 可选，当前步索引，用于中断审计。

        Returns:
            (mask, status)：叠加最短运行后的模式掩码，以及含 caes_min_run_event 的状态字典。
        """
        if int(self.min_steps) <= 1:
            return physical_mask, {**self.status(), "caes_min_run_event": None}
        event: dict[str, Any] | None = None
        if self.active_mode is not None and not physical_mask.allows(self.active_mode):
            event = self.interrupt("locked_mode_no_longer_safe", step=step)

        if self.active_mode is not None:
            mode = self.active_mode
            mask = ModeMask(
                discharge=mode == CaesMode.DISCHARGE and physical_mask.discharge,
                idle=False,
                charge=mode == CaesMode.CHARGE and physical_mask.charge,
            )
        elif steps_remaining < self.min_steps:
            # 尾段不能再启动非 idle 段，已启动的段不会走到这里。
            mask = ModeMask(discharge=False, idle=physical_mask.idle, charge=False)
        else:
            mask = physical_mask
        return mask, {**self.status(), "caes_min_run_event": event}

    def record_success(self, mode: CaesMode, *, step: int) -> dict[str, Any] | None:
        """主 FMU 步进成功后推进最短运行计数；幅值不是锁定对象。

        Args:
            mode: 已校验通过的 CAES 模式(CaesMode)。
            step: 当前决策步索引。

        Returns:
            若本段达到 min_steps 则返回完成段审计 dict，否则返回 None。

        Raises:
            RuntimeError: 传入模式与当前锁定模式不一致。
        """
        if mode == CaesMode.IDLE:
            return None
        if self.active_mode is None:
            self.active_mode = mode
            self.completed_steps = 0
        if mode != self.active_mode:
            raise RuntimeError("CAES 最短运行状态与已校验模式不一致")
        self.completed_steps += 1
        if self.completed_steps < self.min_steps:
            return None
        segment = {
            "mode": int(mode),
            "steps": self.completed_steps,
            "completed": True,
            "interrupted": False,
            "end_step": step,
        }
        self.segments.append(segment)
        self.active_mode = None
        self.completed_steps = 0
        return segment

    def interrupt(
        self, reason: str, *, step: int | None = None
    ) -> dict[str, Any] | None:
        """将未达最短时长的活动段记为安全中断并解锁。

        Args:
            reason: 中断原因字符串（审计用）。
            step: 可选，中断发生时的步索引。

        Returns:
            中断事件审计 dict；若无活动段则返回 None。
        """
        if self.active_mode is None:
            return None
        event = {
            "mode": int(self.active_mode),
            "steps": self.completed_steps,
            "completed": False,
            "interrupted": True,
            "reason": reason,
            "end_step": step,
        }
        self.segments.append(event)
        self.interruptions.append(event)
        self.active_mode = None
        self.completed_steps = 0
        return event

    def status(self) -> dict[str, Any]:
        """返回当前最短运行锁定状态的快照。"""
        return {
            "caes_min_run_steps": self.min_steps,
            "caes_locked_mode": (
                None if self.active_mode is None else int(self.active_mode)
            ),
            "caes_locked_steps_completed": self.completed_steps,
            "caes_locked_steps_remaining": (
                0 if self.active_mode is None else self.min_steps - self.completed_steps
            ),
        }

    def summary(self) -> dict[str, Any]:
        """汇总 episode 内所有段完成率与中断统计。"""
        completed = sum(1 for item in self.segments if item["completed"])
        total = len(self.segments)
        return {
            "caes_run_segments": list(self.segments),
            "caes_min_run_completed_segments": completed,
            "caes_min_run_interruption_count": len(self.interruptions),
            "caes_min_run_compliance_rate": 1.0 if total == 0 else completed / total,
        }
