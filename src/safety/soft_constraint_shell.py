"""软约束外壳：GiveSafe / FMU 外的可切换恢复层。

不修改 GiveSafe 的 ``use_fallback`` 契约。默认关闭；打开后：
- 策略侧：GiveSafe 找不到动作时返回保守合法动作；
- 环境侧：预检拒绝且主 FMU 未推进时，用保守动作再试一次并扣分；
- 后验硬约束 / 求解器失败仍中止（孪生上限）。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from actions import CaesMode
from actions.caes_u import u_from_mode_mag
from envs.failures import FeasibleSetEmpty
from safety.constraint_reward import ConstraintRewardCalculator
from safety.safety_result import SafetyCheckResult

# 预检类失败：主 FMU 未推进，允许外壳二次步进
PRECHECK_FAILURE_TYPES = frozenset(
    {
        "StaticActionViolation",
        "ForbiddenModeViolation",
        "DynamicStateConstraintViolation",
        "FeasibleSetEmpty",
    }
)

# 后验 / 求解器：不再二次步进
NO_RETRY_FAILURE_TYPES = frozenset(
    {
        "PostStepHardConstraintViolation",
        "FmuNumericalFailure",
        "FmiLifecycleFailure",
        "NonFiniteOutputFailure",
    }
)

_DEFAULT_RECOVERY_WEIGHT = 5.0


def conservative_recover_action(env: Any) -> dict[str, np.ndarray]:
    """在当前可行域内取保守动作：满火电 + 电池 0（或中点）+ 优先 idle。

    Args:
        env: 需实现 ``get_feasible_action_spec`` 的电力系统环境。

    Returns:
        可直接 ``env.step`` 的三连续动作字典。
    """
    try:
        feasible = env.get_feasible_action_spec()
    except FeasibleSetEmpty:
        return {
            "u_tp": np.asarray([1.0], dtype=np.float32),
            "u_battery": np.asarray([0.0], dtype=np.float32),
            "u_caes": np.asarray([0.0], dtype=np.float32),
        }
    except Exception:
        return {
            "u_tp": np.asarray([1.0], dtype=np.float32),
            "u_battery": np.asarray([0.0], dtype=np.float32),
            "u_caes": np.asarray([0.0], dtype=np.float32),
        }

    lo_tp = float(feasible.u_tp_low)
    hi_tp = float(feasible.u_tp_high)
    lo_b = float(feasible.u_battery_low)
    hi_b = float(feasible.u_battery_high)
    u_tp = hi_tp
    u_bat = 0.0 if lo_b <= 0.0 <= hi_b else 0.5 * (lo_b + hi_b)

    mask = feasible.mode_mask
    if mask.idle:
        mode, mag = CaesMode.IDLE, 0.0
    elif mask.discharge:
        mode, mag = CaesMode.DISCHARGE, 0.2
    elif mask.charge:
        mode, mag = CaesMode.CHARGE, 0.2
    else:
        # Oracle 掩码全空：仍尝试 idle（与 linprog 空集分支一致）
        mode, mag = CaesMode.IDLE, 0.0

    return {
        "u_tp": np.asarray([float(np.clip(u_tp, lo_tp, hi_tp))], dtype=np.float32),
        "u_battery": np.asarray([float(np.clip(u_bat, lo_b, hi_b))], dtype=np.float32),
        "u_caes": np.asarray([float(u_from_mode_mag(mode, mag))], dtype=np.float32),
    }


class SoftConstraintShell:
    """软约束外壳：保守恢复 + 约束罚项计数。"""

    def __init__(
        self,
        *,
        recovery_weight: float = _DEFAULT_RECOVERY_WEIGHT,
        base_rejection_cost: float = 1.0,
        reward_calc: ConstraintRewardCalculator | None = None,
    ):
        """初始化外壳与罚项计算器。

        Args:
            recovery_weight: ``soft_shell_recovery`` 违规权重。
            base_rejection_cost: 约束奖励基础成本。
            reward_calc: 可选已构造的约束奖励计算器。
        """
        self.recovery_count = 0
        if reward_calc is not None:
            self.reward_calc = reward_calc
        else:
            self.reward_calc = ConstraintRewardCalculator(
                {
                    "base_rejection_cost": float(base_rejection_cost),
                    "weights": {"soft_shell_recovery": float(recovery_weight)},
                }
            )

    def recover(self, env: Any) -> dict[str, np.ndarray]:
        """返回保守合法动作并增加命中计数。"""
        self.recovery_count += 1
        return conservative_recover_action(env)

    def penalty_terms(self) -> dict[str, float]:
        """外壳恢复对应的约束罚项（负奖励）。"""
        safety = SafetyCheckResult(
            safe=False,
            rejection_stage="soft_shell",
            violation_type="soft_shell_recovery",
            violation_severity=1.0,
            normalized_violations={"soft_shell_recovery": 1.0},
        )
        return self.reward_calc.calculate(safety)

    def apply_penalty(self, reward: float, info: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        """把外壳罚项并入本步奖励与 ``info``。"""
        terms = self.penalty_terms()
        pen = float(terms["constraint_reward"])
        out = dict(info)
        out["soft_shell_applied"] = True
        out["soft_shell_count"] = int(self.recovery_count)
        merged = dict(out.get("reward_terms") or {})
        merged.update(terms)
        merged["economic_reward"] = float(reward)
        merged["total_training_reward"] = float(reward) + pen
        out["reward_terms"] = merged
        out["constraint_reward"] = pen
        return float(reward) + pen, out

    def reset_episode(self) -> None:
        """回合开始时清零计数（可选；累计也可跨回合保留）。"""
        self.recovery_count = 0


def is_precheck_failure(info: Mapping[str, Any]) -> bool:
    """是否为「主 FMU 未推进」的预检类失败。"""
    ft = info.get("failure_type")
    if ft not in PRECHECK_FAILURE_TYPES:
        return False
    if info.get("transition_valid"):
        return False
    # 未调用主 FMU，或显式标记时钟未动
    status = info.get("fmu_status")
    if status in ("not_called", None) and not info.get("action_executed_by_main_fmu"):
        return True
    if info.get("simulation_time_unchanged"):
        return True
    if status == "not_called":
        return True
    return False


def is_no_retry_failure(info: Mapping[str, Any]) -> bool:
    """后验 / 求解器失败，外壳不得二次步进。"""
    return info.get("failure_type") in NO_RETRY_FAILURE_TYPES


class SoftConstraintEnv:
    """环境包装：预检失败时用保守动作再试一次；后验失败原样返回。"""

    def __init__(self, env: Any, shell: SoftConstraintShell | None = None):
        """绑定底层环境与外壳实例。

        Args:
            env: ``PowerSystemEnv`` 或兼容对象。
            shell: 可选已有外壳；默认新建。
        """
        self.env = env
        self.shell = shell if shell is not None else SoftConstraintShell()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        self.shell.reset_episode()
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not is_precheck_failure(info):
            return obs, reward, terminated, truncated, info
        if is_no_retry_failure(info):
            return obs, reward, terminated, truncated, info

        recover = self.shell.recover(self.env)
        obs2, reward2, term2, trunc2, info2 = self.env.step(recover)
        if info2.get("transition_valid") and info2.get("physically_valid"):
            reward2, info2 = self.shell.apply_penalty(float(reward2), info2)
            info2["soft_shell_recovered_from"] = info.get("failure_type")
            return obs2, reward2, term2, trunc2, info2

        # 二次步进仍失败：原样返回第二次结果（可能仍是空集或预检）
        info2 = dict(info2)
        info2["soft_shell_applied"] = True
        info2["soft_shell_recovery_failed"] = True
        info2["soft_shell_recovered_from"] = info.get("failure_type")
        info2["soft_shell_count"] = int(self.shell.recovery_count)
        return obs2, reward2, term2, trunc2, info2

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
