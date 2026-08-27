"""FS-HSAC collector: routes physical / rejection into split replay."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from replay.fs_hsac_replay import FSHSACReplayBuffer
from safety import NoSafeActionFoundError
from safety.givesafe_controller import GiveSafeController


class FSHSACCollector:
    """Twin-closed loop (paper Alg. 1): GiveSafe, then one FMU hour, split replay.

    Accepted physical steps go to the Bellman buffer. Rejections never advance
    the twin and train only the residual feasibility classifier.
    """

    def __init__(self, buffer: FSHSACReplayBuffer, controller: GiveSafeController):
        self.buffer = buffer
        self.controller = controller
        self.stats = {
            "givesafe_rejection_count": 0,
            "oracle_rejection_count": 0,
            "shadow_fmu_rejection_count": 0,
            "physical_count": 0,
            "post_step_failure_count": 0,
            "no_safe_action_found_count": 0,
            "policy_attempt_count": 0,
            "main_fmu_execution_count": 0,
            "forbidden_action_attempts": 0,
            "fine_failure_counts": {},
        }

    def on_episode_reset(self, start_time: float) -> None:
        reset = getattr(self.controller, "on_episode_reset", None)
        if callable(reset):
            reset(start_time)

    def step_with_givesafe(
        self,
        env,
        propose_fn: Callable[[], dict],
        *,
        deterministic: bool = False,
        feasible=None,
    ) -> tuple[Any, ...]:
        if env.last_outputs is None:
            raise RuntimeError("环境未 reset")
        obs_before = np.asarray(env.build_observation(), dtype=np.float32)
        if feasible is None:
            feasible = env.get_feasible_action_spec()

        def on_rejection(action, safety, terms):
            self.stats["policy_attempt_count"] += 1
            stage = str(getattr(safety, "rejection_stage", "") or "oracle")
            self.buffer.add_rejection(
                obs=obs_before,
                action=action,
                feasible=feasible,
                failure_type=str(getattr(safety, "violation_type", None) or stage),
                metadata={
                    "rejection_stage": stage,
                    "constraint_reward": float((terms or {}).get("constraint_reward", 0.0)),
                },
            )
            self.stats["givesafe_rejection_count"] += 1
            if stage == "oracle":
                self.stats["oracle_rejection_count"] += 1
                self.stats["forbidden_action_attempts"] += 1
            elif stage == "shadow":
                self.stats["shadow_fmu_rejection_count"] += 1
            fine = getattr(safety, "violation_type", None) or "unknown"
            self.stats["fine_failure_counts"][fine] = (
                self.stats["fine_failure_counts"].get(fine, 0) + 1
            )

        try:
            gs = self.controller.select_safe_action(
                env.last_outputs,
                env.previous_thermal,
                propose_fn,
                deterministic=deterministic,
                on_rejection=on_rejection,
                feasible_override=feasible,
            )
        except NoSafeActionFoundError as exc:
            self.stats["no_safe_action_found_count"] += 1
            self.stats["policy_attempt_count"] += int(exc.attempts)
            info = {
                "transition_valid": False,
                "physically_valid": False,
                "failure_type": "NoSafeActionFound",
                "failure_reason": str(exc),
                "action_executed_by_main_fmu": False,
                "transition_type": None,
                "givesafe_attempt_count": exc.attempts,
                "givesafe_rejected_attempts": int(exc.attempts),
            }
            return obs_before, 0.0, False, True, info

        self.stats["policy_attempt_count"] += 1
        assert gs.safe_action is not None
        action = gs.safe_action
        obs, reward, terminated, truncated, info = env.step(action)
        self.stats["main_fmu_execution_count"] += 1
        info = dict(info or {})
        info["givesafe_attempt_count"] = int(getattr(gs, "attempt_count", 1) or 1)
        info["givesafe_rejected_attempts"] = max(0, info["givesafe_attempt_count"] - 1)

        if not info.get("transition_valid", True) or not info.get("physically_valid", True):
            try:
                next_feas = env.get_feasible_action_spec()
            except Exception:
                next_feas = feasible
            self.buffer.add_post_step_failure(
                obs=obs_before,
                next_obs=np.asarray(obs, dtype=np.float32),
                action=action,
                feasible=feasible,
                next_feasible=next_feas,
                failure_type=str(info.get("fine_failure_type") or info.get("failure_type") or "post_step"),
            )
            self.stats["post_step_failure_count"] += 1
            info["transition_type"] = "post_step_failure"
            info["transition_valid"] = False
            return obs, reward, terminated, truncated, info

        try:
            next_feas = env.get_feasible_action_spec()
        except Exception:
            next_feas = feasible
        self.buffer.add_physical(
            obs=obs_before,
            next_obs=np.asarray(obs, dtype=np.float32),
            action=action,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            feasible=feasible,
            next_feasible=next_feas,
        )
        self.stats["physical_count"] += 1
        info["transition_type"] = "physical"
        info["transition_valid"] = True
        return obs, reward, terminated, truncated, info
