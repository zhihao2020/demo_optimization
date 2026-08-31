"""GiveSafe 转移收集器：拒绝自环写入 GiveSafeReplay；安全动作才执行主 FMU。"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from actions.caes_u import (
    CHARGE_HI,
    CHARGE_LO,
    DISCHARGE_HI,
    DISCHARGE_LO,
    ENDPOINT_SNAP_HARD_FAIL,
    feasible_bound_dict,
    mag_from_u,
    mode_from_u,
)
from actions.validator import physical_from_dict
from replay.hybrid_replay_buffer import HybridGiveSafeReplayBuffer
from safety import GiveSafeController, NoSafeActionFoundError, ShadowFmuValidator
from safety.safety_result import SafetyCheckResult
from safety.soft_constraint_shell import SoftConstraintShell
from .buffer import SafetyDataset, Transition


class GiveSafeTransitionCollector:
    """GiveSafe 转移收集器：拒绝自环写入 GiveSafeReplay；安全动作才执行主 FMU。"""

    def __init__(
        self,
        buffer: HybridGiveSafeReplayBuffer,
        controller: GiveSafeController,
        shadow: ShadowFmuValidator | None = None,
        safety_dataset: SafetyDataset | None = None,
        *,
        soft_shell: bool = False,
    ):
        self.buffer = buffer
        self.controller = controller
        self.shadow = shadow
        self.safety_dataset = safety_dataset if safety_dataset is not None else SafetyDataset()
        self.soft_shell = bool(soft_shell)
        self._shell = SoftConstraintShell() if self.soft_shell else None
        self.stats = {
            "policy_attempt_count": 0,
            "givesafe_rejection_count": 0,
            "oracle_rejection_count": 0,
            "shadow_fmu_rejection_count": 0,
            "givesafe_false_safe_count": 0,
            "no_safe_action_found_count": 0,
            "soft_shell_recovery_count": 0,
            "physical_transition_count": 0,
            "main_fmu_execution_count": 0,
            "post_step_hard_constraint_violation_count": 0,
            "main_fmu_unsafe_execution_count": 0,
            "forbidden_action_attempts": 0,
            "caes_mode_counts": {0: 0, 1: 0, 2: 0},
            "valid_transition_count": 0,
            "rejected_transition_count": 0,
            "fine_failure_counts": {},
            "numerical_endpoint_snap_count": 0,
            "max_endpoint_snap_abs": 0.0,
        }

    def _record_endpoint_snap(self, action: Any) -> None:
        if not isinstance(action, dict):
            return
        if not bool(action.get("caes_endpoint_snapped")):
            return
        self.stats["numerical_endpoint_snap_count"] = (
            int(self.stats.get("numerical_endpoint_snap_count", 0)) + 1
        )
        delta = abs(float(action.get("caes_endpoint_snap_delta", 0.0) or 0.0))
        prev = float(self.stats.get("max_endpoint_snap_abs", 0.0) or 0.0)
        if delta > prev:
            self.stats["max_endpoint_snap_abs"] = delta
        if delta > float(ENDPOINT_SNAP_HARD_FAIL):
            raise RuntimeError(
                f"endpoint snap {delta} exceeds {ENDPOINT_SNAP_HARD_FAIL}; "
                "this is a decoder bug, not float32 ULP"
            )

    def on_episode_reset(self, start_time: float = 0.0) -> None:
        if self._shell is not None:
            self._shell.reset_episode()
        if self.shadow is not None:
            self.shadow.on_episode_reset(start_time)

    def _store_physical_success(
        self,
        env,
        obs_before,
        obs,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
        action: dict,
        sim_time_before: float,
        valid_steps_before: int,
    ) -> tuple[Any, ...]:
        """把一次成功的主 FMU 步进写入 physical replay。"""
        _ = sim_time_before
        assert env.valid_episode_steps == valid_steps_before + 1
        physical = info.get("physical_action") or physical_from_dict(action).as_dict()
        u_c = float(physical["u_caes"])
        mode = int(mode_from_u(u_c))
        physical = dict(physical)
        physical["caes_mode"] = mode
        physical["caes_magnitude"] = float(mag_from_u(u_c))
        self.stats["caes_mode_counts"][mode] = self.stats["caes_mode_counts"].get(mode, 0) + 1
        self.stats["physical_transition_count"] += 1
        self.stats["valid_transition_count"] += 1
        if self.shadow is not None:
            self.shadow.on_physical_success(physical)
        spec = info.get("feasible_action_spec")
        if spec is not None and hasattr(spec, "u_tp_low"):
            bounds = feasible_bound_dict(spec)
        else:
            d = spec if isinstance(spec, dict) else {}

            def _f(key: str, default: float) -> float:
                v = d.get(key, info.get(key, default))
                return float(default if v is None else v)

            bounds = {
                "u_tp_low": _f("u_tp_dynamic_low", 1.0 / 3.0),
                "u_tp_high": _f("u_tp_dynamic_high", 1.0),
                "u_battery_low": _f("u_battery_dynamic_low", -1.0),
                "u_battery_high": _f("u_battery_dynamic_high", 1.0),
                "u_caes_discharge_low": _f("u_caes_discharge_low", DISCHARGE_LO),
                "u_caes_discharge_high": _f("u_caes_discharge_high", DISCHARGE_HI),
                "u_caes_charge_low": _f("u_caes_charge_low", CHARGE_LO),
                "u_caes_charge_high": _f("u_caes_charge_high", CHARGE_HI),
            }
        next_feasible = env.get_feasible_action_spec()
        mask = np.asarray(
            [
                bool(info.get("caes_discharge_allowed", True)),
                bool(info.get("caes_idle_allowed", True)),
                bool(info.get("caes_charge_allowed", True)),
            ],
            dtype=bool,
        )
        terms = dict(info.get("reward_terms") or {})
        terms.setdefault("economic_reward", float(reward))
        terms.setdefault("constraint_reward", float(info.get("constraint_reward", 0.0) or 0.0))
        terms.setdefault("total_training_reward", float(reward))
        tr = Transition(
            observation=np.asarray(obs_before, dtype=np.float32),
            hybrid_action=dict(physical),
            decoded_fmu_action=dict(physical),
            reward=float(reward),
            next_observation=np.asarray(obs, dtype=np.float32),
            terminated=bool(terminated or truncated),
            truncated=bool(truncated),
            valid_mode_mask=mask,
            dynamic_action_bounds=bounds,
            next_valid_mode_mask=next_feasible.mode_mask.as_bool_array(),
            next_dynamic_action_bounds=feasible_bound_dict(next_feasible),
            reward_terms=terms,
            constraint_metadata={"soft_shell": bool(info.get("soft_shell_applied"))},
            physically_valid=True,
            oracle_predicted_next_state=info.get("oracle_predicted_next_state"),
            residuals=info.get("residuals"),
            distance_to_physical_boundary=info.get("distance_to_physical_boundary"),
            distance_to_safe_boundary=info.get("distance_to_safe_boundary"),
            oracle_version=info.get("oracle_version"),
            transition_type="physical",
        )
        stored = self.buffer.add_physical(tr)
        info["stored_in_physical_replay"] = stored
        info["stored_in_givesafe_replay"] = False
        info["transition_type"] = "physical"
        info.setdefault("constraint_reward", 0.0)
        info["economic_reward"] = float(terms.get("economic_reward", reward))
        info["total_training_reward"] = float(reward)
        info["action_executed_by_main_fmu"] = True
        return obs, reward, terminated, truncated, info

    def _store_rejection(
        self,
        env,
        obs: np.ndarray,
        action: dict,
        safety: SafetyCheckResult,
        terms: dict[str, float],
    ) -> None:
        physical = physical_from_dict(action)
        bounds = {
            "u_tp_low": float(
                getattr(env, "_current_feasible", None).u_tp_low
                if getattr(env, "_current_feasible", None)
                else 1 / 3
            ),
            "u_tp_high": float(
                getattr(env, "_current_feasible", None).u_tp_high
                if getattr(env, "_current_feasible", None)
                else 1.0
            ),
            "u_battery_low": float(
                getattr(env, "_current_feasible", None).u_battery_low
                if getattr(env, "_current_feasible", None)
                else -1.0
            ),
            "u_battery_high": float(
                getattr(env, "_current_feasible", None).u_battery_high
                if getattr(env, "_current_feasible", None)
                else 1.0
            ),
        }
        try:
            feasible = env.get_feasible_action_spec()
            mask = feasible.mode_mask.as_bool_array()
            bounds = {
                "u_tp_low": feasible.u_tp_low,
                "u_tp_high": feasible.u_tp_high,
                "u_battery_low": feasible.u_battery_low,
                "u_battery_high": feasible.u_battery_high,
            }
        except Exception:
            mask = np.ones(3, dtype=bool)
        act = physical.as_dict()
        tr = Transition(
            observation=np.asarray(obs, dtype=np.float32),
            hybrid_action=act,
            decoded_fmu_action=act,
            reward=float(terms["constraint_reward"]),
            next_observation=np.asarray(obs, dtype=np.float32),
            terminated=False,
            truncated=False,
            valid_mode_mask=mask,
            dynamic_action_bounds=bounds,
            next_valid_mode_mask=mask,
            next_dynamic_action_bounds=dict(bounds),
            reward_terms=dict(terms),
            constraint_metadata={
                "rejection_stage": safety.rejection_stage,
                "violation_type": safety.violation_type,
                "normalized_violations": safety.normalized_violations,
                "shadow_validation_used": safety.shadow_validation_used,
                "shadow_failure_reason": safety.shadow_failure_reason,
            },
            physically_valid=False,
            oracle_predicted_next_state=dict(safety.predicted_next_state or {}),
            oracle_version=self.controller.oracle.oracle_version,
            transition_type="givesafe_rejection",
        )
        self.buffer.add_givesafe_rejection(tr)
        self.stats["givesafe_rejection_count"] += 1
        self.stats["rejected_transition_count"] += 1
        if safety.rejection_stage == "oracle":
            self.stats["oracle_rejection_count"] += 1
            self.stats["forbidden_action_attempts"] += 1
        elif safety.rejection_stage == "shadow":
            self.stats["shadow_fmu_rejection_count"] += 1
        fine = safety.violation_type or "unknown"
        self.stats["fine_failure_counts"][fine] = (
            self.stats["fine_failure_counts"].get(fine, 0) + 1
        )

    def step_with_givesafe(
        self,
        env,
        propose_fn: Callable[[], dict],
        *,
        deterministic: bool = False,
    ) -> tuple[Any, ...]:
        if env.last_outputs is None:
            raise RuntimeError("环境未 reset")
        obs_before = env.build_observation()
        sim_time_before = float(getattr(env.adapter, "time", 0.0))
        valid_steps_before = int(env.valid_episode_steps)
        feasible = env.get_feasible_action_spec()

        def propose_and_audit():
            action = propose_fn()
            self._record_endpoint_snap(action)
            return action

        def on_rejection(action, safety, terms):
            self.stats["policy_attempt_count"] += 1
            self._store_rejection(env, obs_before, action, safety, terms)

        try:
            gs = self.controller.select_safe_action(
                env.last_outputs,
                env.previous_thermal,
                propose_and_audit,
                deterministic=deterministic,
                on_rejection=on_rejection,
                feasible_override=feasible,
            )
        except NoSafeActionFoundError as exc:
            self.stats["no_safe_action_found_count"] += 1
            self.stats["policy_attempt_count"] += int(exc.attempts)
            if not self.soft_shell or self._shell is None:
                info = {
                    "transition_valid": False,
                    "physically_valid": False,
                    "failure_type": "NoSafeActionFound",
                    "failure_reason": str(exc),
                    "action_executed_by_main_fmu": False,
                    "transition_type": None,
                    "stored_in_physical_replay": False,
                    "stored_in_givesafe_replay": True,
                    "givesafe_attempt_count": exc.attempts,
                    "simulation_time_unchanged": True,
                }
                return obs_before, 0.0, False, True, info
            # 软外壳：保守动作推进主 FMU（非 GiveSafe use_fallback）
            action = self._shell.recover(env)
            self.stats["soft_shell_recovery_count"] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            self.stats["main_fmu_execution_count"] += 1
            info = dict(info)
            info["givesafe_attempt_count"] = int(exc.attempts)
            info["soft_shell_applied"] = True
            info["soft_shell_recovered_from"] = "NoSafeActionFound"
            if info.get("transition_valid") and info.get("physically_valid"):
                reward, info = self._shell.apply_penalty(float(reward), info)
                return self._store_physical_success(
                    env,
                    obs_before,
                    obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                    action,
                    sim_time_before,
                    valid_steps_before,
                )
            info["action_executed_by_main_fmu"] = bool(info.get("transition_valid"))
            info["stored_in_physical_replay"] = False
            info["stored_in_givesafe_replay"] = False
            info["transition_type"] = "soft_shell_failed"
            return obs_before, float(reward), bool(terminated), True, info

        self.stats["policy_attempt_count"] += 1
        assert gs.safe_action is not None
        action = gs.safe_action
        obs, reward, terminated, truncated, info = env.step(action)
        self.stats["main_fmu_execution_count"] += 1

        info["givesafe_attempt_count"] = gs.attempt_count
        info["givesafe_rejected_attempts"] = len(gs.rejected_actions)
        info["action_executed_by_main_fmu"] = bool(info.get("transition_valid"))
        info["policy_attempt_index"] = gs.attempt_count

        valid = bool(info.get("physically_valid") and info.get("transition_valid"))
        if not valid:
            self.stats["givesafe_false_safe_count"] += 1
            self.stats["post_step_hard_constraint_violation_count"] += 1
            self.stats["main_fmu_unsafe_execution_count"] += 1
            self.stats["rejected_transition_count"] += 1
            physical = physical_from_dict(action)
            terms = self.controller.reward_calc.calculate(
                SafetyCheckResult(
                    safe=False,
                    rejection_stage="false_safe",
                    violation_type=info.get("fine_failure_type") or "unknown",
                    violation_severity=1.0,
                    normalized_violations={"unknown": 1.0},
                )
            )
            mask = np.ones(3, dtype=bool)
            bounds = {
                "u_tp_low": 1 / 3,
                "u_tp_high": 1.0,
                "u_battery_low": -1.0,
                "u_battery_high": 1.0,
            }
            tr = Transition(
                observation=np.asarray(obs_before, dtype=np.float32),
                hybrid_action=physical.as_dict(),
                decoded_fmu_action=physical.as_dict(),
                reward=float(terms["constraint_reward"]),
                next_observation=np.asarray(obs_before, dtype=np.float32),
                terminated=False,
                truncated=True,
                valid_mode_mask=mask,
                dynamic_action_bounds=bounds,
                reward_terms=dict(terms),
                constraint_metadata={
                    "failure": info.get("failure_type"),
                    "fine": info.get("fine_failure_type"),
                },
                physically_valid=False,
                transition_type="givesafe_rejection",
            )
            self.buffer.add_givesafe_rejection(tr)
            if info.get("failure_record") or info.get("actual_fmu_outputs"):
                self.safety_dataset.add_from_failure_record(
                    {
                        **(info.get("failure_record") or {}),
                        "previous_observation": dict(env.last_outputs or {}),
                        "hybrid_action": info.get("physical_action"),
                        "label_safe": False,
                    }
                )
            info["stored_in_physical_replay"] = False
            info["stored_in_givesafe_replay"] = True
            info["transition_type"] = "givesafe_false_safe"
            assert float(getattr(env.adapter, "time", sim_time_before)) >= sim_time_before
            return obs_before, float(terms["constraint_reward"]), False, True, info

        return self._store_physical_success(
            env,
            obs_before,
            obs,
            float(reward),
            terminated,
            truncated,
            info,
            action,
            sim_time_before,
            valid_steps_before,
        )
