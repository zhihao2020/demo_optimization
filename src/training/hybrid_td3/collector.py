"""只收集 physically_valid 转移；失败写入 SafetyDataset。"""

from __future__ import annotations

from typing import Any

import numpy as np

from actions import CaesMode, HybridAction
from actions.validator import hybrid_from_dict

from .buffer import EconomicReplayBuffer, FilteredReplayBuffer, SafetyDataset, Transition


class ValidTransitionCollector:
    """有效转移收集器(ValidTransitionCollector)：仅经济 replay 存物理有效步，失败进安全数据集。"""

    def __init__(
        self,
        buffer: FilteredReplayBuffer | EconomicReplayBuffer,
        safety_dataset: SafetyDataset | None = None,
    ):
        """绑定经济 replay 与可选安全数据集。

        Args:
            buffer: 过滤经济 replay 缓冲区。
            safety_dataset: 安全样本集；None 时自动创建空集。
        """
        self.buffer = buffer
        self.safety_dataset = safety_dataset if safety_dataset is not None else SafetyDataset()
        self.stats = {
            "forbidden_action_attempts": 0,
            "precheck_rejections": 0,
            "post_step_constraint_failures": 0,
            "fmu_numerical_failures": 0,
            "fmi_failures": 0,
            "feasible_set_empty": 0,
            "valid_transition_count": 0,
            "rejected_transition_count": 0,
            "caes_mode_counts": {0: 0, 1: 0, 2: 0},
            "fine_failure_counts": {},
        }

    def step_and_store(self, env, policy_action: dict | HybridAction) -> tuple[Any, ...]:
        """执行 env.step 并按有效性分流到 replay / SafetyDataset。

        Args:
            env: 电力系统环境。
            policy_action: 策略输出的混合动作。

        Returns:
            与 ``env.step`` 相同格式的 (obs, reward, terminated, truncated, info) 元组；
            可行集为空时返回自环占位 info。
        """
        obs_before = env.build_observation()
        prev_outputs = dict(env.last_outputs) if env.last_outputs else {}
        try:
            feasible = env.get_feasible_action_spec()
        except Exception as exc:
            # FeasibleSetEmpty 等
            from envs.failures import FeasibleSetEmpty
            if isinstance(exc, FeasibleSetEmpty):
                self.stats["feasible_set_empty"] += 1
                self.stats["rejected_transition_count"] += 1
                self.stats["fine_failure_counts"]["feasible_set_empty"] = (
                    self.stats["fine_failure_counts"].get("feasible_set_empty", 0) + 1
                )
                info = {
                    "transition_valid": False,
                    "physically_valid": False,
                    "failure_type": "FeasibleSetEmpty",
                    "fine_failure_type": "feasible_set_empty",
                    "failure_reason": str(exc),
                }
                return obs_before, 0.0, False, True, info
            raise
        obs, reward, terminated, truncated, info = env.step(policy_action)
        valid = bool(info.get("physically_valid") and info.get("transition_valid"))
        ft = info.get("failure_type")
        fine = info.get("fine_failure_type")
        if fine:
            self.stats["fine_failure_counts"][fine] = self.stats["fine_failure_counts"].get(fine, 0) + 1
        if ft in ("StaticActionViolation", "ForbiddenModeViolation", "DynamicStateConstraintViolation"):
            self.stats["forbidden_action_attempts"] += 1
            self.stats["precheck_rejections"] += 1
        elif ft == "PostStepHardConstraintViolation":
            self.stats["post_step_constraint_failures"] += 1
        elif ft == "FmuNumericalFailure":
            self.stats["fmu_numerical_failures"] += 1
        elif ft == "FmiLifecycleFailure":
            self.stats["fmi_failures"] += 1
        elif ft == "FeasibleSetEmpty":
            self.stats["feasible_set_empty"] += 1
        if not valid:
            self.stats["rejected_transition_count"] += 1
            # 写入 SafetyDataset，明确不写经济 buffer
            if ft == "PostStepHardConstraintViolation" or info.get("actual_fmu_outputs") is not None:
                self.safety_dataset.add_from_failure_record(
                    {
                        **(info.get("failure_record") or {}),
                        "previous_observation": prev_outputs,
                        "hybrid_action": info.get("hybrid_action"),
                        "decoded_fmu_action": {
                            "u_tp": info.get("decoded_u_tp"),
                            "u_battery": info.get("decoded_u_battery"),
                            "u_caes": info.get("decoded_u_caes"),
                        },
                        "oracle_predicted_next_state": info.get("oracle_predicted_next_state"),
                        "actual_fmu_outputs": info.get("actual_fmu_outputs"),
                        "residuals": info.get("residuals"),
                        "dangerous_residual": info.get("dangerous_residual"),
                        "distance_to_physical_boundary": info.get("distance_to_physical_boundary"),
                        "distance_to_safe_boundary": info.get("distance_to_safe_boundary"),
                        "fine_failure_type": fine,
                        "triggering_constraint": info.get("triggering_constraint"),
                        "modelica_assert_message": info.get("modelica_assert_message") or info.get("failure_reason"),
                        "oracle_version": info.get("oracle_version"),
                        "episode": info.get("episode"),
                        "step": info.get("step"),
                        "run_id": getattr(env, "run_id", None),
                        "last_valid_state": prev_outputs,
                    }
                )
            dummy = Transition(
                observation=obs_before,
                hybrid_action={},
                decoded_fmu_action={},
                reward=0.0,
                next_observation=obs_before,
                terminated=True,
                valid_mode_mask=feasible.mode_mask.as_bool_array(),
                dynamic_action_bounds={
                    "u_tp_low": feasible.u_tp_low,
                    "u_tp_high": feasible.u_tp_high,
                    "u_battery_low": feasible.u_battery_low,
                    "u_battery_high": feasible.u_battery_high,
                },
                reward_terms={},
                physically_valid=False,
            )
            self.buffer.add(dummy)
            return obs, reward, terminated, truncated, info
        hybrid = hybrid_from_dict(info["hybrid_action"]) if "hybrid_action" in info else hybrid_from_dict(policy_action)
        self.stats["caes_mode_counts"][int(hybrid.caes_mode)] = (
            self.stats["caes_mode_counts"].get(int(hybrid.caes_mode), 0) + 1
        )
        next_feasible = env.get_feasible_action_spec() if env.last_outputs is not None else feasible
        # safe 样本也进入 SafetyDataset（供校准）
        self.safety_dataset.add_safe_transition(
            previous_observation=prev_outputs,
            hybrid_action={
                "u_tp": float(hybrid.u_tp),
                "u_battery": float(hybrid.u_battery),
                "caes_mode": int(hybrid.caes_mode),
                "caes_magnitude": float(0.0 if hybrid.caes_mode == CaesMode.IDLE else hybrid.caes_magnitude),
            },
            decoded_fmu_action={
                "u_tp": float(info["decoded_u_tp"]),
                "u_battery": float(info["decoded_u_battery"]),
                "u_caes": float(info["decoded_u_caes"]),
            },
            predicted=info.get("oracle_predicted_next_state"),
            actual=info.get("observations"),
            residuals=info.get("residuals"),
            distances_physical=info.get("distance_to_physical_boundary"),
            distances_safe=info.get("distance_to_safe_boundary"),
            oracle_version=info.get("oracle_version"),
        )
        transition = Transition(
            observation=np.asarray(obs_before, dtype=np.float32),
            hybrid_action={
                "u_tp": float(hybrid.u_tp),
                "u_battery": float(hybrid.u_battery),
                "caes_mode": int(hybrid.caes_mode),
                "caes_magnitude": float(0.0 if hybrid.caes_mode == CaesMode.IDLE else hybrid.caes_magnitude),
            },
            decoded_fmu_action={
                "u_tp": float(info["decoded_u_tp"]),
                "u_battery": float(info["decoded_u_battery"]),
                "u_caes": float(info["decoded_u_caes"]),
            },
            reward=float(reward),
            next_observation=np.asarray(obs, dtype=np.float32),
            terminated=bool(terminated or truncated),
            valid_mode_mask=feasible.mode_mask.as_bool_array(),
            dynamic_action_bounds={
                "u_tp_low": feasible.u_tp_low,
                "u_tp_high": feasible.u_tp_high,
                "u_battery_low": feasible.u_battery_low,
                "u_battery_high": feasible.u_battery_high,
            },
            next_valid_mode_mask=next_feasible.mode_mask.as_bool_array(),
            next_dynamic_action_bounds={
                "u_tp_low": next_feasible.u_tp_low,
                "u_tp_high": next_feasible.u_tp_high,
                "u_battery_low": next_feasible.u_battery_low,
                "u_battery_high": next_feasible.u_battery_high,
            },
            reward_terms=dict(info.get("reward_terms") or {}),
            constraint_metadata={"failure_type": None},
            physically_valid=True,
            oracle_predicted_next_state=info.get("oracle_predicted_next_state"),
            residuals=info.get("residuals"),
            distance_to_physical_boundary=info.get("distance_to_physical_boundary"),
            distance_to_safe_boundary=info.get("distance_to_safe_boundary"),
            safety_probability=info.get("safety_probability"),
            safety_threshold=info.get("safety_threshold"),
            oracle_version=info.get("oracle_version"),
        )
        stored = self.buffer.add(transition)
        if stored:
            self.stats["valid_transition_count"] += 1
        info["stored_in_replay"] = stored
        return obs, reward, terminated, truncated, info
