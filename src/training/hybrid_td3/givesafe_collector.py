"""GiveSafe 转移收集器：拒绝自环写入 GiveSafeReplay；安全动作才执行主 FMU。"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from actions import CaesMode, HybridAction, HybridActionDecoder
from actions.validator import hybrid_from_dict
from replay.hybrid_replay_buffer import HybridGiveSafeReplayBuffer
from safety import GiveSafeController, NoSafeActionFoundError, ShadowFmuValidator
from safety.safety_result import SafetyCheckResult
from .buffer import SafetyDataset, Transition


class GiveSafeTransitionCollector:
    """GiveSafe 转移收集器(GiveSafeTransitionCollector)：拒绝自环写入 GiveSafeReplay；安全动作才执行主 FMU。"""

    def __init__(
        self,
        buffer: HybridGiveSafeReplayBuffer,
        controller: GiveSafeController,
        shadow: ShadowFmuValidator | None = None,
        safety_dataset: SafetyDataset | None = None,
    ):
        """绑定 GiveSafe replay、控制器与影子 FMU。

        Args:
            buffer: 混合 GiveSafe 分区 replay。
            controller: GiveSafe 安全控制器。
            shadow: 可选影子 FMU 校验器。
            safety_dataset: 安全样本集；None 时自动创建。
        """
        self.buffer = buffer
        self.controller = controller
        self.shadow = shadow
        self.safety_dataset = safety_dataset if safety_dataset is not None else SafetyDataset()
        self.decoder = HybridActionDecoder()
        self.stats = {
            "policy_attempt_count": 0,
            "givesafe_rejection_count": 0,
            "oracle_rejection_count": 0,
            "shadow_fmu_rejection_count": 0,
            "givesafe_false_safe_count": 0,
            "no_safe_action_found_count": 0,
            "physical_transition_count": 0,
            "main_fmu_execution_count": 0,
            "post_step_hard_constraint_violation_count": 0,
            "main_fmu_unsafe_execution_count": 0,
            "forbidden_action_attempts": 0,
            "caes_mode_counts": {0: 0, 1: 0, 2: 0},
            "valid_transition_count": 0,
            "rejected_transition_count": 0,
            "fine_failure_counts": {},
        }

    def on_episode_reset(self, start_time: float = 0.0) -> None:
        """回合重置时通知影子 FMU。

        Args:
            start_time: 仿真起始时间（秒）。

        Returns:
            无。
        """
        if self.shadow is not None:
            self.shadow.on_episode_reset(start_time)

    def _store_rejection(
        self,
        env,
        obs: np.ndarray,
        action: dict,
        safety: SafetyCheckResult,
        terms: dict[str, float],
    ) -> None:
        """将 GiveSafe 拒绝样本以自环转移写入 givesafe replay。

        Args:
            env: 当前环境，用于查询可行域。
            obs: 步前观测。
            action: 被拒绝的混合动作。
            safety: GiveSafe 检查结果。
            terms: 约束奖励分项。

        Returns:
            无。
        """
        hybrid = hybrid_from_dict(action)
        physical = self.decoder.decode(hybrid)
        bounds = {
            "u_tp_low": float(getattr(env, "_current_feasible", None).u_tp_low if getattr(env, "_current_feasible", None) else 1 / 3),
            "u_tp_high": float(getattr(env, "_current_feasible", None).u_tp_high if getattr(env, "_current_feasible", None) else 1.0),
            "u_battery_low": float(getattr(env, "_current_feasible", None).u_battery_low if getattr(env, "_current_feasible", None) else -1.0),
            "u_battery_high": float(getattr(env, "_current_feasible", None).u_battery_high if getattr(env, "_current_feasible", None) else 1.0),
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
        tr = Transition(
            observation=np.asarray(obs, dtype=np.float32),
            hybrid_action={
                "u_tp": float(hybrid.u_tp),
                "u_battery": float(hybrid.u_battery),
                "caes_mode": int(hybrid.caes_mode),
                "caes_magnitude": float(0.0 if hybrid.caes_mode == CaesMode.IDLE else hybrid.caes_magnitude),
            },
            decoded_fmu_action=physical.as_dict(),
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
        self.stats["fine_failure_counts"][fine] = self.stats["fine_failure_counts"].get(fine, 0) + 1

    def step_with_givesafe(
        self,
        env,
        propose_fn: Callable[[], dict],
        *,
        deterministic: bool = False,
    ) -> tuple[Any, ...]:
        """经 GiveSafe 环执行一步：拒绝不推进主 FMU，成功才写入 physical replay。

        Args:
            env: 电力系统环境，需已 reset。
            propose_fn: 无环境副作用的候选动作采样 callable。
            deterministic: 是否确定性 GiveSafe 搜索。

        Returns:
            (obs, reward, terminated, truncated, info) 元组。

        Raises:
            RuntimeError: 环境未 reset 时抛出。
        """
        if env.last_outputs is None:
            raise RuntimeError("环境未 reset")
        obs_before = env.build_observation()
        sim_time_before = float(getattr(env.adapter, "time", 0.0))
        valid_steps_before = int(env.valid_episode_steps)
        feasible = env.get_feasible_action_spec()

        def on_rejection(action, safety, terms):
            """记录一次被安全给予拒绝的策略尝试。"""
            self.stats["policy_attempt_count"] += 1
            self._store_rejection(env, obs_before, action, safety, terms)

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
                "stored_in_physical_replay": False,
                "stored_in_givesafe_replay": True,
                "givesafe_attempt_count": exc.attempts,
                "simulation_time_unchanged": True,
            }
            return obs_before, 0.0, False, True, info

        # 计入最后一次成功尝试
        self.stats["policy_attempt_count"] += 1
        assert gs.safe_action is not None
        action = gs.safe_action
        obs, reward, terminated, truncated, info = env.step(action)
        self.stats["main_fmu_execution_count"] += 1

        # 时间/物理步语义校验字段
        info["givesafe_attempt_count"] = gs.attempt_count
        info["givesafe_rejected_attempts"] = len(gs.rejected_actions)
        info["action_executed_by_main_fmu"] = bool(info.get("transition_valid"))
        info["policy_attempt_index"] = gs.attempt_count

        valid = bool(info.get("physically_valid") and info.get("transition_valid"))
        if not valid:
            # false-safe：一级+shadow 通过但主 FMU 后验失败
            self.stats["givesafe_false_safe_count"] += 1
            self.stats["post_step_hard_constraint_violation_count"] += 1
            self.stats["main_fmu_unsafe_execution_count"] += 1
            self.stats["rejected_transition_count"] += 1
            # 自环约束样本（状态用执行前）
            hybrid = hybrid_from_dict(action)
            physical = self.decoder.decode(hybrid)
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
            bounds = {"u_tp_low": 1 / 3, "u_tp_high": 1.0, "u_battery_low": -1.0, "u_battery_high": 1.0}
            tr = Transition(
                observation=np.asarray(obs_before, dtype=np.float32),
                hybrid_action={
                    "u_tp": float(hybrid.u_tp),
                    "u_battery": float(hybrid.u_battery),
                    "caes_mode": int(hybrid.caes_mode),
                    "caes_magnitude": float(hybrid.caes_magnitude),
                },
                decoded_fmu_action=physical.as_dict(),
                reward=float(terms["constraint_reward"]),
                next_observation=np.asarray(obs_before, dtype=np.float32),
                terminated=False,
                truncated=True,
                valid_mode_mask=mask,
                dynamic_action_bounds=bounds,
                reward_terms=dict(terms),
                constraint_metadata={"failure": info.get("failure_type"), "fine": info.get("fine_failure_type")},
                physically_valid=False,
                transition_type="givesafe_rejection",
            )
            self.buffer.add_givesafe_rejection(tr)
            if info.get("failure_record") or info.get("actual_fmu_outputs"):
                self.safety_dataset.add_from_failure_record(
                    {
                        **(info.get("failure_record") or {}),
                        "previous_observation": dict(env.last_outputs or {}),
                        "hybrid_action": info.get("hybrid_action"),
                        "label_safe": False,
                    }
                )
            info["stored_in_physical_replay"] = False
            info["stored_in_givesafe_replay"] = True
            info["transition_type"] = "givesafe_false_safe"
            # 不能证明主 FMU 恢复 → episode 已由 env truncated
            assert float(getattr(env.adapter, "time", sim_time_before)) >= sim_time_before
            return obs_before, float(terms["constraint_reward"]), False, True, info

        # 成功物理转移
        assert env.valid_episode_steps == valid_steps_before + 1
        hybrid = hybrid_from_dict(info.get("hybrid_action") or action)
        self.stats["caes_mode_counts"][int(hybrid.caes_mode)] = (
            self.stats["caes_mode_counts"].get(int(hybrid.caes_mode), 0) + 1
        )
        self.stats["physical_transition_count"] += 1
        self.stats["valid_transition_count"] += 1
        physical = {
            "u_tp": float(info["decoded_u_tp"]),
            "u_battery": float(info["decoded_u_battery"]),
            "u_caes": float(info["decoded_u_caes"]),
        }
        if self.shadow is not None:
            self.shadow.on_physical_success(physical)
        bounds = {
            "u_tp_low": float(info.get("u_tp_dynamic_low", 1 / 3)),
            "u_tp_high": float(info.get("u_tp_dynamic_high", 1.0)),
            "u_battery_low": float(info.get("u_battery_dynamic_low", -1.0)),
            "u_battery_high": float(info.get("u_battery_dynamic_high", 1.0)),
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
        terms.setdefault("constraint_reward", 0.0)
        terms.setdefault("total_training_reward", float(reward))
        tr = Transition(
            observation=np.asarray(obs_before, dtype=np.float32),
            hybrid_action={
                "u_tp": float(hybrid.u_tp),
                "u_battery": float(hybrid.u_battery),
                "caes_mode": int(hybrid.caes_mode),
                "caes_magnitude": float(0.0 if hybrid.caes_mode == CaesMode.IDLE else hybrid.caes_magnitude),
            },
            decoded_fmu_action=physical,
            reward=float(reward),
            next_observation=np.asarray(obs, dtype=np.float32),
            terminated=bool(terminated or truncated),
            truncated=bool(truncated),
            valid_mode_mask=mask,
            dynamic_action_bounds=bounds,
            next_valid_mode_mask=next_feasible.mode_mask.as_bool_array(),
            next_dynamic_action_bounds={
                "u_tp_low": next_feasible.u_tp_low, "u_tp_high": next_feasible.u_tp_high,
                "u_battery_low": next_feasible.u_battery_low, "u_battery_high": next_feasible.u_battery_high,
            },
            reward_terms=terms,
            constraint_metadata={},
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
        info["constraint_reward"] = 0.0
        info["economic_reward"] = float(reward)
        info["total_training_reward"] = float(reward)
        return obs, reward, terminated, truncated, info
