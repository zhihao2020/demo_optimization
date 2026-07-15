"""混合动作 Gymnasium 环境：Dict 动作空间 + 动态可行域 + 硬约束预检。

数据流概要：
  策略 HybridAction → Decoder → 物理 {u_tp,u_battery,u_caes}
  →（可选 GiveSafe）→ FmuAdapter.step → 经济 RewardCalculator
  非法/FMU 失败：reward=0、truncated=True，不写入经济 replay。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from gymnasium.spaces import Box, Dict, Discrete

from actions import (
    CaesMode,
    DynamicFeasibleActionSet,
    FeasibilityOracle,
    HybridAction,
    HybridActionDecoder,
    HybridActionValidator,
    PhysicalFmuAction,
)
from actions.failure_taxonomy import classify_failure
from actions.validator import hybrid_from_dict
from envs.failures import (
    ConstraintFailure,
    DynamicStateConstraintViolation,
    FailureRecord,
    FeasibleSetEmpty,
    FmiLifecycleFailure,
    FmuNumericalFailure,
    NonFiniteOutputFailure,
    PostStepHardConstraintViolation,
    StaticActionViolation,
)
from fmu import FmuAdapter, FmuSolverError, build_registry
from .observation_builder import ObservationBuilder
from .forecast_provider import ForecastProvider
from .reward_calculator import RewardCalculator
from .termination_checker import TerminationChecker


class HybridDictSpace(Dict):
    """Dict 混合动作空间。sample() 返回保守合法点，供 check_env/基线使用；不修正策略输出。"""

    def sample(self, mask: Any = None) -> dict:
        # 固定合法：满火电、储能待机。策略/随机探索必须经 FeasibilityOracle。
        return {
            "u_tp": np.asarray([1.0], dtype=np.float32),
            "u_battery": np.asarray([0.0], dtype=np.float32),
            "caes_mode": 1,  # IDLE
            "caes_magnitude": np.asarray([0.0], dtype=np.float32),
        }


class PowerSystemEnv(gym.Env):
    """火电+电池+CAES+风光荷的一小时决策环境。

    - 动作：Dict（连续火电/电池 + 离散 CAES 模式 + 幅值），非凸合法集由 Oracle 表达。
    - 观测：物理输出（固定顺序）+ 可选 24h 日前 forecast。
    - 硬约束拒绝不调用 FMU、不算经济 reward；见 ``docs/RL奖励于成本配置.md``。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config_path: str | Path = "src/config/env_config.yaml",
        reward_config_path: str | Path = "src/config/reward_config.yaml",
        device_params_path: str | Path = "src/config/device_params.yaml",
        margins_path: str | Path = "src/config/feasibility_margins.yaml",
        adapter: Any | None = None,
        require_complete_reward: bool = False,
        run_id: str = "default",
        forecast_enabled: bool | None = None,
    ):
        """加载 YAML、构建 registry/FMU/Oracle；``adapter`` 可注入假对象便于单测。"""
        super().__init__()
        self.root = Path(__file__).resolve().parents[2]
        self.config_path = self._resolve(config_path)
        with self.config_path.open(encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)
        reward_path = self._resolve(reward_config_path)
        with reward_path.open(encoding="utf-8") as stream:
            reward_config = yaml.safe_load(stream)
        reward_config["decision_interval_seconds"] = self.config["fmu"]["decision_interval_seconds"]
        reward_config["episode_steps"] = int(self.config["fmu"]["episode_steps"])
        self.registry = build_registry(self.root / self.config["fmu"]["path"], self.config)
        self.observation_builder = ObservationBuilder(self.registry)
        forecast_cfg = self.config.get("forecast") or {}
        self.forecast_enabled = bool(forecast_cfg.get("enabled", True)) if forecast_enabled is None else bool(forecast_enabled)
        self.forecast_provider: ForecastProvider | None = None
        if self.forecast_enabled:
            self.forecast_provider = ForecastProvider(
                self.root,
                forecast_cfg,
                annual_horizon_hours=int(self.config["fmu"]["annual_horizon_hours"]),
                step_seconds=float(self.config["fmu"]["decision_interval_seconds"]),
            )
        forecast_low = self.forecast_provider.feature_low if self.forecast_provider is not None else np.empty(0, dtype=np.float32)
        forecast_high = self.forecast_provider.feature_high if self.forecast_provider is not None else np.empty(0, dtype=np.float32)
        self.observation_space = Box(
            low=np.concatenate((self.observation_builder.low, forecast_low)),
            high=np.concatenate((self.observation_builder.high, forecast_high)),
            dtype=np.float32,
        )
        self.action_space = HybridDictSpace(
            {
                "u_tp": Box(low=np.array([1.0 / 3.0], dtype=np.float32), high=np.array([1.0], dtype=np.float32), dtype=np.float32),
                "u_battery": Box(low=np.array([-1.0], dtype=np.float32), high=np.array([1.0], dtype=np.float32), dtype=np.float32),
                "caes_mode": Discrete(3),
                "caes_magnitude": Box(low=np.array([0.0], dtype=np.float32), high=np.array([1.0], dtype=np.float32), dtype=np.float32),
            }
        )
        self.decoder = HybridActionDecoder()
        self.hybrid_validator = HybridActionValidator()
        self.oracle = FeasibilityOracle(
            params_path=self._resolve(device_params_path),
            margins_path=self._resolve(margins_path),
        )
        self.adapter = adapter or FmuAdapter(
            self.root / self.config["fmu"]["path"],
            float(self.config["fmu"]["communication_step_seconds"]),
            self.registry,
        )
        self.reward_calculator = RewardCalculator(reward_config, require_complete=require_complete_reward)
        self.termination_checker = TerminationChecker()
        ratio = float(self.config["fmu"]["decision_interval_seconds"]) / float(
            self.config["fmu"]["communication_step_seconds"]
        )
        if ratio <= 0 or not ratio.is_integer():
            raise ValueError("decision_interval_seconds / communication_step_seconds 必须是正整数")
        self.n_substeps = int(ratio)
        self.episode_steps = int(self.config["fmu"]["episode_steps"])
        self.step_index = 0
        self.valid_episode_steps = 0
        self.episode_index = 0
        self.run_id = run_id
        self.last_outputs: dict[str, float] | None = None
        self.previous_thermal = 0.0
        self.initial_soc: dict[str, float] | None = None
        self.episode_failed = False
        self._current_feasible: DynamicFeasibleActionSet | None = None
        self.failure_counts: dict[str, int] = {}
        self.failure_records: list[FailureRecord] = []
        self.last_step_diagnostics: dict[str, Any] = {}
        self._pending_action_meta: dict[str, Any] = {}

    def build_observation(self) -> np.ndarray:
        """当前 FMU 物理输出 + 可选只读日前 forecast 的唯一 observation 构造入口。"""
        if self.last_outputs is None:
            raise RuntimeError("环境未 reset")
        physical = self.observation_builder.build(self.last_outputs)
        if self.forecast_provider is None:
            return physical
        forecast = self.forecast_provider.at_time(float(self.adapter.time))
        return np.concatenate((physical, forecast)).astype(np.float32, copy=False)

    def _resolve(self, path: str | Path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.root / p

    def get_feasible_action_spec(self) -> DynamicFeasibleActionSet:
        if self.last_outputs is None:
            raise RuntimeError("环境未 reset")
        self._current_feasible = self.oracle.compute(self.last_outputs, self.previous_thermal)
        if self.oracle.is_feasible_set_empty(self._current_feasible):
            raise FeasibleSetEmpty("当前状态动态可行集为空")
        return self._current_feasible

    def decode_action(self, action: dict | HybridAction) -> PhysicalFmuAction:
        hybrid = action if isinstance(action, HybridAction) else hybrid_from_dict(action)
        return self.decoder.decode(hybrid)

    def set_action_meta(self, meta: dict[str, Any] | None) -> None:
        """由 SafeActionGenerator 注入 safety/oracle 诊断，供 step 日志使用。"""
        self._pending_action_meta = dict(meta or {})

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        start = float((options or {}).get("start_time", self.config["fmu"].get("start_time_seconds", 0.0)))
        try:
            self.last_outputs = self.adapter.reset(start)
        except FmuSolverError as exc:
            raise FmiLifecycleFailure(str(exc)) from exc
        self.step_index = 0
        self.valid_episode_steps = 0
        self.episode_failed = False
        self.previous_thermal = float(self.last_outputs["p_thermal"])
        self.reward_calculator.reset(self.last_outputs)
        self.initial_soc = {
            k: float(self.last_outputs[k])
            for k in ("battery_soc", "caes_gas_soc", "caes_hot_soc", "caes_cold_soc")
        }
        self._current_feasible = self.oracle.compute(self.last_outputs, self.previous_thermal)
        self._pending_action_meta = {}
        observation = self.build_observation()
        info = {
            "time": self.adapter.time,
            "initial_outputs": dict(self.last_outputs),
            "initial_soc": dict(self.initial_soc),
            "feasible_action_spec": self._current_feasible.as_dict(),
            "oracle_version": self.oracle.oracle_version,
            "episode": self.episode_index,
        }
        self.episode_index += 1
        return observation, info

    def step(self, action: dict | HybridAction):
        """预检 →（合法则）推进 FMU → 经济 reward；失败返回 reward=0 且 truncated。"""
        if self.last_outputs is None:
            raise RuntimeError("必须先 reset")
        action_meta = dict(self._pending_action_meta)
        self._pending_action_meta = {}
        feasible = self.oracle.compute(self.last_outputs, self.previous_thermal)
        self._current_feasible = feasible
        if self.oracle.is_feasible_set_empty(feasible):
            exc = FeasibleSetEmpty("当前状态动态可行集为空")
            self._count(exc.failure_type)
            self.episode_failed = True
            info = self._reject_info(None, feasible, exc, action_meta=action_meta)
            self._record_failure(info, hybrid=None, physical=None, actual=None, predicted=None)
            return self.build_observation(), 0.0, False, True, info

        hybrid: HybridAction | None = None
        try:
            hybrid = action if isinstance(action, HybridAction) else hybrid_from_dict(action)
            self.hybrid_validator.validate(hybrid, feasible)
            ok, reason = self.oracle.check_action_executable(hybrid, self.last_outputs, feasible, self.previous_thermal)
            if not ok:
                raise DynamicStateConstraintViolation(reason or "预检失败")
        except (ConstraintFailure, ValueError, TypeError, KeyError) as exc:
            if not isinstance(exc, ConstraintFailure):
                exc = StaticActionViolation(str(exc))
            self._count(exc.failure_type)
            info = self._reject_info(hybrid if hybrid is not None else action, feasible, exc, action_meta=action_meta)
            # 不调用 FMU、不算经济 reward、不产生有效转移
            obs = self.build_observation()
            return obs, 0.0, False, True, info

        physical = self.decoder.decode(hybrid)
        mag_logged = 0.0 if hybrid.caes_mode == CaesMode.IDLE else float(hybrid.caes_magnitude)
        predicted = self.oracle.predict_next_state(self.last_outputs, hybrid, self.previous_thermal)
        physical_dist, safe_dist = self.oracle.distances_to_bounds(self.last_outputs)
        outputs: dict[str, float] | None = None
        try:
            for _ in range(self.n_substeps):
                outputs = self.adapter.step(physical.as_dict())
            assert outputs is not None
            if any(not np.isfinite(float(v)) for v in outputs.values()):
                raise NonFiniteOutputFailure(
                    f"输出含 NaN/Inf: {outputs}",
                    fine_type="nonfinite_output",
                    triggering_constraint="nonfinite_output",
                )
            post_ok, post_reason = self.oracle.post_step_hard_ok(outputs)
            if not post_ok:
                fine, trig = classify_failure(
                    failure_type="PostStepHardConstraintViolation",
                    reason=post_reason,
                    outputs=outputs,
                    params=self.oracle.params,
                )
                raise PostStepHardConstraintViolation(
                    post_reason or "后验硬约束违反",
                    fine_type=fine,
                    triggering_constraint=trig,
                )
        except ConstraintFailure as exc:
            self._count(exc.failure_type)
            self.episode_failed = True
            info = self._failure_info(
                hybrid, physical, mag_logged, feasible, exc, applied=None,
                predicted=predicted, actual=outputs, action_meta=action_meta,
                physical_dist=physical_dist, safe_dist=safe_dist,
            )
            self._record_failure(info, hybrid=hybrid, physical=physical, actual=outputs, predicted=predicted)
            return self.build_observation(), 0.0, False, True, info
        except FmuSolverError as exc:
            # 分类：生命周期 vs 数值
            msg = str(exc).lower()
            if "reset" in msg or "instantiate" in msg or "lifecycle" in msg:
                fail = FmiLifecycleFailure(str(exc), fine_type="nonlinear_solver_failure")
            else:
                fail = FmuNumericalFailure(str(exc), fine_type="nonlinear_solver_failure")
            self._count(fail.failure_type)
            self.episode_failed = True
            info = self._failure_info(
                hybrid, physical, mag_logged, feasible, fail, applied=None,
                predicted=predicted, actual=outputs, action_meta=action_meta,
                physical_dist=physical_dist, safe_dist=safe_dist,
            )
            self._record_failure(info, hybrid=hybrid, physical=physical, actual=outputs, predicted=predicted)
            return self.build_observation(), 0.0, False, True, info

        residuals = self.oracle.residual(predicted, outputs)
        dang = self.oracle.dangerous_residual(
            residuals, mode=hybrid.caes_mode, u_battery=hybrid.u_battery
        )
        next_step = self.step_index + 1
        terminated, term_reason = self.termination_checker.terminated(outputs)
        truncated = next_step >= self.episode_steps
        is_final = truncated or terminated
        episode_completed = truncated and not self.episode_failed
        self.valid_episode_steps += 1
        reward, terms = self.reward_calculator.calculate(
            outputs,
            self.previous_thermal,
            is_final_step=is_final,
            episode_completed=episode_completed and is_final,
            no_failure=not self.episode_failed,
            valid_episode_steps=self.valid_episode_steps,
        )
        self.reward_calculator.step_in_episode = self.valid_episode_steps
        self.step_index = next_step
        self.previous_thermal = float(outputs["p_thermal"])
        self.last_outputs = outputs
        self._current_feasible = self.oracle.compute(outputs, self.previous_thermal)
        observation = self.build_observation()
        info = {
            "time": self.adapter.time,
            "step": self.step_index,
            "episode": self.episode_index - 1,
            "requested_u_tp": hybrid.u_tp,
            "requested_u_battery": hybrid.u_battery,
            "requested_caes_mode": int(hybrid.caes_mode),
            "requested_caes_magnitude": mag_logged,
            "decoded_u_tp": physical.u_tp,
            "decoded_u_battery": physical.u_battery,
            "decoded_u_caes": physical.u_caes,
            "applied_action": physical.as_dict(),
            "hybrid_action": {
                "u_tp": hybrid.u_tp,
                "u_battery": hybrid.u_battery,
                "caes_mode": int(hybrid.caes_mode),
                "caes_magnitude": mag_logged,
            },
            "reward_terms": terms,
            "fmu_status": "ok",
            "termination_reason": term_reason,
            "transition_valid": True,
            "physically_valid": True,
            "failure_type": None,
            "fine_failure_type": None,
            "failure_reason": None,
            "stored_in_replay": True,
            "oracle_version": self.oracle.oracle_version,
            "oracle_predicted_next_state": {k: float(predicted[k]) for k in predicted if k not in ("caes_mode", "caes_magnitude")},
            "residuals": residuals,
            "dangerous_residual": dang,
            "distance_to_physical_boundary": physical_dist,
            "distance_to_safe_boundary": safe_dist,
            "safety_probability": action_meta.get("safety_probability"),
            "safety_threshold": action_meta.get("safety_threshold"),
            "safety_model_version": action_meta.get("safety_model_version"),
            **feasible.as_dict(),
            "observations": dict(outputs),
            "initial_soc": dict(self.initial_soc) if self.initial_soc else None,
        }
        self.last_step_diagnostics = info
        if is_final and self.initial_soc:
            info.update(self._episode_summary(outputs, terms, episode_completed))
        return observation, float(reward), terminated, truncated, info

    def step_physical_for_test(self, action: dict[str, float] | PhysicalFmuAction | np.ndarray):
        """仅用于单元测试与规则辅助；不是正式策略接口。"""
        if isinstance(action, PhysicalFmuAction):
            physical = action
        elif isinstance(action, np.ndarray):
            physical = PhysicalFmuAction(float(action[0]), float(action[1]), float(action[2]))
        else:
            physical = PhysicalFmuAction(float(action["u_tp"]), float(action["u_battery"]), float(action["u_caes"]))
        self.hybrid_validator.validate_physical_static(physical)
        # 将物理动作反解为最近 HybridAction 再走正式 step 路径
        hybrid = self._physical_to_hybrid(physical)
        return self.step(hybrid)

    def _physical_to_hybrid(self, physical: PhysicalFmuAction) -> HybridAction:
        u = physical.u_caes
        if abs(u) <= 1e-9:
            mode, mag = CaesMode.IDLE, 0.0
        elif u < 0:
            mode = CaesMode.DISCHARGE
            mag = (u - (-1.0)) / (-0.33 - (-1.0))
        else:
            mode = CaesMode.CHARGE
            mag = (u - 0.86) / (1.0 - 0.86)
        mag = float(np.clip(mag, 0.0, 1.0))
        return HybridAction(physical.u_tp, physical.u_battery, mode, mag)

    def _episode_summary(self, outputs: dict[str, float], terms: dict[str, float], completed: bool) -> dict[str, Any]:
        assert self.initial_soc is not None
        summary: dict[str, Any] = {
            "episode_period_hours": self.episode_steps,
            "episode_valid_steps": self.valid_episode_steps,
            "episode_completed": completed,
        }
        for key in ("battery_soc", "caes_gas_soc", "caes_hot_soc", "caes_cold_soc"):
            short = key.replace("_soc", "")
            summary[f"initial_{key}"] = self.initial_soc[key]
            summary[f"final_{key}"] = float(outputs[key])
            summary[f"{key}_delta"] = float(outputs[key]) - self.initial_soc[key]
        summary["terminal_soc_bonus"] = terms.get("terminal_soc_bonus", 0.0)
        summary["terminal_soc_l1_error"] = terms.get("terminal_soc_l1_error", 0.0)
        summary["terminal_soc_l2_error"] = terms.get("terminal_soc_l2_error", 0.0)
        summary["terminal_soc_satisfied"] = terms.get("terminal_soc_satisfied", 0.0)
        return summary

    def _reject_info(
        self,
        action: Any,
        feasible: DynamicFeasibleActionSet,
        exc: ConstraintFailure,
        action_meta: dict | None = None,
    ) -> dict[str, Any]:
        hybrid = action if isinstance(action, HybridAction) else None
        fine, trig = classify_failure(
            failure_type=exc.failure_type,
            reason=exc.reason,
            params=self.oracle.params,
        )
        if getattr(exc, "fine_type", None) and exc.fine_type != "unknown":
            fine = exc.fine_type
            trig = exc.triggering_constraint or fine
        meta = action_meta or {}
        physical_dist, safe_dist = (
            self.oracle.distances_to_bounds(self.last_outputs) if self.last_outputs else ({}, {})
        )
        return {
            "time": self.adapter.time,
            "step": self.step_index,
            "episode": self.episode_index - 1,
            "transition_valid": False,
            "physically_valid": False,
            "failure_type": exc.failure_type,
            "fine_failure_type": fine,
            "triggering_constraint": trig,
            "failure_reason": exc.reason,
            "stored_in_replay": False,
            "fmu_status": "not_called",
            "reward_terms": {},
            "feasible_action_spec": feasible.as_dict(),
            "oracle_version": self.oracle.oracle_version,
            "distance_to_physical_boundary": physical_dist,
            "distance_to_safe_boundary": safe_dist,
            "safety_probability": meta.get("safety_probability"),
            "safety_threshold": meta.get("safety_threshold"),
            "safety_model_version": meta.get("safety_model_version"),
            "last_valid_outputs": dict(self.last_outputs) if self.last_outputs else None,
            **feasible.as_dict(),
            "requested_caes_mode": int(hybrid.caes_mode) if hybrid else None,
            "requested_u_tp": hybrid.u_tp if hybrid else None,
            "requested_u_battery": hybrid.u_battery if hybrid else None,
            "requested_caes_magnitude": hybrid.caes_magnitude if hybrid else None,
        }

    def _failure_info(
        self,
        hybrid: HybridAction,
        physical: PhysicalFmuAction,
        mag_logged: float,
        feasible: DynamicFeasibleActionSet,
        exc: ConstraintFailure,
        applied: dict | None,
        *,
        predicted: dict[str, float] | None = None,
        actual: dict[str, float] | None = None,
        action_meta: dict | None = None,
        physical_dist: dict | None = None,
        safe_dist: dict | None = None,
    ) -> dict[str, Any]:
        fine = getattr(exc, "fine_type", None) or "unknown"
        trig = getattr(exc, "triggering_constraint", None) or fine
        if fine == "unknown":
            fine, trig = classify_failure(
                failure_type=exc.failure_type,
                reason=exc.reason,
                outputs=actual,
                params=self.oracle.params,
            )
        residuals = None
        dang = None
        if predicted is not None and actual is not None:
            residuals = self.oracle.residual(predicted, actual)
            dang = self.oracle.dangerous_residual(residuals, mode=hybrid.caes_mode, u_battery=hybrid.u_battery)
        meta = action_meta or {}
        return {
            "time": self.adapter.time,
            "step": self.step_index,
            "episode": self.episode_index - 1,
            "requested_u_tp": hybrid.u_tp,
            "requested_u_battery": hybrid.u_battery,
            "requested_caes_mode": int(hybrid.caes_mode),
            "requested_caes_magnitude": mag_logged,
            "decoded_u_tp": physical.u_tp,
            "decoded_u_battery": physical.u_battery,
            "decoded_u_caes": physical.u_caes,
            "hybrid_action": {
                "u_tp": hybrid.u_tp,
                "u_battery": hybrid.u_battery,
                "caes_mode": int(hybrid.caes_mode),
                "caes_magnitude": mag_logged,
            },
            "decoded_fmu_action": physical.as_dict(),
            "applied_action": applied,
            "transition_valid": False,
            "physically_valid": False,
            "failure_type": exc.failure_type,
            "fine_failure_type": fine,
            "triggering_constraint": trig,
            "failure_reason": exc.reason,
            "stored_in_replay": False,
            "fmu_status": "failure",
            "fmu_error": exc.reason,
            "modelica_assert_message": exc.reason,
            "reward_terms": {},
            "last_valid_outputs": dict(self.last_outputs) if self.last_outputs else None,
            "oracle_predicted_next_state": predicted,
            "actual_fmu_outputs": actual,
            "residuals": residuals,
            "dangerous_residual": dang,
            "distance_to_physical_boundary": physical_dist,
            "distance_to_safe_boundary": safe_dist,
            "oracle_version": self.oracle.oracle_version,
            "safety_probability": meta.get("safety_probability"),
            "safety_threshold": meta.get("safety_threshold"),
            "safety_model_version": meta.get("safety_model_version"),
            **feasible.as_dict(),
        }

    def _record_failure(
        self,
        info: dict[str, Any],
        *,
        hybrid: HybridAction | None,
        physical: PhysicalFmuAction | None,
        actual: dict | None,
        predicted: dict | None,
    ) -> FailureRecord:
        rec = FailureRecord(
            run_id=self.run_id,
            episode=int(info.get("episode") or 0),
            step=int(info.get("step") or 0),
            simulation_time=float(info.get("time") or 0.0),
            failure_type=str(info.get("failure_type") or "unknown"),
            fine_failure_type=str(info.get("fine_failure_type") or "unknown"),
            triggering_constraint=str(info.get("triggering_constraint") or "unknown"),
            previous_observation=dict(self.last_outputs) if self.last_outputs else None,
            hybrid_action=info.get("hybrid_action")
            or (
                {
                    "u_tp": hybrid.u_tp,
                    "u_battery": hybrid.u_battery,
                    "caes_mode": int(hybrid.caes_mode),
                    "caes_magnitude": hybrid.caes_magnitude,
                }
                if hybrid
                else None
            ),
            decoded_fmu_action=physical.as_dict() if physical else info.get("decoded_fmu_action"),
            oracle_dynamic_bounds={
                "u_tp_low": float(info.get("u_tp_dynamic_low", 0.0) or 0.0),
                "u_tp_high": float(info.get("u_tp_dynamic_high", 0.0) or 0.0),
                "u_battery_low": float(info.get("u_battery_dynamic_low", 0.0) or 0.0),
                "u_battery_high": float(info.get("u_battery_dynamic_high", 0.0) or 0.0),
            },
            oracle_mode_mask={
                "discharge": bool(info.get("caes_discharge_allowed")),
                "idle": bool(info.get("caes_idle_allowed")),
                "charge": bool(info.get("caes_charge_allowed")),
            },
            oracle_predicted_next_state=predicted or info.get("oracle_predicted_next_state"),
            actual_fmu_outputs=actual or info.get("actual_fmu_outputs"),
            last_valid_state=dict(self.last_outputs) if self.last_outputs else None,
            distance_to_physical_boundary=info.get("distance_to_physical_boundary"),
            distance_to_safe_boundary=info.get("distance_to_safe_boundary"),
            residuals=info.get("residuals"),
            dangerous_residual=info.get("dangerous_residual"),
            fmu_status=info.get("fmu_status"),
            modelica_assert_message=info.get("modelica_assert_message") or info.get("failure_reason"),
            oracle_version=self.oracle.oracle_version,
            safety_probability=info.get("safety_probability"),
            safety_threshold=info.get("safety_threshold"),
            safety_model_version=info.get("safety_model_version"),
        )
        self.failure_records.append(rec)
        info["failure_record"] = rec.to_dict()
        return rec

    def _count(self, failure_type: str) -> None:
        self.failure_counts[failure_type] = self.failure_counts.get(failure_type, 0) + 1

    def close(self) -> None:
        self.adapter.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self.close()
