"""策略评估：支持 Hybrid Dict 动作与审计日志。"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from dataclasses import asdict

from envs.failures import FeasibleSetEmpty
from safety import NoSafeActionFoundError
from safety.soft_constraint_shell import SoftConstraintEnv, SoftConstraintShell


def _jsonify(obj):
    if obj is None:
        return None
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _jsonify(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (float, int, str, bool)):
        return obj
    return str(obj)


def _grid_contract_p_lim_w(env) -> float:
    rc = getattr(env, "reward_calculator", None)
    cfg = getattr(rc, "config", None) or {}
    gc = cfg.get("grid_contract") or {}
    return float(gc.get("p_lim_w", 2.0e8))


def _finalize_metric_rates(metrics: dict[str, float]) -> None:
    """Derive rates / peak-valley stats after the episode loop."""
    res_avail = float(metrics.get("renewable_available_mwh", 0.0))
    curt = float(metrics.get("curtailment_energy_mwh", 0.0))
    wind_a = float(metrics.get("wind_available_mwh", 0.0))
    pv_a = float(metrics.get("pv_available_mwh", 0.0))
    wind_u = float(metrics.get("wind_actual_mwh", 0.0))
    pv_u = float(metrics.get("pv_actual_mwh", 0.0))
    metrics["curtailment_rate"] = float(curt / max(res_avail, 1e-9))
    metrics["renewable_utilization"] = float((wind_u + pv_u) / max(res_avail, 1e-9))
    metrics["wind_utilization"] = float(wind_u / max(wind_a, 1e-9))
    metrics["pv_utilization"] = float(pv_u / max(pv_a, 1e-9))
    g_max = float(metrics.get("grid_export_max_mw", 0.0))
    g_min = float(metrics.get("grid_import_max_mw", 0.0))  # stored as positive import
    # peak-to-valley of signed P_grid: max(export) - min(import as negative) = g_max + g_min
    metrics["grid_peak_valley_mw"] = float(g_max + g_min)
    times = metrics.pop("_decision_times_s", None)
    if isinstance(times, list) and times:
        arr = np.asarray(times, dtype=np.float64)
        metrics["decision_time_mean_s"] = float(arr.mean())
        metrics["decision_time_p95_s"] = float(np.quantile(arr, 0.95))
        metrics["decision_time_max_s"] = float(arr.max())
        metrics["decision_time_sum_s"] = float(arr.sum())
    else:
        metrics.setdefault("decision_time_mean_s", 0.0)
        metrics.setdefault("decision_time_p95_s", 0.0)
        metrics.setdefault("decision_time_max_s", 0.0)
        metrics.setdefault("decision_time_sum_s", 0.0)


def evaluate_policy(
    env,
    policy: Any,
    output_csv: Path | None = None,
    gamma: float = 0.99,
    *,
    reset_options: dict[str, Any] | None = None,
    max_steps: int | None = None,
    soft_shell: bool = False,
    deterministic: bool = True,
) -> dict[str, Any]:
    """评估单个时间窗口内的策略表现。

    Args:
        env: 电力系统环境(PowerSystemEnv) 实例。
        policy: 需实现 ``predict(obs, deterministic=...)`` 的策略对象。
        output_csv: 可选逐步轨迹 CSV 路径。
        gamma: 折扣因子，用于计算折扣回报。
        reset_options: 传给 ``env.reset(options=...)`` 的选项，如 ``start_time``。
        max_steps: 最大步数；用于年度尾窗不足一周时截断。
        soft_shell: 为 True 时包装预检重试，并在 ``predict`` 抛 NoSafeAction 时用保守动作。
        deterministic: 传给 ``policy.predict``；主表 greedy 为 True。

    Returns:
        含步数、奖励、成本分项、SOC、CAES 合规率等字段的字典。

    Raises:
        ValueError: ``max_steps`` 非正时抛出。
    """
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps 必须为正数")
    shell = SoftConstraintShell() if soft_shell else None
    step_env = SoftConstraintEnv(env, shell) if soft_shell else env
    obs, info0 = step_env.reset(options=reset_options)
    if hasattr(policy, "on_episode_reset"):
        policy.on_episode_reset(info0)
    rows: list[dict[str, Any]] = []
    totals: dict[str, float] = {}
    metrics: dict[str, Any] = {
        "curtailment_energy_mwh": 0.0,
        "unserved_energy_mwh": 0.0,
        "battery_throughput_mwh": 0.0,
        "caes_throughput_mwh": 0.0,
        "thermal_generation_mwh": 0.0,
        "max_thermal_ramp_mw": 0.0,
        "wind_available_mwh": 0.0,
        "pv_available_mwh": 0.0,
        "wind_actual_mwh": 0.0,
        "pv_actual_mwh": 0.0,
        "renewable_available_mwh": 0.0,
        "grid_export_max_mw": 0.0,
        "grid_import_max_mw": 0.0,
        "grid_abs_max_mw": 0.0,
        "max_grid_ramp_mw": 0.0,
        "grid_contract_excess_mwh": 0.0,
        "grid_contract_violation_hours": 0.0,
        "solver_timeout_count": 0.0,
        "solver_fail_count": 0.0,
        "caes_hours_nonzero": 0.0,
        "caes_hours_discharge": 0.0,
        "caes_hours_idle": 0.0,
        "caes_hours_charge": 0.0,
        "caes_mode_switches": 0.0,
        "caes_gas_soc_min": None,
        "caes_gas_soc_max": None,
        "_decision_times_s": [],
    }
    previous_thermal: float | None = None
    previous_grid: float | None = None
    previous_caes_mode: int | None = None
    p_lim_w = _grid_contract_p_lim_w(env)
    weekly_raw = 0.0
    weekly_reward = 0.0
    weekly_discounted = 0.0
    terminal_bonus = 0.0
    forbidden = 0
    invalid_transition = 0
    soft_shell_count = 0
    caes_segments: list[dict[str, Any]] = []
    caes_interruptions = 0
    while True:
        t_dec0 = time.perf_counter()
        try:
            predicted = policy.predict(obs, deterministic=deterministic)
            action = predicted[0] if isinstance(predicted, tuple) else predicted
        except (NoSafeActionFoundError, FeasibleSetEmpty) as exc:
            if soft_shell and shell is not None:
                action = shell.recover(env if not isinstance(step_env, SoftConstraintEnv) else step_env.env)
                soft_shell_count += 1
            else:
                failure = {
                    "eval_status": "failed_no_safe_action",
                    "failed_step": len(rows),
                    "completed_steps": len(rows),
                    "failure_type": getattr(exc, "failure_type", type(exc).__name__),
                    "reason": str(exc),
                    "attempts": int(getattr(exc, "attempts", 0) or 0),
                    "rejection_reasons": list(getattr(exc, "reasons", []) or []),
                    "first_check": _jsonify(getattr(exc, "first_check", None)),
                    "policy_action": _jsonify((getattr(exc, "rejected", None) or [None])[0]),
                    "last_outputs": _jsonify(getattr(env, "last_outputs", None)),
                    "previous_thermal": getattr(env, "previous_thermal", None),
                }
                spec = None
                if hasattr(env, "get_feasible_action_spec"):
                    try:
                        fs = env.get_feasible_action_spec()
                        spec = fs.as_dict() if hasattr(fs, "as_dict") else _jsonify(fs)
                    except Exception:
                        spec = None
                failure["feasible_action_spec"] = spec
                rejected = getattr(exc, "rejected", None)
                first_rejected = None
                if isinstance(rejected, (list, tuple)) and rejected:
                    first_rejected = rejected[0]
                elif isinstance(rejected, dict):
                    first_rejected = rejected
                failure["raw_policy_action"] = _jsonify(first_rejected)
                decoded = None
                if isinstance(first_rejected, dict):
                    decoded = {
                        "u_tp": first_rejected.get("u_tp"),
                        "u_battery": first_rejected.get("u_battery"),
                        "u_caes": first_rejected.get("u_caes"),
                        "caes_mode": first_rejected.get("caes_mode", first_rejected.get("mode")),
                        "caes_magnitude": first_rejected.get(
                            "caes_magnitude", first_rejected.get("mag")
                        ),
                    }
                failure["decoded_physical_action"] = decoded
                failure["caes_mode"] = None if decoded is None else decoded.get("caes_mode")
                failure["caes_magnitude"] = None if decoded is None else decoded.get("caes_magnitude")
                failure["time"] = rows[-1].get("time") if rows else info0.get("time")
                failure["state"] = _jsonify(obs)
                if isinstance(spec, dict):
                    failure["device_bounds"] = {
                        "u_tp": (spec.get("u_tp_dynamic_low"), spec.get("u_tp_dynamic_high")),
                        "u_battery": (
                            spec.get("u_battery_dynamic_low"),
                            spec.get("u_battery_dynamic_high"),
                        ),
                    }
                    failure["joint_caes_support"] = {
                        "discharge": spec.get("joint_caes_discharge"),
                        "charge": spec.get("joint_caes_charge"),
                        "mode_mask": spec.get("joint_mode_mask"),
                    }
                    failure["conditional_thermal_support"] = failure["device_bounds"]["u_tp"]
                    failure["conditional_battery_support"] = failure["device_bounds"]["u_battery"]
                    failure["grid_limits"] = {
                        "g_min_W": spec.get("grid_low_w") or spec.get("grid_g_min_W"),
                        "g_max_W": spec.get("grid_high_w") or spec.get("grid_g_max_W"),
                        "residual_W": spec.get("grid_residual_w") or spec.get("grid_residual_W"),
                    }
                pred_grid = None
                first = getattr(exc, "first_check", None)
                if first is not None:
                    failure["oracle_rejection_stage"] = getattr(first, "rejection_stage", None)
                    failure["rejection_stage"] = getattr(first, "rejection_stage", None)
                    failure["violation_type"] = getattr(first, "violation_type", None)
                    failure["oracle_rejection_reason"] = getattr(first, "oracle_rejection_reason", None)
                    failure["shadow_rejection_reason"] = getattr(first, "shadow_failure_reason", None)
                    failure["shadow_failure_reason"] = getattr(first, "shadow_failure_reason", None)
                    pred = getattr(first, "predicted_next_state", None) or {}
                    pred_grid = pred.get("p_grid")
                failure["predicted_p_grid"] = pred_grid
                if output_csv and rows:
                    output_csv.parent.mkdir(parents=True, exist_ok=True)
                    fieldnames = []
                    seen: set[str] = set()
                    for row in rows:
                        for key in row:
                            if key not in seen:
                                seen.add(key)
                                fieldnames.append(key)
                    with output_csv.open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                        writer.writeheader()
                        writer.writerows(rows)
                out = {
                    "steps": len(rows),
                    "valid_steps": sum(1 for r in rows if r.get("transition_valid")),
                    "eval_status": "failed_no_safe_action",
                    "eval_failed": True,
                    "failure": failure,
                    "episode_reward": weekly_reward,
                    "metrics": metrics,
                    "cost_terms": totals,
                    "fmu_failure_count": sum(1 for r in rows if r.get("fmu_status") == "failure"),
                    "forbidden_action_count": forbidden,
                    "invalid_transition_count": invalid_transition,
                }
                _finalize_metric_rates(metrics)
                return out
        # Prefer controller-reported solve time when present (MILP/linprog).
        solve_s = getattr(policy, "last_solve_s", None)
        if solve_s is None and hasattr(policy, "agent"):
            solve_s = getattr(policy.agent, "last_solve_s", None)
        if solve_s is None and hasattr(policy, "ctrl"):
            solve_s = getattr(policy.ctrl, "last_solve_s", None)
        decision_s = float(solve_s) if solve_s is not None else float(time.perf_counter() - t_dec0)
        metrics["_decision_times_s"].append(decision_s)
        if getattr(policy, "last_solve_timed_out", False) or getattr(
            getattr(policy, "agent", None), "last_solve_timed_out", False
        ):
            metrics["solver_timeout_count"] += 1.0
        if getattr(policy, "last_solve_failed", False) or getattr(
            getattr(policy, "agent", None), "last_solve_failed", False
        ):
            metrics["solver_fail_count"] += 1.0
        obs, reward, terminated, truncated, info = step_env.step(action)
        if info.get("soft_shell_applied"):
            soft_shell_count = max(
                soft_shell_count, int(info.get("soft_shell_count") or soft_shell_count + 1)
            )
        if hasattr(policy, "on_transition"):
            policy.on_transition(info)
        if info.get("failure_type") in ("StaticActionViolation", "ForbiddenModeViolation", "DynamicStateConstraintViolation"):
            forbidden += 1
        if not info.get("transition_valid", False):
            invalid_transition += 1
        terms = info.get("reward_terms") or {}
        weekly_raw += float(terms.get("raw_total_cost", 0.0))
        weekly_reward += float(reward)
        weekly_discounted += (gamma ** len(rows)) * float(reward)
        terminal_bonus += float(terms.get("terminal_soc_bonus", 0.0))
        if info.get("caes_min_run_completed_segment"):
            caes_segments.append(dict(info["caes_min_run_completed_segment"]))
        if info.get("caes_min_run_final_event"):
            caes_segments.append(dict(info["caes_min_run_final_event"]))
            caes_interruptions += 1
        event = (info.get("feasible_action_spec") or {}).get("caes_min_run_event")
        if event:
            caes_segments.append(dict(event))
            caes_interruptions += 1
        row = {
            "time": info.get("time"),
            "step": len(rows),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "fmu_status": info.get("fmu_status"),
            "fmu_error": info.get("fmu_error") or info.get("failure_reason"),
            "failure_type": info.get("failure_type"),
            "transition_valid": info.get("transition_valid"),
            "requested_u_tp": info.get("requested_u_tp"),
            "requested_u_battery": info.get("requested_u_battery"),
            "requested_caes_mode": info.get("requested_caes_mode"),
            "requested_u_caes": info.get("requested_u_caes"),
            "decoded_u_tp": info.get("decoded_u_tp"),
            "decoded_u_battery": info.get("decoded_u_battery"),
            "decoded_u_caes": info.get("decoded_u_caes"),
            **{f"rt_{k}": v for k, v in terms.items()},
            **{f"obs_{spec.name}": float(obs[i]) for i, spec in enumerate(env.observation_builder.specs)},
            **{k: info.get(k) for k in (
                "u_tp_dynamic_low", "u_tp_dynamic_high",
                "u_battery_dynamic_low", "u_battery_dynamic_high",
                "caes_discharge_allowed", "caes_idle_allowed", "caes_charge_allowed",
                "caes_locked_mode", "caes_locked_steps_completed", "caes_locked_steps_remaining",
            )},
        }
        rows.append(row)
        for key, value in terms.items():
            if key in (
                "cost_reference",
                "cost_reference_missing",
                "terminal_soc_tolerance",
                "terminal_soc_l1_error",
                "terminal_soc_l2_error",
                "terminal_soc_satisfied",
                "terminal_soc_bonus",
            ):
                totals[key] = float(value)  # 取最近一步（最终步覆盖）
            else:
                totals[key] = totals.get(key, 0.0) + float(value)
        current = info.get("last_valid_outputs") or env.last_outputs or {}
        dt_hours = env.config["fmu"]["decision_interval_seconds"] / 3600.0
        if current:
            metrics["curtailment_energy_mwh"] += float(current.get("p_curtailment", 0)) * 1e-6 * dt_hours
            metrics["unserved_energy_mwh"] += float(current.get("p_unserved", 0)) * 1e-6 * dt_hours
            metrics["battery_throughput_mwh"] += abs(float(current.get("p_battery", 0))) * 1e-6 * dt_hours
            metrics["caes_throughput_mwh"] += abs(float(current.get("p_caes", 0))) * 1e-6 * dt_hours
            if abs(float(current.get("p_caes", 0))) > 1e6:
                metrics["caes_hours_nonzero"] += 1.0
            mode = info.get("requested_caes_mode")
            try:
                mode_i = int(mode)
            except (TypeError, ValueError):
                mode_i = 1
            if previous_caes_mode is not None and mode_i != previous_caes_mode:
                metrics["caes_mode_switches"] += 1.0
            previous_caes_mode = mode_i
            if mode_i == 0:
                metrics["caes_hours_discharge"] += 1.0
            elif mode_i == 2:
                metrics["caes_hours_charge"] += 1.0
            else:
                metrics["caes_hours_idle"] += 1.0
            soc_g = current.get("caes_gas_soc")
            if soc_g is not None:
                sg = float(soc_g)
                lo = metrics["caes_gas_soc_min"]
                hi = metrics["caes_gas_soc_max"]
                metrics["caes_gas_soc_min"] = sg if lo is None else min(float(lo), sg)
                metrics["caes_gas_soc_max"] = sg if hi is None else max(float(hi), sg)
            metrics["thermal_generation_mwh"] += abs(float(current.get("p_thermal", 0))) * 1e-6 * dt_hours
            # Generation channels are negative watts in the FMU; report positive MWh.
            w_av = abs(float(current.get("p_wind_available", 0))) * 1e-6 * dt_hours
            p_av = abs(float(current.get("p_pv_available", 0))) * 1e-6 * dt_hours
            w_ac = abs(float(current.get("p_wind_actual", 0))) * 1e-6 * dt_hours
            p_ac = abs(float(current.get("p_pv_actual", 0))) * 1e-6 * dt_hours
            metrics["wind_available_mwh"] += w_av
            metrics["pv_available_mwh"] += p_av
            metrics["wind_actual_mwh"] += w_ac
            metrics["pv_actual_mwh"] += p_ac
            metrics["renewable_available_mwh"] += w_av + p_av
            p_grid = float(current.get("p_grid", 0.0))
            p_grid_mw = p_grid * 1e-6
            # Convention: p_grid > 0 buy/import, p_grid < 0 sell/export.
            if p_grid_mw > 0:
                metrics["grid_import_max_mw"] = max(float(metrics["grid_import_max_mw"]), p_grid_mw)
            else:
                metrics["grid_export_max_mw"] = max(float(metrics["grid_export_max_mw"]), -p_grid_mw)
            metrics["grid_abs_max_mw"] = max(float(metrics["grid_abs_max_mw"]), abs(p_grid_mw))
            if abs(p_grid) > p_lim_w + 1.0:
                metrics["grid_contract_violation_hours"] += dt_hours
                metrics["grid_contract_excess_mwh"] += (abs(p_grid) - p_lim_w) * 1e-6 * dt_hours
            if previous_thermal is not None:
                metrics["max_thermal_ramp_mw"] = max(
                    metrics["max_thermal_ramp_mw"],
                    abs(float(current["p_thermal"]) - previous_thermal) * 1e-6,
                )
            if previous_grid is not None:
                metrics["max_grid_ramp_mw"] = max(
                    float(metrics["max_grid_ramp_mw"]),
                    abs(p_grid - previous_grid) * 1e-6,
                )
            previous_thermal = float(current["p_thermal"])
            previous_grid = p_grid
        if terminated or truncated:
            break
        if max_steps is not None and len(rows) >= max_steps:
            break
    if output_csv and rows:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        # Soft-shell rows may add constraint_reward keys mid-episode; union all fields.
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    last = env.last_outputs or {}
    if shell is not None:
        soft_shell_count = max(soft_shell_count, int(shell.recovery_count))
    _finalize_metric_rates(metrics)
    lo = metrics.get("caes_gas_soc_min")
    hi = metrics.get("caes_gas_soc_max")
    if lo is not None and hi is not None:
        metrics["caes_gas_soc_range"] = float(hi) - float(lo)
    else:
        metrics["caes_gas_soc_range"] = None
    init_soc = info0.get("initial_soc") or {}

    def _soc(block: dict[str, Any] | None, key: str) -> float:
        raw = (block or {}).get(key)
        return float(raw) if raw is not None else float("nan")

    e_terminal = abs(_soc(last, "battery_soc") - _soc(init_soc, "battery_soc")) + abs(
        _soc(last, "caes_gas_soc") - _soc(init_soc, "caes_gas_soc")
    )
    fmu_fail = sum(1 for r in rows if r.get("fmu_status") == "failure")
    eval_status = "ok"
    eval_failed = False
    failure = None
    if fmu_fail > 0:
        eval_status = "failed_fmu"
        eval_failed = True
        fail_row = next((r for r in rows if r.get("fmu_status") == "failure"), {})
        failure = {
            "eval_status": "failed_fmu",
            "failure_type": "FMUFailure",
            "failed_step": fail_row.get("step"),
            "completed_steps": len(rows),
            "fmu_error": fail_row.get("fmu_error"),
            "time": fail_row.get("time"),
        }
    return {
        "steps": len(rows),
        "valid_steps": sum(1 for r in rows if r.get("transition_valid")),
        "eval_status": eval_status,
        "eval_failed": eval_failed,
        "failure": failure,
        "soft_shell": bool(soft_shell),
        "soft_shell_count": int(soft_shell_count),
        "soft_shell_hours": int(soft_shell_count),
        "episode_reward": weekly_reward,
        "weekly_episode_reward": weekly_reward,
        "weekly_discounted_return": weekly_discounted,
        "weekly_raw_total_cost": weekly_raw,
        "weekly_terminal_soc_bonus": float(totals.get("terminal_soc_bonus", terminal_bonus)),
        "terminal_soc_satisfied": bool(float((rows[-1].get("rt_terminal_soc_satisfied") if rows else 0) or 0)),
        "cost_terms": totals,
        "metrics": metrics,
        "terminal_soc": {name: float(last.get(name, float("nan"))) for name in ("battery_soc", "caes_gas_soc", "caes_hot_soc", "caes_cold_soc")},
        "initial_soc": info0.get("initial_soc"),
        "e_terminal": float(e_terminal),
        "fmu_failure_count": fmu_fail,
        "forbidden_action_count": forbidden,
        "invalid_transition_count": invalid_transition,
        "action_violation_count": forbidden,
        "economic_cashflow_total": float(last.get("economic_cashflow_total", float("nan"))),
        "economic_cashflow_components": {
            name: float(last.get(f"economic_cashflow_{name}", float("nan")))
            for name in ("wind", "pv", "thermal", "battery", "caes", "load", "grid")
        },
        "caes_run_segments": caes_segments,
        "caes_min_run_interruption_count": caes_interruptions,
        "caes_min_run_compliance_rate": (
            1.0 if not caes_segments else sum(bool(item.get("completed")) for item in caes_segments) / len(caes_segments)
        ),
        "gamma": gamma,
    }


def evaluate_annual_policy(
    env,
    policy: Any,
    *,
    annual_horizon_hours: int,
    gamma: float = 0.99,
    output_dir: Path | None = None,
    continuous_soc: bool = False,
) -> dict[str, Any]:
    """按训练一致的周窗口滑动覆盖全年评估。

    Args:
        env: 电力系统环境(PowerSystemEnv) 实例。
        policy: 评估用策略，需支持 ``predict`` 及可选 ``on_episode_reset`` / ``on_transition``。
        annual_horizon_hours: 年度总小时数。
        gamma: 各窗口折扣因子。
        output_dir: 可选目录，每窗写入 ``window_XXXXh.csv``。
        continuous_soc: 若 True，改为**单次连续年轨迹**（SOC 不跨周 reset），
            见 ``evaluate_continuous_annual_policy``；主表默认 False（周窗口 reset）。

    Returns:
        全年汇总字典：总步数、总成本、各能量指标、违规计数等。

    Raises:
        ValueError: 年度小时数、决策间隔或 episode 长度配置非法时抛出。
    """
    if continuous_soc:
        return evaluate_continuous_annual_policy(
            env,
            policy,
            annual_horizon_hours=annual_horizon_hours,
            gamma=gamma,
            output_dir=output_dir,
        )
    if annual_horizon_hours <= 0:
        raise ValueError("annual_horizon_hours 必须为正数")
    step_hours = float(env.config["fmu"]["decision_interval_seconds"]) / 3600.0
    if step_hours <= 0 or not step_hours.is_integer():
        raise ValueError("年度评估要求整数小时决策间隔")
    episode_hours = int(env.episode_steps * step_hours)
    if episode_hours <= 0:
        raise ValueError("episode_steps 必须为正数")

    windows: list[dict[str, Any]] = []
    starts = list(range(0, annual_horizon_hours, episode_hours))
    for start_hour in tqdm(starts, desc="AnnualEval", unit="win", dynamic_ncols=True):
        hours = min(episode_hours, annual_horizon_hours - start_hour)
        output_csv = None
        if output_dir is not None:
            output_csv = output_dir / f"window_{start_hour:04d}h.csv"
        windows.append(
            evaluate_policy(
                env,
                policy,
                output_csv=output_csv,
                gamma=gamma,
                reset_options={"start_time": float(start_hour * 3600)},
                max_steps=hours,
            )
        )

    metric_names = tuple(windows[0]["metrics"]) if windows else ()
    return {
        "annual_horizon_hours": annual_horizon_hours,
        "protocol": "weekly_reset",
        "windows": len(windows),
        "steps": sum(int(item["steps"]) for item in windows),
        "valid_steps": sum(int(item["valid_steps"]) for item in windows),
        "annual_raw_total_cost": sum(float(item["weekly_raw_total_cost"]) for item in windows),
        "annual_episode_reward": sum(float(item["weekly_episode_reward"]) for item in windows),
        "window_discounted_return_sum": sum(float(item["weekly_discounted_return"]) for item in windows),
        "metrics": {name: sum(float(item["metrics"][name]) for item in windows) for name in metric_names},
        "fmu_failure_count": sum(int(item["fmu_failure_count"]) for item in windows),
        "forbidden_action_count": sum(int(item["forbidden_action_count"]) for item in windows),
        "invalid_transition_count": sum(int(item["invalid_transition_count"]) for item in windows),
        "terminal_soc_satisfied_windows": sum(bool(item["terminal_soc_satisfied"]) for item in windows),
        "annual_economic_cashflow": sum(
            float(item["cost_terms"].get("economic_cashflow_delta", 0.0)) for item in windows
        ),
        "caes_min_run_interruption_count": sum(int(item["caes_min_run_interruption_count"]) for item in windows),
    }


def evaluate_continuous_annual_policy(
    env,
    policy: Any,
    *,
    annual_horizon_hours: int,
    gamma: float = 0.99,
    output_dir: Path | None = None,
    start_time: float = 0.0,
    snapshot_hours: int | None = None,
) -> dict[str, Any]:
    """连续年 SOC 协议：单次 reset 后连续滚动全年，储能状态跨周传递。

    与主表 ``evaluate_annual_policy``（周窗口 reset）的差异：
    - **不**在每 168 h 边界重置 FMU / SOC；
    - episode 长度取 ``annual_horizon_hours``（通常 8760），仅在年终触发
      terminal SOC 门控；若环境构造时已设 ``episode_steps=8760``，则不再临时改写；
    - 仍按 ``snapshot_hours``（默认 168）切片汇总周级 KPI，便于与主表对照。

    推荐构造::

        env = PowerSystemEnv(episode_steps=8760, scenario_id=\"year_001\")
        evaluate_continuous_annual_policy(env, policy, annual_horizon_hours=8760)

    Args:
        env: PowerSystemEnv。
        policy: 评估策略。
        annual_horizon_hours: 年度小时数（通常 8760）。
        gamma: 折扣因子（用于切片回报；连续年总回报亦累计）。
        output_dir: 可选，写逐步 CSV 与周切片摘要。
        start_time: 仿真起点秒。
        snapshot_hours: 周切片长度；None 时若当前 episode 已是全年则用 168，
            否则用环境原 ``episode_steps`` 小时数。

    Returns:
        连续年汇总字典，含 ``protocol=\"continuous_soc\"`` 与 ``window_snapshots``。

    Raises:
        ValueError: 配置非法时抛出。
    """
    if annual_horizon_hours <= 0:
        raise ValueError("annual_horizon_hours 必须为正数")
    step_hours = float(env.config["fmu"]["decision_interval_seconds"]) / 3600.0
    if step_hours <= 0 or not step_hours.is_integer():
        raise ValueError("年度评估要求整数小时决策间隔")
    step_hours_i = int(step_hours)
    if annual_horizon_hours % step_hours_i != 0:
        raise ValueError("annual_horizon_hours 须整除决策间隔小时数")
    annual_steps = int(annual_horizon_hours // step_hours_i)
    default_episode = int(env.episode_steps)
    if snapshot_hours is not None:
        snap_h = int(snapshot_hours)
    elif default_episode == annual_steps:
        snap_h = 168  # 环境已是全年 episode，切片仍按周
    else:
        snap_h = int(default_episode * step_hours_i)
    if snap_h <= 0 or snap_h % step_hours_i != 0:
        raise ValueError("snapshot_hours 必须为正且整除决策间隔")
    snap_steps = int(snap_h // step_hours_i)

    # 若环境尚未配置为全年 episode，临时拉长；已是全年则原样使用。
    already_annual = default_episode == annual_steps
    saved_env_steps = int(env.episode_steps)
    saved_rc_steps = int(env.reward_calculator.episode_steps)
    saved_rc_cfg = env.reward_calculator.config.get("episode_steps")
    if not already_annual:
        env.episode_steps = annual_steps
        env.reward_calculator.episode_steps = annual_steps
        env.reward_calculator.config["episode_steps"] = annual_steps

    output_csv = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_csv = output_dir / "continuous_year.csv"

    try:
        full = evaluate_policy(
            env,
            policy,
            output_csv=output_csv,
            gamma=gamma,
            reset_options={"start_time": float(start_time)},
            max_steps=annual_steps,
        )
    finally:
        if not already_annual:
            env.episode_steps = saved_env_steps
            env.reward_calculator.episode_steps = saved_rc_steps
            if saved_rc_cfg is None:
                env.reward_calculator.config.pop("episode_steps", None)
            else:
                env.reward_calculator.config["episode_steps"] = saved_rc_cfg

    # 从逐步 CSV 重建周切片（若无 CSV 则仅返回年汇总）
    window_snapshots: list[dict[str, Any]] = []
    if output_csv is not None and output_csv.is_file():
        window_snapshots = _continuous_year_window_snapshots(
            output_csv, snap_steps=snap_steps, gamma=gamma
        )
        if output_dir is not None:
            snap_path = output_dir / "window_snapshots.json"
            import json

            snap_path.write_text(
                json.dumps(window_snapshots, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

    n_windows = (annual_steps + snap_steps - 1) // snap_steps
    return {
        "annual_horizon_hours": annual_horizon_hours,
        "protocol": "continuous_soc",
        "start_time": float(start_time),
        "snapshot_hours": snap_h,
        "windows": len(window_snapshots) if window_snapshots else n_windows,
        "steps": int(full["steps"]),
        "valid_steps": int(full["valid_steps"]),
        "annual_raw_total_cost": float(full["weekly_raw_total_cost"]),
        "annual_episode_reward": float(full["weekly_episode_reward"]),
        "window_discounted_return_sum": float(full["weekly_discounted_return"]),
        "metrics": dict(full["metrics"]),
        "fmu_failure_count": int(full["fmu_failure_count"]),
        "forbidden_action_count": int(full["forbidden_action_count"]),
        "invalid_transition_count": int(full["invalid_transition_count"]),
        "terminal_soc_satisfied_year_end": bool(full["terminal_soc_satisfied"]),
        "terminal_soc": full.get("terminal_soc"),
        "initial_soc": full.get("initial_soc"),
        "annual_economic_cashflow": float(
            (full.get("cost_terms") or {}).get("economic_cashflow_delta", 0.0)
        ),
        "caes_min_run_interruption_count": int(full["caes_min_run_interruption_count"]),
        "window_snapshots": window_snapshots,
        "note": (
            "Single continuous FMU trajectory; SOC carries across week boundaries. "
            "Year-end terminal SoC gate only. Main-table weekly_reset is separate."
        ),
    }


def _continuous_year_window_snapshots(
    csv_path: Path,
    *,
    snap_steps: int,
    gamma: float,
) -> list[dict[str, Any]]:
    """从连续年逐步 CSV 按 snap_steps 切片汇总。

    Args:
        csv_path: ``evaluate_policy`` 写出的逐步轨迹。
        snap_steps: 每窗步数。
        gamma: 窗内折扣因子。

    Returns:
        各窗摘要列表。
    """
    rows: list[dict[str, str]] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return []

    snapshots: list[dict[str, Any]] = []
    for start in range(0, len(rows), snap_steps):
        chunk = rows[start : start + snap_steps]
        rew = 0.0
        disc = 0.0
        raw = 0.0
        cash = 0.0
        for i, row in enumerate(chunk):
            r = float(row.get("reward") or 0.0)
            rew += r
            disc += (gamma**i) * r
            raw += float(row.get("rt_raw_total_cost") or 0.0)
            cash += float(row.get("rt_economic_cashflow_delta") or 0.0)
        last = chunk[-1]
        soc_keys = ("battery_soc", "caes_gas_soc", "caes_hot_soc", "caes_cold_soc")
        term_soc = {
            k: float(last[f"obs_{k}"]) if f"obs_{k}" in last and last[f"obs_{k}"] not in ("", None)
            else float("nan")
            for k in soc_keys
        }
        # obs_* 可能不在 CSV 列名中——回退 rt / 直接字段
        for k in soc_keys:
            if term_soc[k] != term_soc[k]:  # nan
                for cand in (f"obs_{k}", k, f"rt_{k}"):
                    if cand in last and last[cand] not in ("", None):
                        try:
                            term_soc[k] = float(last[cand])
                            break
                        except (TypeError, ValueError):
                            pass
        snapshots.append(
            {
                "window_index": len(snapshots),
                "start_step": start,
                "steps": len(chunk),
                "episode_reward": rew,
                "discounted_return": disc,
                "raw_total_cost": raw,
                "economic_cashflow_delta": cash,
                "net_cashflow_j": -raw,
                "terminal_soc": term_soc,
                "invalid_transition_count": sum(
                    1 for r in chunk if str(r.get("transition_valid")).lower() in ("false", "0")
                ),
            }
        )
    return snapshots
