"""统一 KPI：从 evaluate_policy 结果抽取，并相对 B0 计算变化。

Primary ranking uses comprehensive monetary objective (generalized cashflow):
  ΔJ_gen = cash − carbon − CUT − deg − caes_startup − grid_contract
Physical KPIs (renewables, flexibility, reliability, decision time) are reported
separately and never invent FS-HSAC-only bonus scores.
"""

from __future__ import annotations

from typing import Any


_COST_KEYS = (
    "economic_cashflow_delta",
    "cashflow_delta",
    "generalized_cashflow_delta",
    "carbon_cost_cny",
    "curtailment_cost_cny",
    "unserved_cost_cny",
    "cut_cost_cny",
    "battery_deg_cost_cny",
    "caes_startup_cost_cny",
    "grid_contract_cost_cny",
    "market_buy_cost",
    "market_sell_revenue",
    "market_energy_buy_mwh",
    "market_energy_sell_mwh",
    "external_cost_cny",
    "raw_total_cost",
    "raw_generalized_cost",
)


def _f(d: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                continue
    return float(default)


def extract_kpi_from_eval(res: dict[str, Any], *, wall_s: float = 0.0, fmu_steps: int | None = None) -> dict[str, Any]:
    terms = res.get("cost_terms") or {}
    metrics = res.get("metrics") or {}
    # Primary: generalized cashflow sum (ΔJ_gen = CF − carbon − CUT − deg − su − grid).
    j_gen = terms.get("generalized_cashflow_delta")
    j_cf = terms.get("economic_cashflow_delta") or terms.get("cashflow_delta")
    if j_gen is None:
        j_gen = j_cf if j_cf is not None else 0.0
    if j_cf is None:
        j_cf = j_gen

    cut_cost = _f(terms, "cut_total_cost_cny", "cut_cost_cny")
    if cut_cost == 0.0:
        cut_cost = _f(terms, "curtailment_cost_cny") + _f(terms, "unserved_cost_cny")

    # Comprehensive cost CC ≡ −J_gen (lower better when framed as cost).
    cc = -float(j_gen)

    n_steps = float(res.get("steps") or metrics.get("decision_time_count") or 0.0) or 1.0
    timeout = _f(metrics, "solver_timeout_count")
    fail = _f(metrics, "solver_fail_count")

    return {
        # --- economics (primary ranking) ---
        "episode_reward": res.get("episode_reward"),
        "sum_delta_j_gen": float(j_gen),
        "comprehensive_cost_cny": float(cc),
        "net_cashflow_j": float(j_cf),
        "raw_total_cost": terms.get("raw_total_cost") or res.get("weekly_raw_total_cost"),
        "economic_reward": terms.get("economic_reward"),
        "market_buy_cost": terms.get("market_buy_cost"),
        "market_sell_revenue": terms.get("market_sell_revenue"),
        "market_energy_buy_mwh": terms.get("market_energy_buy_mwh"),
        "market_energy_sell_mwh": terms.get("market_energy_sell_mwh"),
        "carbon_cost_cny": _f(terms, "carbon_cost_cny"),
        "curtailment_cost_cny": _f(terms, "curtailment_cost_cny"),
        "unserved_cost_cny": _f(terms, "unserved_cost_cny"),
        "cut_cost_cny": float(cut_cost),
        "battery_deg_cost_cny": _f(terms, "battery_deg_cost_cny"),
        "caes_startup_cost_cny": _f(terms, "caes_startup_cost_cny"),
        "grid_contract_cost_cny": _f(terms, "grid_contract_cost_cny"),
        "external_cost_cny": _f(terms, "external_cost_cny"),
        # --- renewables ---
        "curtailment_mwh": metrics.get("curtailment_energy_mwh"),
        "renewable_available_mwh": metrics.get("renewable_available_mwh"),
        "wind_available_mwh": metrics.get("wind_available_mwh"),
        "pv_available_mwh": metrics.get("pv_available_mwh"),
        "wind_actual_mwh": metrics.get("wind_actual_mwh"),
        "pv_actual_mwh": metrics.get("pv_actual_mwh"),
        "curtailment_rate": metrics.get("curtailment_rate"),
        "renewable_utilization": metrics.get("renewable_utilization"),
        # --- flexibility / grid ---
        "grid_contract_excess_mwh": metrics.get("grid_contract_excess_mwh"),
        "grid_contract_violation_hours": metrics.get("grid_contract_violation_hours"),
        "grid_abs_max_mw": metrics.get("grid_abs_max_mw"),
        "grid_import_max_mw": metrics.get("grid_import_max_mw"),
        "grid_export_max_mw": metrics.get("grid_export_max_mw"),
        "grid_peak_valley_mw": metrics.get("grid_peak_valley_mw"),
        "max_grid_ramp_mw": metrics.get("max_grid_ramp_mw"),
        # --- reliability / executability ---
        "unserved_mwh": metrics.get("unserved_energy_mwh"),
        "thermal_mwh": metrics.get("thermal_generation_mwh"),
        "battery_throughput_mwh": metrics.get("battery_throughput_mwh"),
        "caes_throughput_mwh": metrics.get("caes_throughput_mwh"),
        "terminal_soc_satisfied": res.get("terminal_soc_satisfied"),
        "terminal_soc_l1": terms.get("terminal_soc_l1_error") or terms.get("terminal_soc_l1_energy"),
        "terminal_soc_l1_full": terms.get("terminal_soc_l1_full"),
        "terminal_soc": res.get("terminal_soc"),
        "fmu_failure_count": res.get("fmu_failure_count"),
        "invalid_transition_count": res.get("invalid_transition_count"),
        "forbidden_action_count": res.get("forbidden_action_count"),
        "soft_shell_count": res.get("soft_shell_count"),
        "givesafe_reject_proxy": res.get("forbidden_action_count"),
        # --- online compute ---
        "decision_time_mean_s": metrics.get("decision_time_mean_s"),
        "decision_time_p95_s": metrics.get("decision_time_p95_s"),
        "decision_time_max_s": metrics.get("decision_time_max_s"),
        "decision_time_sum_s": metrics.get("decision_time_sum_s"),
        "solver_timeout_count": timeout,
        "solver_fail_count": fail,
        "solver_timeout_rate": float(timeout / n_steps),
        "wall_s": wall_s,
        "fmu_steps": fmu_steps if fmu_steps is not None else res.get("valid_steps"),
        "steps": res.get("steps"),
        "valid_steps": res.get("valid_steps"),
    }


def cost_identity_holds(terms: dict[str, Any], *, atol: float = 1.0) -> bool:
    """Check ΔJ_gen ≈ cash − carbon − CUT − deg − su − grid (accumulated)."""
    j_gen = _f(terms, "generalized_cashflow_delta")
    cash = _f(terms, "economic_cashflow_delta", "cashflow_delta")
    carbon = _f(terms, "carbon_cost_cny")
    cut = _f(terms, "cut_total_cost_cny", "cut_cost_cny")
    if cut == 0.0:
        cut = _f(terms, "curtailment_cost_cny") + _f(terms, "unserved_cost_cny")
    deg = _f(terms, "battery_deg_cost_cny")
    su = _f(terms, "caes_startup_cost_cny")
    grid = _f(terms, "grid_contract_cost_cny")
    expected = cash - carbon - cut - deg - su - grid
    return abs(j_gen - expected) <= atol


def relative_to_baseline(kpi: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    def d(key: str) -> float | None:
        try:
            return float(kpi.get(key) or 0.0) - float(base.get(key) or 0.0)
        except Exception:
            return None

    thr_b = float(base.get("battery_throughput_mwh") or 0.0) + float(base.get("caes_throughput_mwh") or 0.0)
    thr_k = float(kpi.get("battery_throughput_mwh") or 0.0) + float(kpi.get("caes_throughput_mwh") or 0.0)
    th_b = float(base.get("thermal_mwh") or 0.0) or 1e-9
    return {
        "delta_j_vs_b0": d("net_cashflow_j"),
        "delta_cc_vs_b0": d("comprehensive_cost_cny"),
        "delta_curtailment_mwh": d("curtailment_mwh"),
        "delta_unserved_mwh": d("unserved_mwh"),
        "delta_contract_excess_mwh": d("grid_contract_excess_mwh"),
        "thermal_ratio_vs_b0": float(kpi.get("thermal_mwh") or 0.0) / th_b,
        "storage_throughput_ratio_vs_b0": thr_k / (thr_b + 1e-9),
        "delta_episode_reward": d("episode_reward"),
    }
