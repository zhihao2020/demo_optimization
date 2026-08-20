"""Unified KPI extraction and metric finalization (comprehensive-cost alignment)."""

from __future__ import annotations

from optimization.metrics import cost_identity_holds, extract_kpi_from_eval
from training.evaluate_td3 import _finalize_metric_rates


def test_cost_identity_holds_with_all_channels():
    terms = {
        "economic_cashflow_delta": 1000.0,
        "generalized_cashflow_delta": 700.0,
        "carbon_cost_cny": 50.0,
        "cut_total_cost_cny": 80.0,
        "battery_deg_cost_cny": 40.0,
        "caes_startup_cost_cny": 30.0,
        "grid_contract_cost_cny": 100.0,
    }
    assert cost_identity_holds(terms, atol=1e-6)
    terms["generalized_cashflow_delta"] = 701.0
    assert not cost_identity_holds(terms, atol=1e-6)


def test_extract_kpi_four_groups_and_decision_time():
    res = {
        "episode_reward": 12.0,
        "steps": 168,
        "valid_steps": 168,
        "fmu_failure_count": 0,
        "invalid_transition_count": 1,
        "forbidden_action_count": 2,
        "soft_shell_count": 0,
        "terminal_soc_satisfied": True,
        "terminal_soc": {"battery_soc": 0.5},
        "cost_terms": {
            "generalized_cashflow_delta": 1.0e6,
            "economic_cashflow_delta": 1.2e6,
            "carbon_cost_cny": 5.0e4,
            "curtailment_cost_cny": 1.0e4,
            "unserved_cost_cny": 0.0,
            "cut_total_cost_cny": 1.0e4,
            "battery_deg_cost_cny": 2.0e4,
            "caes_startup_cost_cny": 4.617e3,
            "grid_contract_cost_cny": 1.5e4,
            "terminal_soc_l1_error": 0.01,
        },
        "metrics": {
            "curtailment_energy_mwh": 12.0,
            "unserved_energy_mwh": 0.0,
            "renewable_available_mwh": 100.0,
            "curtailment_rate": 0.12,
            "renewable_utilization": 0.88,
            "grid_contract_excess_mwh": 3.0,
            "grid_contract_violation_hours": 5.0,
            "grid_abs_max_mw": 250.0,
            "grid_peak_valley_mw": 400.0,
            "max_grid_ramp_mw": 80.0,
            "thermal_generation_mwh": 20000.0,
            "battery_throughput_mwh": 400.0,
            "caes_throughput_mwh": 800.0,
            "decision_time_mean_s": 0.02,
            "decision_time_p95_s": 0.05,
            "decision_time_max_s": 0.1,
            "decision_time_sum_s": 3.36,
            "solver_timeout_count": 0.0,
            "solver_fail_count": 0.0,
        },
    }
    kpi = extract_kpi_from_eval(res, wall_s=10.0)
    assert kpi["sum_delta_j_gen"] == 1.0e6
    assert kpi["comprehensive_cost_cny"] == -1.0e6
    assert kpi["curtailment_mwh"] == 12.0
    assert kpi["curtailment_rate"] == 0.12
    assert kpi["grid_contract_excess_mwh"] == 3.0
    assert kpi["decision_time_p95_s"] == 0.05
    assert kpi["solver_timeout_rate"] == 0.0
    assert kpi["valid_steps"] == 168
    for key in (
        "carbon_cost_cny",
        "caes_startup_cost_cny",
        "grid_contract_cost_cny",
        "renewable_utilization",
        "grid_peak_valley_mw",
        "unserved_mwh",
        "decision_time_mean_s",
    ):
        assert key in kpi
        assert kpi[key] is not None


def test_finalize_metric_rates_curtailment_and_missing_times():
    metrics = {
        "curtailment_energy_mwh": 10.0,
        "renewable_available_mwh": 100.0,
        "wind_available_mwh": 60.0,
        "pv_available_mwh": 40.0,
        "wind_actual_mwh": 55.0,
        "pv_actual_mwh": 35.0,
        "grid_export_max_mw": 100.0,
        "grid_import_max_mw": 150.0,
    }
    _finalize_metric_rates(metrics)
    assert abs(metrics["curtailment_rate"] - 0.1) < 1e-9
    assert abs(metrics["renewable_utilization"] - 0.9) < 1e-9
    assert abs(metrics["grid_peak_valley_mw"] - 250.0) < 1e-9
    assert "_decision_times_s" not in metrics
    assert metrics["decision_time_mean_s"] == 0.0
    assert metrics["decision_time_p95_s"] == 0.0


def test_finalize_metric_rates_with_decision_times():
    metrics = {
        "curtailment_energy_mwh": 0.0,
        "renewable_available_mwh": 1.0,
        "wind_available_mwh": 1.0,
        "pv_available_mwh": 0.0,
        "wind_actual_mwh": 1.0,
        "pv_actual_mwh": 0.0,
        "grid_export_max_mw": 0.0,
        "grid_import_max_mw": 0.0,
        "_decision_times_s": [0.1, 0.2, 0.3, 0.4],
    }
    _finalize_metric_rates(metrics)
    assert abs(metrics["decision_time_mean_s"] - 0.25) < 1e-9
    assert metrics["decision_time_max_s"] == 0.4
    assert "_decision_times_s" not in metrics


def test_kpi_from_eval_compat_missing_decision_fields():
    res = {
        "episode_reward": 1.0,
        "steps": 10,
        "valid_steps": 10,
        "cost_terms": {"generalized_cashflow_delta": 100.0},
        "metrics": {"curtailment_energy_mwh": 0.0, "unserved_energy_mwh": 0.0},
    }
    kpi = extract_kpi_from_eval(res)
    assert kpi["sum_delta_j_gen"] == 100.0
    assert kpi["comprehensive_cost_cny"] == -100.0
    assert kpi["decision_time_mean_s"] is None
    assert kpi["solver_timeout_rate"] == 0.0
