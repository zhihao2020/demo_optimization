"""Deterministic GiveSafe tries once; eval records failed_no_safe_action."""
from __future__ import annotations

from safety.givesafe_controller import GiveSafeController
from safety.no_safe_action import NoSafeActionFoundError
from training.evaluate_td3 import evaluate_policy


class _AlwaysBad:
    def predict(self, obs, deterministic=True):
        raise NoSafeActionFoundError(
            "denied",
            attempts=1,
            rejected=[{"u_tp": 1.0}],
            reasons=["oracle:grid:p_grid"],
        )


def test_deterministic_give_safe_uses_one_attempt():
    n = {"k": 0}

    class _Oracle:
        oracle_version = "t"
        def compute(self, *a, **k):
            raise AssertionError("not used")

    ctrl = GiveSafeController(
        oracle=type("O", (), {"oracle_version": "t"})(),
        config={"givesafe": {"use_fallback": False, "max_attempts_per_env_step": 64}},
    )
    # Replace checker so every sample is unsafe without needing a live FMU.
    class _Fail:
        def check(self, *a, **k):
            from safety.safety_result import SafetyCheckResult
            n["k"] += 1
            return SafetyCheckResult(safe=False, rejection_stage="oracle", violation_type="grid", oracle_rejection_reason="p_grid")

    ctrl.checker = _Fail()
    ctrl.oracle = type("O", (), {"oracle_version": "t"})()

    def propose():
        return {"u_tp": 1.0, "u_battery": 0.0, "u_caes": 0.0}

    try:
        ctrl.select_safe_action({}, 0.0, propose, deterministic=True)
        raise AssertionError("should raise")
    except NoSafeActionFoundError as exc:
        assert exc.attempts == 1
        assert n["k"] == 1
        assert exc.first_check is not None
        assert exc.first_check.violation_type == "grid"


def test_evaluate_policy_records_no_safe_action(tmp_path):
    class _Env:
        last_outputs = {"p_grid": 0.0}
        previous_thermal = 0.0
        adapter = type("A", (), {"time": 0.0})()
        valid_episode_steps = 0
        config = {"fmu": {"decision_interval_seconds": 3600}}
        reward_calculator = type("R", (), {"config": {}})()

        def reset(self, options=None):
            return [0.0], {"initial_soc": {"battery_soc": 0.5, "caes_gas_soc": 0.5}}

        def close(self):
            return None

    out = evaluate_policy(_Env(), _AlwaysBad(), tmp_path / "eval.csv", deterministic=True)
    assert out["eval_status"] == "failed_no_safe_action"
    assert out["eval_failed"] is True
    assert out["failure"]["failure_type"] == "NoSafeActionFound"
    assert out["steps"] == 0
    assert "feasible_action_spec" in out["failure"]
    assert "raw_policy_action" in out["failure"]
    assert "decoded_physical_action" in out["failure"]
    assert "state" in out["failure"]
