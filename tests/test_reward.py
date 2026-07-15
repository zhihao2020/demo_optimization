from envs.reward_calculator import RewardCalculator


def _outputs(**overrides):
    data = {"p_grid": 1e6, "p_thermal": -10e6, "p_battery": 0, "p_caes": 0, "p_curtailment": 0, "p_unserved": 0,
            "battery_soc": .5, "caes_gas_soc": .5, "caes_hot_soc": .5, "caes_cold_soc": .5}
    data.update(overrides)
    return data


def _calculator():
    config = {
        "decision_interval_seconds": 3600,
        "episode_steps": 168,
        "buy_price_yuan_per_mwh": 100,
        "sell_price_yuan_per_mwh": 50,
        "thermal_a": 0,
        "thermal_b": 0,
        "thermal_c": 0,
        "curtailment_yuan_per_mwh": 20,
        "unserved_yuan_per_mwh": 10000,
        "battery_throughput_yuan_per_mwh": 1,
        "caes_throughput_yuan_per_mwh": 1,
        "ramp_yuan_per_mw": 2,
        "cost_reference": {"value": 1000.0},
        "terminal_soc": {
            "enabled": True,
            "mode": "binary_bonus",
            "bonus": 5.0,
            "tolerance": 0.01,
            "weights": {"battery_soc": 1.0, "caes_gas_soc": 1.0, "caes_hot_soc": 1.0, "caes_cold_soc": 1.0},
            "require_complete_episode": True,
            "require_no_failure": True,
        },
    }
    calculator = RewardCalculator(config)
    calculator.reset(_outputs())
    return calculator


def test_reward_equals_negative_normalized_cost():
    reward, terms = _calculator().calculate(
        _outputs(p_unserved=1e6, p_curtailment=1e6),
        -10e6,
        is_final_step=False,
        episode_completed=False,
        no_failure=True,
    )
    assert abs(reward - (-terms["normalized_cost"])) < 1e-9
    assert terms["raw_unserved_cost"] > terms["raw_curtailment_cost"] > 0
    assert "solver_failure_cost" not in terms


def test_ramp_increases_raw_cost():
    calculator = _calculator()
    _, base = calculator.calculate(_outputs(), -10e6, is_final_step=False, episode_completed=False, no_failure=True)
    _, ramped = calculator.calculate(_outputs(p_thermal=-20e6), -10e6, is_final_step=False, episode_completed=False, no_failure=True)
    assert ramped["raw_ramp_cost"] > base["raw_ramp_cost"]
