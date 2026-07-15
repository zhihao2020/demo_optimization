from envs.reward_calculator import RewardCalculator


def _outputs(**overrides):
    data = {
        "battery_soc": .5, "caes_gas_soc": .5, "caes_hot_soc": .5, "caes_cold_soc": .5,
        "economic_cashflow_total": 100.0,
        "economic_cashflow_wind": 10.0, "economic_cashflow_pv": 10.0,
        "economic_cashflow_thermal": 10.0, "economic_cashflow_battery": 10.0,
        "economic_cashflow_caes": 10.0, "economic_cashflow_load": 10.0,
        "economic_cashflow_grid": 40.0,
    }
    data.update(overrides)
    return data


def _calculator():
    calculator = RewardCalculator({
        "episode_steps": 168,
        "cost_reference": {"value": 1000.0},
        "terminal_soc": {"enabled": False},
    })
    calculator.reset(_outputs())
    return calculator


def test_reward_is_modelica_cashflow_delta_not_python_repricing():
    current = _outputs(economic_cashflow_total=160.0, economic_cashflow_grid=100.0)
    reward, terms = _calculator().calculate(
        current, is_final_step=False, episode_completed=False, no_failure=True,
    )
    assert reward == 0.06
    assert terms["economic_cashflow_delta"] == 60.0
    assert terms["raw_total_cost"] == -60.0
    assert terms["raw_grid_cost"] == -60.0
    assert "raw_thermal_cost" in terms


def test_reset_baseline_makes_first_delta_relative_to_fmu_reset_value():
    calculator = _calculator()
    _, first = calculator.calculate(_outputs(economic_cashflow_total=110.0), is_final_step=False, episode_completed=False, no_failure=True)
    _, second = calculator.calculate(_outputs(economic_cashflow_total=105.0), is_final_step=False, episode_completed=False, no_failure=True)
    assert first["economic_cashflow_delta"] == 10.0
    assert second["economic_cashflow_delta"] == -5.0
