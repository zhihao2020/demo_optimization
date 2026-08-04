"""分时电价与 price-taker 结算单测。"""

from pathlib import Path

import numpy as np

from economics.project_kpi import annual_savings_cny, irr, lcoe, npv, project_cashflows, simple_payback_years
from envs.reward_calculator import RewardCalculator
from market.price_profile import PriceProfile
from market.settlement import composition_terms, grid_cashflow_cny, settle_grid_step


ROOT = Path(__file__).resolve().parents[1]


def test_price_profile_valley_peak_hours():
    profile = PriceProfile(
        ROOT,
        {
            "price_path": "data/price_tou.csv",
            "horizon_hours": 24,
            "buy_scale_yuan_per_kwh": 1.0,
            "sell_scale_yuan_per_kwh": 1.0,
        },
        annual_horizon_hours=8760,
        step_seconds=3600.0,
    )
    # 山东 2026 代理购电分时：1 月 hour0=平段 F≈0.635；hour16=尖峰 S≈1.136
    buy0, sell0 = profile.prices_at(0.0)
    buy16, sell16 = profile.prices_at(16 * 3600.0)
    assert abs(buy0 - 0.63457) < 1e-5
    assert abs(buy16 - 1.13597) < 1e-5
    assert buy16 > buy0
    assert abs(sell0 - 0.1875) < 1e-6
    feats = profile.features_at(0.0)
    assert feats.shape == (48,)
    assert abs(feats[0] - 0.63457) < 1e-5


def test_buy_positive_grid_is_cost_cashflow_negative():
    # 100 MW 购电 1h @ 0.6 元/kWh = 0.6*1000*100 = 60000 元成本
    cf = grid_cashflow_cny(100e6, 0.6, 0.1, 1.0)
    assert abs(cf - (-60000.0)) < 1e-6
    # 售电
    cf_sell = grid_cashflow_cny(-50e6, 0.6, 0.2, 1.0)
    assert abs(cf_sell - 10000.0) < 1e-6


def test_reward_market_replaces_fmu_grid_delta():
    calc = RewardCalculator(
        {
            "episode_steps": 168,
            "cost_reference": {"value": 1000.0},
            "terminal_soc": {"enabled": False},
        }
    )
    base = {
        "battery_soc": 0.5,
        "caes_gas_soc": 0.5,
        "caes_hot_soc": 0.5,
        "caes_cold_soc": 0.5,
        "economic_cashflow_total": 100.0,
        "economic_cashflow_wind": 10.0,
        "economic_cashflow_pv": 10.0,
        "economic_cashflow_thermal": 10.0,
        "economic_cashflow_battery": 10.0,
        "economic_cashflow_caes": 10.0,
        "economic_cashflow_load": 10.0,
        "economic_cashflow_grid": 40.0,
        "p_grid": 10e6,  # 10 MW 购电
    }
    calc.reset(base)
    # FMU: total +50, grid +30；非电网 +20。市场电网现金流应替换 grid 增量。
    nxt = dict(base)
    nxt["economic_cashflow_total"] = 150.0
    nxt["economic_cashflow_grid"] = 70.0
    # 10 MW * 1h * 0.6 元/kWh = 6000 成本 → cashflow -6000
    reward, terms = calc.calculate(
        nxt,
        is_final_step=False,
        episode_completed=False,
        no_failure=True,
        market_prices={"buy_yuan_per_kwh": 0.6, "sell_yuan_per_kwh": 0.1},
        decision_interval_hours=1.0,
    )
    # economic_delta = 50 - 30 + (-6000) = -5980
    assert abs(terms["economic_cashflow_delta"] - (-5980.0)) < 1e-6
    assert abs(terms["market_grid_cashflow"] - (-6000.0)) < 1e-6
    assert terms["market_settlement_enabled"] == 1.0
    assert abs(reward - (-5980.0 / 1000.0)) < 1e-9


def test_project_kpi_npv_irr_lcoe():
    cfs = project_cashflows(1000.0, [300.0, 300.0, 300.0, 300.0, 300.0])
    assert cfs[0] == -1000.0
    assert npv(cfs, 0.0) == 500.0
    r = irr(cfs)
    assert r == r  # not nan
    assert 0.1 < r < 0.2
    assert simple_payback_years(cfs) < 5
    cost = lcoe(1000.0, [10.0] * 5, [1000.0] * 5, 0.08)
    assert cost > 0
    assert annual_savings_cny(100.0, 70.0) == 30.0


def test_composition_terms():
    logs = [
        settle_grid_step(1e6, 0.6, 0.1, 1.0),
        settle_grid_step(-1e6, 0.6, 0.2, 1.0),
    ]
    c = composition_terms(logs)
    assert c["energy_buy_mwh"] == 1.0
    assert c["energy_sell_mwh"] == 1.0
    assert c["total_buy_cost_cny"] > 0
    assert c["total_sell_revenue_cny"] > 0
