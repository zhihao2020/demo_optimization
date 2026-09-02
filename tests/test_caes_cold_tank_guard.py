"""冷罐方向回归测试。

充电抽冷罐、放电回灌冷罐，冷罐守卫方向必须与气罐/热罐相反。
历史上 alpha_cold 与 alpha_hot 同号，使一步预测认为充电会抬高冷罐，
投影层因此从不拦截把冷罐抽干的充电动作，连续年运行在约 1276 h 触发
Modelica 的 assert(cold_SOC > 0.05)。本文件把该方向锁死。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.actions.caes_u import (
    CHARGE_HI,
    CHARGE_LO,
    DISCHARGE_HI,
    DISCHARGE_LO,
    clamp_u_caes_to_spec,
    u_from_mode_mag,
)
from src.actions.feasibility_oracle import FeasibilityOracle
from src.actions.mode_mask import ModeMask
from src.actions.types import CaesMode, PhysicalFmuAction


def make_outputs(**overrides: float) -> dict[str, float]:
    """构造一个位于安全带中心的 FMU 输出快照。"""
    base = {
        "battery_soc": 0.5,
        "caes_gas_soc": 0.80,
        "caes_hot_soc": 0.50,
        "caes_cold_soc": 0.50,
        "caes_gas_pressure": 8.0e6,
        "caes_gas_temperature": 300.0,
        "caes_hot_temperature": 400.0,
        "caes_cold_temperature": 290.0,
        "p_thermal": -3.0e8,
        "p_battery": 0.0,
        "p_caes": 0.0,
        "p_grid": 0.0,
    }
    base.update(overrides)
    return base


@pytest.fixture(scope="module")
def oracle() -> FeasibilityOracle:
    return FeasibilityOracle()


def test_charge_lowers_cold_soc(oracle: FeasibilityOracle) -> None:
    """充电必须预测冷罐下降、热罐与气罐上升。"""
    out = make_outputs()
    charge = {"u_tp": 1.0, "u_battery": 0.0, "u_caes": 1.0}
    pred = oracle.predict_next_state(out, charge)
    assert pred["caes_gas_soc"] > out["caes_gas_soc"]
    assert pred["caes_hot_soc"] > out["caes_hot_soc"]
    assert pred["caes_cold_soc"] < out["caes_cold_soc"], "充电应抽低冷罐"


def test_discharge_raises_cold_soc(oracle: FeasibilityOracle) -> None:
    """放电必须预测冷罐上升、热罐与气罐下降。"""
    out = make_outputs()
    discharge = {"u_tp": 1.0, "u_battery": 0.0, "u_caes": -1.0}
    pred = oracle.predict_next_state(out, discharge)
    assert pred["caes_gas_soc"] < out["caes_gas_soc"]
    assert pred["caes_hot_soc"] < out["caes_hot_soc"]
    assert pred["caes_cold_soc"] > out["caes_cold_soc"], "放电应回灌冷罐"


def test_charge_blocked_when_cold_near_lower_bound(oracle: FeasibilityOracle) -> None:
    """冷罐贴近下界时必须禁止充电——这是 1276 h 崩溃的直接触发条件。"""
    cold_min = float(oracle.params["caes"]["cold_SOC_min"])
    # 0902 一步充电只抽 ~0.009 SOC；贴近下界 0.01 仍应禁止，0.07 已在一步安全区内。
    out = make_outputs(caes_cold_soc=cold_min + 0.01)
    mask = oracle._caes_mode_mask(out)
    assert not mask.charge, "冷罐见底仍允许充电，冷罐守卫方向未生效"
    assert mask.idle, "idle 不应被禁止"


def test_discharge_blocked_when_cold_near_upper_bound(
    oracle: FeasibilityOracle,
) -> None:
    """冷罐贴近上界时必须禁止放电（放电会继续回灌冷罐）。"""
    cold_max = float(oracle.params["caes"]["cold_SOC_max"])
    out = make_outputs(caes_cold_soc=cold_max - 0.01)
    mask = oracle._caes_mode_mask(out)
    assert not mask.discharge, "冷罐接近上界仍允许放电"
    assert mask.idle


def test_pressure_soc_coupling_matches_fmu(oracle: FeasibilityOracle) -> None:
    """FMU 中 gas_soc = p / 1e7，一步预测的压力斜率必须与之一致。"""
    out = make_outputs()
    # 充电合法带是 [0.86, 1.0]；空隙内的幅值会被 project_u_caes 压成 idle
    charge = {"u_tp": 1.0, "u_battery": 0.0, "u_caes": 0.9}
    pred = oracle.predict_next_state(out, charge)
    d_soc = pred["caes_gas_soc"] - out["caes_gas_soc"]
    d_p = pred["caes_gas_pressure"] - out["caes_gas_pressure"]
    assert d_soc > 0
    assert d_p / d_soc == pytest.approx(1.0e7, rel=1e-6)


def test_alphas_are_asymmetric_and_cold_is_negative(oracle: FeasibilityOracle) -> None:
    """实测充放电斜率不对称，且冷罐系数在两个方向上都为负。"""
    em = oracle.margins["caes"]["energy_model"]
    chg = oracle._caes_alphas(em, +0.1)
    dis = oracle._caes_alphas(em, -0.1)
    assert chg["cold"] < 0 and dis["cold"] < 0
    assert chg["hot"] > 0 and dis["hot"] > 0
    assert abs(dis["gas"]) > abs(chg["gas"]), "放电斜率幅度应大于充电"


# 气罐的实际下界由压力硬界决定：gas_pressure_min_Pa=6.5e6 且 soc=p/1e7，
# 故有效区间是 [0.65, 0.95]，比 gas_SOC_min=0.6 / max=1.0 更紧。
# 0.66 对应 6.6 MPa，落在 idle 压力 envelope（约 0.20 MPa）内，不是「界内」。
GAS_IN_BOUNDS = (0.70, 0.80, 0.90)


def test_mode_mask_never_forbids_idle_inside_bounds(oracle: FeasibilityOracle) -> None:
    """生存性过滤不得把 idle 也禁掉，否则投影层无动作可退。"""
    for cold in (0.06, 0.30, 0.50, 0.70, 0.94):
        for gas in GAS_IN_BOUNDS:
            mask = oracle._caes_mode_mask(
                make_outputs(
                    caes_cold_soc=cold, caes_gas_soc=gas, caes_gas_pressure=gas * 1e7
                )
            )
            assert mask.idle, f"cold={cold}, gas={gas} 下 idle 被禁"


def test_static_wall_tightens_near_cold_bounds(oracle: FeasibilityOracle) -> None:
    """0902 一步模型较准，近界主要靠静态裕度禁充/放，而不是后继可逆性过滤。"""
    lo = float(oracle.params["caes"]["cold_SOC_min"])
    hi = float(oracle.params["caes"]["cold_SOC_max"])
    mid = oracle._caes_mode_mask(make_outputs(caes_cold_soc=0.5))
    near_lo = oracle._caes_mode_mask(make_outputs(caes_cold_soc=lo + 0.01))
    near_hi = oracle._caes_mode_mask(make_outputs(caes_cold_soc=hi - 0.01))
    assert mid.charge and mid.discharge and mid.idle
    assert not near_lo.charge
    assert near_lo.idle
    assert not near_hi.discharge
    assert near_hi.idle


def test_u_from_mode_mag_sign_convention() -> None:
    """符号约定：充电为正、放电为负。"""
    assert u_from_mode_mag(CaesMode.CHARGE, 1.0) > 0
    assert u_from_mode_mag(CaesMode.DISCHARGE, 1.0) < 0


def test_u_from_mode_mag_is_band_position_not_physical_magnitude() -> None:
    """锁死这个易错语义：mag 是带内位置，放电方向与物理幅值反号。

    历史上可行性判定把 mag=1.0 当「最强动作」，对放电取到的却是最弱端 -0.33，
    于是方向被判合法、带内更强的幅值却会越界。
    """
    assert u_from_mode_mag(CaesMode.DISCHARGE, 0.0) == pytest.approx(-1.0)
    assert u_from_mode_mag(CaesMode.DISCHARGE, 1.0) == pytest.approx(-0.33)
    assert u_from_mode_mag(CaesMode.CHARGE, 0.0) == pytest.approx(0.86)
    assert u_from_mode_mag(CaesMode.CHARGE, 1.0) == pytest.approx(1.0)


def test_feasible_set_exposes_magnitude_intervals(oracle: FeasibilityOracle) -> None:
    """可行集必须给出各方向的安全幅值子区间，而不只是方向开关。"""
    out = make_outputs()
    spec = oracle.compute(out, out["p_thermal"]).as_dict()
    assert spec["caes_discharge_allowed"]
    assert spec["caes_charge_allowed"]
    assert DISCHARGE_LO <= spec["u_caes_discharge_low"] <= spec["u_caes_discharge_high"] <= DISCHARGE_HI
    assert CHARGE_LO <= spec["u_caes_charge_low"] <= spec["u_caes_charge_high"] <= CHARGE_HI


def test_every_u_in_reported_interval_passes_precheck(oracle: FeasibilityOracle) -> None:
    """核心一致性：区间内的每个幅值都必须通过逐动作预检。

    这正是历史上失效的性质——掩码说方向合法，带内具体幅值却被拒。
    """
    for cold in (0.15, 0.50, 0.85):
        for gas in GAS_IN_BOUNDS:
            out = make_outputs(
                caes_cold_soc=cold, caes_gas_soc=gas, caes_gas_pressure=gas * 1e7
            )
            feas = oracle.compute(out, out["p_thermal"])
            for span in (feas.u_caes_discharge, feas.u_caes_charge):
                if span is None:
                    continue
                for u in np.linspace(span[0], span[1], 9):
                    ok, why = oracle.check_action_executable(
                        PhysicalFmuAction(0.5, 0.0, float(u)), out, feas, out["p_thermal"]
                    )
                    assert ok, f"cold={cold} gas={gas} u={u}: {why}"


def test_clamp_projects_magnitude_but_not_direction(oracle: FeasibilityOracle) -> None:
    """幅值越界走投影，方向非法必须原样放行以便被验证器拒绝并计入审计。"""
    out = make_outputs()
    feas = oracle.compute(out, out["p_thermal"])
    span = feas.u_caes_discharge
    assert span is not None
    # 幅值越出区间 -> 被夹紧
    too_strong = min(span) - 0.2
    clamped, changed = clamp_u_caes_to_spec(too_strong, feas)
    if too_strong < min(span):
        assert changed and clamped == pytest.approx(min(span))

    # 方向被禁 -> 原样返回，交给验证器拒绝
    blocked = replace(feas, mode_mask=ModeMask(discharge=False, idle=True, charge=True))
    passthrough, changed2 = clamp_u_caes_to_spec(-0.9, blocked)
    assert passthrough == pytest.approx(-0.9)
    assert not changed2


def test_gas_temp_min_is_not_a_feasibility_constraint(oracle: FeasibilityOracle) -> None:
    """气罐温度下界是 Modelica 的 max() 数值钳位，不得参与可行性判定。

    否则寒冷时段气罐贴在钳位上时，一步预测会认为任何放电都越界，
    等于让环境温度把 CAES 放电永久锁死。
    """
    assert oracle.params["caes"]["gas_temp_min_is_numeric_clamp"] is True
    at_clamp = make_outputs(caes_gas_temperature=253.0)
    ok, why = oracle.post_step_hard_ok(at_clamp, use_safe=False)
    assert ok, why
    mask = oracle._caes_mode_mask(at_clamp)
    assert mask.discharge, "气罐贴在温度钳位上时放电被禁"
    # 上界仍是真实约束
    too_hot = make_outputs(caes_gas_temperature=500.0)
    assert not oracle.post_step_hard_ok(too_hot, use_safe=False)[0]


def test_magnitude_caps_report_discharge_correctly(oracle: FeasibilityOracle) -> None:
    """放电 cap 不得恒为 0（旧二分实现因单调性反号一直给 0）。"""
    out = make_outputs()
    caps = oracle.compute(out, out["p_thermal"]).metadata["caes_magnitude_caps"]
    assert caps["discharge"] > 0.0
    assert caps["charge"] > 0.0
