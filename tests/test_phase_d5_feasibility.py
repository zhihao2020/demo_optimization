"""Phase D.5：细粒度失败分类、残差、安全分类器、空可行集与边界应力测试。"""
from __future__ import annotations
import numpy as np

from envs.forecast_provider import DEFAULT_OBSERVATION_DIM
import pytest
from actions import (
    CaesMode,
    FeasibilityOracle,
    SafeActionGenerator,
    SafetyClassifier,
)
from actions.failure_taxonomy import classify_failure
from actions.feasibility_oracle import PREDICTED_STATE_KEYS
from boundary_stress import BoundaryStressTester
from envs.failures import FeasibleSetEmpty, FailureRecord, PostStepHardConstraintViolation
from envs.power_system_env import PowerSystemEnv
from training.hybrid_td3.buffer import EconomicReplayBuffer, FilteredReplayBuffer, SafetyDataset, Transition
from test_env_reset import FakeAdapter


def test_classify_battery_soc_high_low():
    """验证电池 SOC 越上/下界时失败细分类正确。"""
    fine, trig = classify_failure(
        failure_type="PostStepHardConstraintViolation",
        reason="battery_soc=0.95 越物理界 [0.1, 0.9]",
    )
    assert fine == "battery_soc_high"
    fine2, _ = classify_failure(
        failure_type="PostStepHardConstraintViolation",
        reason="battery_soc=0.05 越物理界 [0.1, 0.9]",
    )
    assert fine2 == "battery_soc_low"
def test_classify_caes_and_nonfinite():
    """验证压空 SOC 越界与非有限输出的细分类。"""
    fine, _ = classify_failure(
        failure_type="PostStepHardConstraintViolation",
        reason="caes_gas_soc=0.55 越 assert 界",
        outputs={"caes_gas_soc": 0.55},
        params={"caes": {"gas_SOC_min": 0.6, "gas_SOC_max": 1.0}, "battery": {"SOC_min": 0.1, "SOC_max": 0.9}},
    )
    assert fine == "caes_gas_soc_low"
    fine_nf, _ = classify_failure(failure_type="NonFiniteOutputFailure", reason="battery_soc 非有限")
    assert fine_nf == "nonfinite_output"
def test_failure_record_roundtrip():
    """验证失败记录 FailureRecord 序列化往返一致。"""
    rec = FailureRecord(
        run_id="t",
        episode=1,
        step=3,
        simulation_time=10800.0,
        failure_type="PostStepHardConstraintViolation",
        fine_failure_type="battery_soc_high",
        triggering_constraint="battery_soc",
        hybrid_action={"u_tp": 1.0, "u_battery": 1.0, "u_caes": 0.0},
    )
    d = rec.to_dict()
    rec2 = FailureRecord.from_dict(d)
    assert rec2.fine_failure_type == "battery_soc_high"
def test_oracle_predict_residual_and_version():
    """验证可行性神谕预测、残差与版本号。"""
    oracle = FeasibilityOracle.from_root()
    assert oracle.oracle_version.startswith("d5")
    outputs = {
        "battery_soc": 0.5,
        "caes_gas_soc": 0.85,
        "caes_hot_soc": 0.5,
        "caes_cold_soc": 0.5,
        "caes_gas_pressure": 8.5e6,
        "caes_gas_temperature": 300.0,
        "caes_hot_temperature": 400.0,
        "caes_cold_temperature": 290.0,
        "p_thermal": -1.5e8,
        "p_wind_actual": -1e6,
        "p_pv_actual": 0.0,
        "p_load_actual": 1.5e8,
    }
    from actions import PhysicalFmuAction
    action = PhysicalFmuAction(1.0, 0.5, 0.0)
    pred = oracle.predict_next_state(outputs, action, previous_thermal_w=-1.5e8)
    for k in ("battery_soc", "p_thermal", "p_grid"):
        assert k in pred
    # 人为 actual
    actual = dict(pred)
    actual["battery_soc"] = pred["battery_soc"] + 0.01
    res = oracle.residual(pred, actual)
    assert abs(res["battery_soc"] - 0.01) < 1e-9
    dang = oracle.dangerous_residual(res, mode=CaesMode.IDLE, u_battery=0.5)
    assert "battery_soc_high" in dang
def test_caes_mode_specific_mask_not_gas_alone():
    """验证压空模式掩码由气/热/冷联合约束，非仅气库。"""
    oracle = FeasibilityOracle.from_root()
    # hot 已近上限：即使 gas 仍有空间，charge 也应被联合约束压住
    outputs = {
        "battery_soc": 0.5,
        "caes_gas_soc": 0.7,
        "caes_hot_soc": 0.94,
        "caes_cold_soc": 0.5,
        "caes_gas_pressure": 8.0e6,
        "caes_gas_temperature": 300.0,
        "caes_hot_temperature": 400.0,
        "caes_cold_temperature": 290.0,
        "p_thermal": -1.5e8,
        "p_wind_actual": 0.0,
        "p_pv_actual": 0.0,
        "p_load_actual": 1.5e8,
    }
    feas = oracle.compute(outputs, previous_thermal_w=-1.5e8)
    assert feas.mode_mask.idle is True
    # charge 在 hot 近界时应为 False（联合约束）
    assert feas.mode_mask.charge is False
def test_thermal_bounds_use_actual_previous_p_thermal():
    """验证火电动态界基于上一时刻实际火电功率。"""
    oracle = FeasibilityOracle.from_root()
    outputs = {
        "battery_soc": 0.5,
        "caes_gas_soc": 0.85,
        "caes_hot_soc": 0.5,
        "caes_cold_soc": 0.5,
        "caes_gas_pressure": 8.5e6,
        "caes_gas_temperature": 300.0,
        "caes_hot_temperature": 400.0,
        "caes_cold_temperature": 290.0,
        "p_thermal": -1.5e8,  # 请求满发
        "p_wind_actual": 0.0,
        "p_pv_actual": 0.0,
        "p_load_actual": 1.5e8,
    }
    # 实际上一功率较低（u≈0.5）
    p_actual_prev = -0.5 * 1.5e8
    feas = oracle.compute(outputs, previous_thermal_w=p_actual_prev)
    # 应从 u≈0.5 的爬坡界出发，而非从 1.0
    assert feas.u_tp_high < 0.9
def test_economic_buffer_rejects_poison_reward_and_invalid():
    """验证经济回放缓冲拒绝异常奖励样本。"""
    buf = EconomicReplayBuffer(capacity=10)
    t = Transition(
        observation=np.zeros(DEFAULT_OBSERVATION_DIM, dtype=np.float32),
        hybrid_action={"u_tp": 1.0, "u_battery": 0.0, "u_caes": 0.0},
        decoded_fmu_action={"u_tp": 1.0, "u_battery": 0.0, "u_caes": 0.0},
        reward=-1e9,
        next_observation=np.zeros(DEFAULT_OBSERVATION_DIM, dtype=np.float32),
        terminated=False,
        valid_mode_mask=np.array([True, True, True]),
        dynamic_action_bounds={"u_tp_low": 1 / 3, "u_tp_high": 1.0, "u_battery_low": -1.0, "u_battery_high": 1.0},
        reward_terms={},
        physically_valid=True,
    )
    assert buf.add(t) is False
    assert len(buf) == 0
def test_safety_dataset_separate_from_economic():
    """验证安全数据集与经济缓冲相互独立。"""
    ds = SafetyDataset()
    ds.add_from_failure_record(
        {
            "fine_failure_type": "battery_soc_high",
            "previous_observation": {"battery_soc": 0.88},
            "hybrid_action": {"u_tp": 1.0, "u_battery": 1.0, "u_caes": 0.0},
        }
    )
    buf = FilteredReplayBuffer()
    assert len(ds) == 1
    assert len(buf) == 0
def test_safety_classifier_false_safe_metric():
    """验证安全分类器输出 false_safe_rate 与 unsafe_recall。"""
    clf = SafetyClassifier(threshold=0.5, model_version="test")
    # 构造可分特征：battery_soc 高 + 充电 => unsafe
    X = []
    y = []
    ftypes = []
    for soc, ub, label in [(0.2, -0.5, 1.0), (0.3, 0.0, 1.0), (0.85, 1.0, 0.0), (0.88, 0.8, 0.0)] * 20:
        outputs = {"battery_soc": soc, "caes_gas_soc": 0.8, "caes_hot_soc": 0.5, "caes_cold_soc": 0.5,
                   "caes_gas_pressure": 8e6, "caes_gas_temperature": 300.0}
        action = {"u_tp": 1.0, "u_battery": ub, "u_caes": 0.0}
        X.append(clf.featurize(outputs, action))
        y.append(label)
        ftypes.append("" if label > 0.5 else "battery_soc_high")
    X = np.stack(X)
    y = np.asarray(y)
    clf.fit(X, y, epochs=300, class_weight_unsafe=5.0)
    metrics = clf.evaluate(X, y, failure_types=ftypes, threshold=0.5)
    assert "false_safe_rate" in metrics.to_dict()
    assert metrics.unsafe_recall >= 0.5
def test_empty_feasible_set_raises():
    """验证可行集为空时安全动作生成器抛 FeasibleSetEmpty。"""
    oracle = FeasibilityOracle.from_root()
    gen = SafeActionGenerator(oracle, classifier=None, max_resamples=2)

    class EmptyOracle(FeasibilityOracle):
        """始终返回空可行集的测试用可行性神谕(EmptyOracle)。"""

        def compute(self, outputs, previous_thermal_w=None):
            """返回刻意为空的动态可行动作集。

            Args:
                outputs: 观测字典（忽略）。
                previous_thermal_w: 上一火电功率（忽略）。

            Returns:
                空的动态可行动作集(DynamicFeasibleActionSet)。
            """
            from actions import DynamicFeasibleActionSet, ModeMask
            return DynamicFeasibleActionSet(
                u_tp_low=1.0,
                u_tp_high=0.0,
                u_battery_low=1.0,
                u_battery_high=-1.0,
                mode_mask=ModeMask(False, False, False),
                metadata={"feasible_set_empty": True},
            )

        def is_feasible_set_empty(self, feasible):
            """始终判定可行集为空。

            Args:
                feasible: 动态可行动作集（忽略）。

            Returns:
                恒为真。
            """
            return True

    gen.oracle = EmptyOracle(params=oracle.params, margins=oracle.margins)
    with pytest.raises(FeasibleSetEmpty):
        gen.generate(
            {"battery_soc": 0.5, "caes_gas_soc": 0.8, "caes_hot_soc": 0.5, "caes_cold_soc": 0.5,
             "caes_gas_pressure": 8e6, "caes_gas_temperature": 300, "caes_hot_temperature": 400,
             "caes_cold_temperature": 290, "p_thermal": -1.5e8, "p_wind_actual": 0, "p_pv_actual": 0,
             "p_load_actual": 1e8},
            -1.5e8,
            lambda feas: {"u_tp": np.asarray([1.0]), "u_battery": np.asarray([0.0]), "u_caes": np.asarray([0.0])},
        )
def test_boundary_stress_tester_unit_sampling():
    """验证边界应力测试器能采样出合法场景与动作字段。"""
    oracle = FeasibilityOracle.from_root()
    tester = BoundaryStressTester(oracle=oracle, seed=1)
    outputs = {
        "battery_soc": 0.16,
        "caes_gas_soc": 0.66,
        "caes_hot_soc": 0.12,
        "caes_cold_soc": 0.12,
        "caes_gas_pressure": 6.7e6,
        "caes_gas_temperature": 300.0,
        "caes_hot_temperature": 400.0,
        "caes_cold_temperature": 290.0,
        "p_thermal": -1.0e8,
        "p_wind_actual": 0.0,
        "p_pv_actual": 0.0,
        "p_load_actual": 1.2e8,
    }
    action, scenario = tester.sample_boundary_action(outputs, previous_thermal_w=-1.0e8)
    assert scenario in BoundaryStressTester.SCENARIOS
    assert "u_tp" in action
def test_env_logs_predicted_next_state_on_success():
    """验证成功步进后 info 含神谕预测下一状态与残差。"""
    env = PowerSystemEnv(adapter=FakeAdapter())
    env.reset(seed=0)
    action = {
        "u_tp": np.asarray([1.0], dtype=np.float32),
        "u_battery": np.asarray([0.0], dtype=np.float32),
        "u_caes": np.asarray([0.0], dtype=np.float32),
    }
    obs, reward, term, trunc, info = env.step(action)
    assert info.get("oracle_predicted_next_state") is not None
    assert "battery_soc" in info["oracle_predicted_next_state"]
    assert info.get("oracle_version")
    assert info.get("residuals") is not None
    env.close()
def test_formal_gates_are_enabled_by_default():
    """验证 Phase E 正式门控默认未阻断训练。"""
    from training.hybrid_td3.train import load_phase_e_gates
    gates = load_phase_e_gates(__import__("pathlib").Path(__file__).resolve().parents[1])
    assert gates.get("formal_default_blocked", True) is False


def test_annual_episode_starts_cover_the_final_fmu_window():
    """验证年评估 episode 起点覆盖末段 FMU 窗口。"""
    from training.hybrid_td3.train import annual_episode_start_seconds

    fmu = {"start_time_seconds": 0, "decision_interval_seconds": 3600, "annual_horizon_hours": 8760}
    starts = [annual_episode_start_seconds(fmu, 168, i) / 3600 for i in range(53)]
    assert starts[:3] == [0.0, 168.0, 336.0]
    assert starts[-1] == 8760.0 - 168.0
    assert starts[-1] + 168.0 == 8760.0
