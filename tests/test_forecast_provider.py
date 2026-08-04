"""ForecastProvider 测试：T+1 前瞻特征、年末 clamp 与 CSV 校验。"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from envs.forecast_provider import FORECAST_FEATURE_DIM, ForecastDataError, ForecastProvider, ForecastSource


ROOT = Path(__file__).resolve().parents[1]


def _provider() -> ForecastProvider:
    """从 env_config 构造 ForecastProvider 实例。

    Returns:
        绑定仓库根目录与配置的 ForecastProvider。
    """
    config = yaml.safe_load((ROOT / "src/config/env_config.yaml").read_text(encoding="utf-8"))
    return ForecastProvider(
        ROOT,
        config["forecast"],
        annual_horizon_hours=config["fmu"]["annual_horizon_hours"],
        step_seconds=config["fmu"]["decision_interval_seconds"],
    )


def test_forecast_is_t_plus_1_horizon_major_and_scaled():
    """验证 at_time(0) 返回 T+1 四特征且按配置缩放。"""
    provider = _provider()
    features = provider.at_time(0.0)
    assert features.shape == (FORECAST_FEATURE_DIM,)
    assert np.allclose(
        features[:4],
        [1.33 / 15.0, 0.0, (262.45 - 273.15) / 40.0, 203661340.2 / 3.0e8],
    )
    assert np.allclose(
        features[4:8],
        [1.09 / 15.0, 0.0, (262.07 - 273.15) / 40.0, 236498877.6 / 3.0e8],
    )


def test_forecast_clamps_at_annual_end_and_rejects_off_grid_time():
    """验证年末时刻重复末行特征，非小时网格时刻抛 ForecastDataError。"""
    provider = _provider()
    features = provider.at_time(8760 * 3600.0).reshape(24, 4)
    assert np.allclose(features, np.repeat(features[:1], 24, axis=0))
    with pytest.raises(ForecastDataError, match="小时网格"):
        provider.at_time(1.0)


def test_forecast_csv_header_and_annual_grid_fail_fast(tmp_path):
    """验证 CSV 表头错误或行数非 8761 时 _read_source 快速失败。"""
    provider = _provider()
    bad = tmp_path / "bad.csv"
    bad.write_text("timestamp,value\n0,1\n", encoding="utf-8")
    with pytest.raises(ForecastDataError, match="表头"):
        provider._read_source(ForecastSource("wind", bad, 0.0, 1.0))

    bad.write_text("time,value\n0,1\n3600,2\n", encoding="utf-8")
    with pytest.raises(ForecastDataError, match="8761 行"):
        provider._read_source(ForecastSource("wind", bad, 0.0, 1.0))


def test_noisy_mode_differs_from_perfect_and_is_reproducible():
    """noisy 与 perfect 不同；同 seed 可复现。"""
    config = yaml.safe_load((ROOT / "src/config/env_config.yaml").read_text(encoding="utf-8"))
    fc = dict(config["forecast"])
    kwargs = dict(
        annual_horizon_hours=config["fmu"]["annual_horizon_hours"],
        step_seconds=config["fmu"]["decision_interval_seconds"],
    )
    perfect = ForecastProvider(ROOT, fc, mode="perfect", **kwargs)
    noisy_a = ForecastProvider(ROOT, fc, mode="noisy", noise_seed=7, **kwargs)
    noisy_b = ForecastProvider(ROOT, fc, mode="noisy", noise_seed=7, **kwargs)
    noisy_c = ForecastProvider(ROOT, fc, mode="noisy", noise_seed=8, **kwargs)
    p0 = perfect.at_time(0.0)
    a0 = noisy_a.at_time(0.0)
    b0 = noisy_b.at_time(0.0)
    c0 = noisy_c.at_time(0.0)
    assert perfect.mode == "perfect"
    assert noisy_a.mode == "noisy"
    assert not np.allclose(p0, a0)
    assert np.allclose(a0, b0)
    assert not np.allclose(a0, c0)
    # 非负通道（除温度）不应出现负值
    a_mat = a0.reshape(24, 4)
    assert np.all(a_mat[:, 0] >= 0.0)
    assert np.all(a_mat[:, 1] >= 0.0)
    assert np.all(a_mat[:, 3] >= 0.0)
