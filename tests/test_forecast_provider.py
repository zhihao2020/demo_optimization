from pathlib import Path

import numpy as np
import pytest
import yaml

from envs.forecast_provider import FORECAST_FEATURE_DIM, ForecastDataError, ForecastProvider, ForecastSource


ROOT = Path(__file__).resolve().parents[1]


def _provider() -> ForecastProvider:
    config = yaml.safe_load((ROOT / "src/config/env_config.yaml").read_text(encoding="utf-8"))
    return ForecastProvider(
        ROOT,
        config["forecast"],
        annual_horizon_hours=config["fmu"]["annual_horizon_hours"],
        step_seconds=config["fmu"]["decision_interval_seconds"],
    )


def test_forecast_is_t_plus_1_horizon_major_and_scaled():
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
    provider = _provider()
    features = provider.at_time(8760 * 3600.0).reshape(24, 4)
    assert np.allclose(features, np.repeat(features[:1], 24, axis=0))
    with pytest.raises(ForecastDataError, match="小时网格"):
        provider.at_time(1.0)


def test_forecast_csv_header_and_annual_grid_fail_fast(tmp_path):
    provider = _provider()
    bad = tmp_path / "bad.csv"
    bad.write_text("timestamp,value\n0,1\n", encoding="utf-8")
    with pytest.raises(ForecastDataError, match="表头"):
        provider._read_source(ForecastSource("wind", bad, 0.0, 1.0))

    bad.write_text("time,value\n0,1\n3600,2\n", encoding="utf-8")
    with pytest.raises(ForecastDataError, match="8761 行"):
        provider._read_source(ForecastSource("wind", bad, 0.0, 1.0))
