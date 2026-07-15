"""读取 ``data/`` 中与 FMU 同源的已知日前时序，提供策略前瞻观测。

CSV 只扩展策略的 observation；FMU 仍是风光负荷的唯一物理驱动与真值来源。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


BASE_OBSERVATION_DIM = 19
DEFAULT_FORECAST_HORIZON_HOURS = 24
FORECAST_CHANNELS = ("wind", "irradiance", "ambient_temperature", "planned_load")
FORECAST_FEATURE_DIM = DEFAULT_FORECAST_HORIZON_HOURS * len(FORECAST_CHANNELS)
DEFAULT_OBSERVATION_DIM = BASE_OBSERVATION_DIM + FORECAST_FEATURE_DIM


class ForecastDataError(ValueError):
    """前瞻 CSV 与训练时间轴契约不一致。"""


@dataclass(frozen=True)
class ForecastSource:
    name: str
    path: Path
    offset: float
    scale: float


class ForecastProvider:
    """严格小时网格上的完美日前预测，特征顺序为 horizon-major。"""

    def __init__(
        self,
        root: Path,
        config: Mapping[str, Any],
        *,
        annual_horizon_hours: int,
        step_seconds: float,
    ) -> None:
        self.root = Path(root)
        self.horizon_hours = int(config.get("horizon_hours", DEFAULT_FORECAST_HORIZON_HOURS))
        self.step_seconds = float(step_seconds)
        self.annual_horizon_hours = int(annual_horizon_hours)
        if self.horizon_hours <= 0:
            raise ForecastDataError("forecast.horizon_hours 必须为正数")
        if self.step_seconds <= 0:
            raise ForecastDataError("forecast 的时间步长必须为正数")

        raw_sources = config.get("sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise ForecastDataError("forecast.sources 必须是非空列表")
        sources: list[ForecastSource] = []
        for raw in raw_sources:
            try:
                name = str(raw["name"])
                path = self.root / str(raw["path"])
                offset = float(raw.get("offset", 0.0))
                scale = float(raw["scale"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ForecastDataError(f"非法 forecast source: {raw!r}") from exc
            if scale == 0.0 or not np.isfinite(scale) or not np.isfinite(offset):
                raise ForecastDataError(f"forecast source {name!r} 的 offset/scale 非法")
            sources.append(ForecastSource(name, path, offset, scale))
        if tuple(item.name for item in sources) != FORECAST_CHANNELS:
            raise ForecastDataError(f"forecast source 顺序必须为 {FORECAST_CHANNELS}")
        self.sources = tuple(sources)
        self._values = np.stack([self._read_source(source) for source in self.sources], axis=1)

    @property
    def feature_dim(self) -> int:
        return self.horizon_hours * len(self.sources)

    @property
    def feature_low(self) -> np.ndarray:
        return np.full(self.feature_dim, -np.inf, dtype=np.float32)

    @property
    def feature_high(self) -> np.ndarray:
        return np.full(self.feature_dim, np.inf, dtype=np.float32)

    def _read_source(self, source: ForecastSource) -> np.ndarray:
        if not source.path.is_file():
            raise ForecastDataError(f"forecast CSV 不存在: {source.path}")
        times: list[float] = []
        values: list[float] = []
        try:
            with source.path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != ["time", "value"]:
                    raise ForecastDataError(f"{source.path.name} 表头必须严格为 time,value")
                for row_number, row in enumerate(reader, start=2):
                    try:
                        time = float(row["time"])
                        value = float(row["value"])
                    except (TypeError, ValueError, KeyError) as exc:
                        raise ForecastDataError(f"{source.path.name}:{row_number} 存在非数值 time/value") from exc
                    if not np.isfinite(time) or not np.isfinite(value):
                        raise ForecastDataError(f"{source.path.name}:{row_number} 含 NaN/Inf")
                    times.append(time)
                    values.append(value)
        except OSError as exc:
            raise ForecastDataError(f"读取 forecast CSV 失败: {source.path}") from exc

        expected_count = self.annual_horizon_hours + 1
        if len(times) != expected_count:
            raise ForecastDataError(
                f"{source.path.name} 必须含 {expected_count} 行年度端点数据，实际为 {len(times)}"
            )
        arr_time = np.asarray(times, dtype=np.float64)
        expected_time = np.arange(expected_count, dtype=np.float64) * self.step_seconds
        if not np.allclose(arr_time, expected_time, rtol=0.0, atol=1e-6):
            raise ForecastDataError(
                f"{source.path.name} 必须从 0 开始、按 {self.step_seconds:g}s 严格覆盖全年小时网格"
            )
        raw_values = np.asarray(values, dtype=np.float64)
        return (raw_values - source.offset) / source.scale

    def at_time(self, simulation_time_seconds: float) -> np.ndarray:
        """返回 ``t+1`` 到 ``t+horizon`` 的缩放特征；年度末用最后值填充。"""
        time = float(simulation_time_seconds)
        index = int(round(time / self.step_seconds))
        if index < 0 or index > self.annual_horizon_hours or not np.isclose(
            time, index * self.step_seconds, rtol=0.0, atol=1e-6
        ):
            raise ForecastDataError(f"仿真时间不在 forecast 小时网格内: {time}")
        future = np.minimum(
            np.arange(index + 1, index + self.horizon_hours + 1),
            self.annual_horizon_hours,
        )
        return self._values[future].astype(np.float32, copy=False).reshape(-1)
