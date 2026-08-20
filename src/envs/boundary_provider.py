"""向 FMU 写入外部边界条件（风速/辐照/气温/计划负荷）。

新导出的 PowerSystem_8760h 已把物理驱动从内嵌 CombiTimeTable 改为四个
RealInput；若不逐步写入，边界会冻在 start 值。本模块只读物理真值 CSV，
不得与 ``forecast`` 观测污染混用——noisy/predicted 模式只能改 observation，
绝不能改 FMU 真值。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

# Python 通道名 -> FMU RealInput 名；顺序固定，与 env_config.boundaries.sources 对齐。
BOUNDARY_CHANNELS: tuple[tuple[str, str], ...] = (
    ("wind", "v_wind_in"),
    ("irradiance", "g_irradiance_in"),
    ("ambient_temperature", "t_air_in"),
    ("planned_load", "p_load_plan_in"),
)
BOUNDARY_FMU_NAMES: tuple[str, ...] = tuple(fmu for _, fmu in BOUNDARY_CHANNELS)


class BoundaryDataError(ValueError):
    """边界 CSV 与仿真时间轴契约不一致。"""


@dataclass(frozen=True)
class BoundarySource:
    """单通道边界数据源（物理单位，不做观测缩放）。"""

    name: str
    fmu_variable: str
    path: Path


class BoundaryProvider:
    """边界条件提供器：按小时网格返回写入 FMU 的物理真值。

    CSV 表头必须严格为 ``time,value``，与 ``data/*.csv`` 及 FMU 内嵌表同源。
    """

    def __init__(
        self,
        root: Path,
        config: Mapping[str, Any],
        *,
        annual_horizon_hours: int,
        step_seconds: float,
    ) -> None:
        self.root = Path(root)
        self.step_seconds = float(step_seconds)
        self.annual_horizon_hours = int(annual_horizon_hours)
        if self.step_seconds <= 0:
            raise BoundaryDataError("boundaries 的时间步长必须为正数")
        if self.annual_horizon_hours <= 0:
            raise BoundaryDataError("annual_horizon_hours 必须为正数")

        raw_sources = config.get("sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise BoundaryDataError("boundaries.sources 必须是非空列表")

        expected = {name: fmu for name, fmu in BOUNDARY_CHANNELS}
        sources: list[BoundarySource] = []
        seen: set[str] = set()
        for raw in raw_sources:
            try:
                name = str(raw["name"])
                fmu_variable = str(raw.get("fmu_variable", expected.get(name, "")))
                path = self.root / str(raw["path"])
            except (KeyError, TypeError, ValueError) as exc:
                raise BoundaryDataError(f"非法 boundary source: {raw!r}") from exc
            if name not in expected:
                raise BoundaryDataError(f"未知 boundary 通道 {name!r}，期望 {tuple(expected)}")
            if fmu_variable != expected[name]:
                raise BoundaryDataError(
                    f"边界通道 {name!r} 的 fmu_variable 必须为 {expected[name]!r}，"
                    f"得到 {fmu_variable!r}"
                )
            if name in seen:
                raise BoundaryDataError(f"重复 boundary 通道 {name!r}")
            seen.add(name)
            sources.append(BoundarySource(name, fmu_variable, path))
        if seen != set(expected):
            raise BoundaryDataError(
                f"boundaries.sources 必须恰好覆盖 {tuple(expected)}，实际 {tuple(seen)}"
            )
        # 按 BOUNDARY_CHANNELS 固定顺序重排，避免 YAML 顺序漂移
        by_name = {s.name: s for s in sources}
        self.sources = tuple(by_name[name] for name, _ in BOUNDARY_CHANNELS)
        self._values = {
            source.fmu_variable: self._read_source(source) for source in self.sources
        }

    def _read_source(self, source: BoundarySource) -> np.ndarray:
        """读取物理单位序列并校验全年小时网格。"""
        if not source.path.is_file():
            raise BoundaryDataError(f"boundary CSV 不存在: {source.path}")
        times: list[float] = []
        values: list[float] = []
        try:
            with source.path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != ["time", "value"]:
                    raise BoundaryDataError(
                        f"{source.path.name} 表头必须严格为 time,value"
                    )
                for row_number, row in enumerate(reader, start=2):
                    try:
                        time = float(row["time"])
                        value = float(row["value"])
                    except (TypeError, ValueError, KeyError) as exc:
                        raise BoundaryDataError(
                            f"{source.path.name}:{row_number} 存在非数值 time/value"
                        ) from exc
                    if not np.isfinite(time) or not np.isfinite(value):
                        raise BoundaryDataError(
                            f"{source.path.name}:{row_number} 含 NaN/Inf"
                        )
                    times.append(time)
                    values.append(value)
        except OSError as exc:
            raise BoundaryDataError(f"读取 boundary CSV 失败: {source.path}") from exc

        expected_count = self.annual_horizon_hours + 1
        if len(times) != expected_count:
            raise BoundaryDataError(
                f"{source.path.name} 必须含 {expected_count} 行年度端点数据，"
                f"实际为 {len(times)}"
            )
        arr_time = np.asarray(times, dtype=np.float64)
        expected_time = np.arange(expected_count, dtype=np.float64) * self.step_seconds
        if not np.allclose(arr_time, expected_time, rtol=0.0, atol=1e-6):
            raise BoundaryDataError(
                f"{source.path.name} 必须从 0 开始、按 {self.step_seconds:g}s "
                "严格覆盖全年小时网格"
            )
        return np.asarray(values, dtype=np.float64)

    def at_time(self, simulation_time_seconds: float) -> dict[str, float]:
        """返回当前通信点应写入 FMU 的边界字典（FMU 变量名 -> 物理值）。

        Args:
            simulation_time_seconds: 当前仿真时刻（秒），须落在小时网格上。

        Returns:
            含四个边界输入的字典。

        Raises:
            BoundaryDataError: 时间不在网格内。
        """
        time = float(simulation_time_seconds)
        index = int(round(time / self.step_seconds))
        if index < 0 or index > self.annual_horizon_hours or not np.isclose(
            time, index * self.step_seconds, rtol=0.0, atol=1e-6
        ):
            raise BoundaryDataError(f"仿真时间不在 boundary 小时网格内: {time}")
        # 序列长度 = annual_horizon_hours + 1；通信点 t=k·dt 取第 k 行。
        return {
            name: float(series[index]) for name, series in self._values.items()
        }
