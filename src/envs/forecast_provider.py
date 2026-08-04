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
# 与 env_config market.horizon_hours=24、buy/sell 两通道一致（见 market.PriceProfile）
DEFAULT_MARKET_HORIZON_HOURS = 24
MARKET_PRICE_CHANNELS = 2
MARKET_FEATURE_DIM = DEFAULT_MARKET_HORIZON_HOURS * MARKET_PRICE_CHANNELS
DEFAULT_OBSERVATION_DIM = BASE_OBSERVATION_DIM + FORECAST_FEATURE_DIM + MARKET_FEATURE_DIM


class ForecastDataError(ValueError):
    """前瞻 CSV 与训练时间轴契约不一致(ForecastDataError)。"""


@dataclass(frozen=True)
class ForecastSource:
    """单通道前瞻数据源(ForecastSource)。

    Attributes:
        name: 通道名，须与 ``FORECAST_CHANNELS`` 顺序一致。
        path: CSV 文件路径。
        offset: 线性缩放偏移量。
        scale: 线性缩放系数（非零有限值）。
    """

    name: str
    path: Path
    offset: float
    scale: float


class ForecastProvider:
    """日前预测提供器(ForecastProvider)。

    在严格小时网格上提供日前预测观测；特征顺序为 horizon-major（先通道后时刻）。

    - ``mode=perfect``：CSV 真值完美前瞻（默认主表）
    - ``mode=noisy``：对真值乘性噪声 + 可选时移，模拟有误差的日前预报；
      **仅污染观测**，不改变 FMU 物理真值与结算
    """

    def __init__(
        self,
        root: Path,
        config: Mapping[str, Any],
        *,
        annual_horizon_hours: int,
        step_seconds: float,
        mode: str | None = None,
        noise_seed: int | None = None,
        noise_sigma: Mapping[str, float] | None = None,
        lag_hours: int | None = None,
    ) -> None:
        """加载并校验全部前瞻 CSV。

        Args:
            root: 项目根目录，用于解析相对路径。
            config: ``forecast`` 配置段（``horizon_hours``、``sources``、可选 ``mode``/``noise``）。
            annual_horizon_hours: 年度仿真小时数（与 FMU 一致）。
            step_seconds: 决策步长（秒），须与 CSV 时间网格一致。
            mode: 覆盖配置的 ``perfect`` / ``noisy``。
            noise_seed: 覆盖噪声种子。
            noise_sigma: 覆盖各通道乘性误差标准差。
            lag_hours: 覆盖预报时移（小时，正=滞后）。

        Raises:
            ForecastDataError: 配置、路径、网格或数值非法。
        """
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
        perfect = np.stack([self._read_source(source) for source in self.sources], axis=1)

        noise_cfg = dict(config.get("noise") or {})
        self.mode = str(mode if mode is not None else config.get("mode", "perfect")).lower()
        if self.mode not in ("perfect", "noisy"):
            raise ForecastDataError(f"未知 forecast.mode={self.mode!r}（支持 perfect|noisy）")
        self.noise_seed = int(
            noise_seed if noise_seed is not None else noise_cfg.get("seed", 0)
        )
        self.lag_hours = int(
            lag_hours if lag_hours is not None else noise_cfg.get("lag_hours", 0)
        )
        default_sigma = {
            "wind": 0.10,
            "irradiance": 0.10,
            "ambient_temperature": 0.0,
            "planned_load": 0.08,
        }
        sigma_map = dict(default_sigma)
        raw_sigma = noise_sigma if noise_sigma is not None else noise_cfg.get("sigma")
        if isinstance(raw_sigma, Mapping):
            for key, val in raw_sigma.items():
                sigma_map[str(key)] = float(val)
        elif raw_sigma is not None:
            # 标量：施加到除温度外的通道
            s = float(raw_sigma)
            for key in ("wind", "irradiance", "planned_load"):
                sigma_map[key] = s
        self.noise_sigma = {name: float(sigma_map.get(name, 0.0)) for name in FORECAST_CHANNELS}

        self._perfect_values = perfect
        if self.mode == "perfect":
            self._values = perfect
        else:
            self._values = self._build_noisy_values(perfect)

    def _build_noisy_values(self, perfect: np.ndarray) -> np.ndarray:
        """由完美序列构造可复现 noisy 日前预报序列。

        Args:
            perfect: 形状 ``(H+1, C)`` 的缩放真值。

        Returns:
            同形状 noisy 序列（非负通道裁剪到 ≥0）。
        """
        rng = np.random.default_rng(self.noise_seed)
        noisy = perfect.copy()
        n_rows, n_ch = perfect.shape
        for c, name in enumerate(FORECAST_CHANNELS):
            sigma = float(self.noise_sigma.get(name, 0.0))
            if sigma <= 0.0:
                continue
            eps = rng.normal(0.0, sigma, size=n_rows)
            noisy[:, c] = perfect[:, c] * (1.0 + eps)
            # 风速/辐照/负荷缩放后仍应非负；温度通道可保留负值（相对 273.15 的偏差）
            if name != "ambient_temperature":
                noisy[:, c] = np.maximum(noisy[:, c], 0.0)
        if self.lag_hours != 0:
            # 正 lag：预报滞后于真值（用更早时刻当真预报）
            shift = int(self.lag_hours)
            idx = np.clip(np.arange(n_rows) - shift, 0, n_rows - 1)
            noisy = noisy[idx]
        if not np.all(np.isfinite(noisy)):
            raise ForecastDataError("noisy forecast 含 NaN/Inf")
        return noisy.astype(np.float64, copy=False)

    @property
    def feature_dim(self) -> int:
        """前瞻特征总维度。

        Returns:
            ``horizon_hours * 通道数``。
        """
        return self.horizon_hours * len(self.sources)

    @property
    def feature_low(self) -> np.ndarray:
        """前瞻特征的 ``Box`` 下界（无界）。

        Returns:
            长度为 ``feature_dim`` 的 ``-inf`` 数组。
        """
        return np.full(self.feature_dim, -np.inf, dtype=np.float32)

    @property
    def feature_high(self) -> np.ndarray:
        """前瞻特征的 ``Box`` 上界（无界）。

        Returns:
            长度为 ``feature_dim`` 的 ``+inf`` 数组。
        """
        return np.full(self.feature_dim, np.inf, dtype=np.float32)

    def _read_source(self, source: ForecastSource) -> np.ndarray:
        """读取单通道 CSV 并校验年度小时网格。

        Args:
            source: 前瞻数据源(ForecastSource)。

        Returns:
            长度为 ``annual_horizon_hours + 1`` 的缩放后序列。

        Raises:
            ForecastDataError: 文件缺失、表头/行数/时间轴或数值非法。
        """
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
                        raise ForecastDataError(
                            f"{source.path.name}:{row_number} 存在非数值 time/value"
                        ) from exc
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
        """返回 ``t+1`` 到 ``t+horizon`` 的缩放特征；年度末用最后值填充。

        Args:
            simulation_time_seconds: 当前仿真时刻（秒），须在小时网格上。

        Returns:
            展平为 ``float32`` 的一维前瞻特征向量。

        Raises:
            ForecastDataError: 仿真时间不在 forecast 小时网格内。
        """
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
