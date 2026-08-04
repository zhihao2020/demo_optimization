"""分时购售电价时序：小时网格，可做当前价与日前前瞻特征。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class PriceProfileError(ValueError):
    """电价 CSV 与时间轴契约不一致。"""


class PriceProfile:
    """``time,buy_yuan_per_kwh,sell_yuan_per_kwh`` 小时表；前瞻顺序 horizon-major: buy,sell。"""

    CHANNELS = ("buy_price", "sell_price")

    def __init__(
        self,
        root: Path,
        config: Mapping[str, Any],
        *,
        annual_horizon_hours: int,
        step_seconds: float,
    ) -> None:
        self.root = Path(root)
        self.horizon_hours = int(config.get("horizon_hours", 24))
        self.step_seconds = float(step_seconds)
        self.annual_horizon_hours = int(annual_horizon_hours)
        if self.horizon_hours <= 0:
            raise PriceProfileError("market.horizon_hours 必须为正")
        if self.step_seconds <= 0:
            raise PriceProfileError("decision step 必须为正")

        # 结算价（realized）；观测可用独立预测价路径
        settle_rel = str(config.get("price_path", "data/price_tou.csv"))
        obs_rel = config.get("obs_price_path") or settle_rel
        self.path = self.root / settle_rel
        self.obs_path = self.root / str(obs_rel)
        self.buy_scale = float(config.get("buy_scale_yuan_per_kwh", 1.0))
        self.sell_scale = float(config.get("sell_scale_yuan_per_kwh", 1.0))
        if self.buy_scale <= 0 or self.sell_scale <= 0:
            raise PriceProfileError("电价归一化 scale 必须为正")

        self._buy, self._sell = self._load_series(self.path)
        if self.obs_path.resolve() == self.path.resolve():
            self._obs_buy, self._obs_sell = self._buy, self._sell
        else:
            self._obs_buy, self._obs_sell = self._load_series(self.obs_path)

    def _load_series(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        if not path.is_file():
            raise PriceProfileError(f"电价 CSV 不存在: {path}")
        times: list[float] = []
        buys: list[float] = []
        sells: list[float] = []
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"time", "buy_yuan_per_kwh", "sell_yuan_per_kwh"}
            if reader.fieldnames is None or required - set(reader.fieldnames):
                raise PriceProfileError(
                    f"{path.name} 表头须含 time,buy_yuan_per_kwh,sell_yuan_per_kwh"
                )
            for row_number, row in enumerate(reader, start=2):
                try:
                    times.append(float(row["time"]))
                    buys.append(float(row["buy_yuan_per_kwh"]))
                    sells.append(float(row["sell_yuan_per_kwh"]))
                except (KeyError, TypeError, ValueError) as exc:
                    raise PriceProfileError(f"{path.name} 第 {row_number} 行非法") from exc

        if len(times) < self.annual_horizon_hours:
            raise PriceProfileError(
                f"电价长度 {len(times)} < annual_horizon_hours={self.annual_horizon_hours}"
            )
        expected = [i * self.step_seconds for i in range(len(times))]
        if any(abs(t - e) > 1e-6 for t, e in zip(times, expected)):
            raise PriceProfileError(f"{path.name} 时间轴必须为 0,dt,2dt,...")

        buy = np.asarray(buys[: self.annual_horizon_hours], dtype=np.float64)
        sell = np.asarray(sells[: self.annual_horizon_hours], dtype=np.float64)
        if not np.all(np.isfinite(buy)) or not np.all(np.isfinite(sell)):
            raise PriceProfileError(f"{path.name} 电价含 NaN/Inf")
        if np.any(buy < 0) or np.any(sell < 0):
            raise PriceProfileError(f"{path.name} 电价不能为负")
        return buy, sell

    @property
    def feature_dim(self) -> int:
        return self.horizon_hours * len(self.CHANNELS)

    @property
    def feature_low(self) -> np.ndarray:
        return np.zeros(self.feature_dim, dtype=np.float32)

    @property
    def feature_high(self) -> np.ndarray:
        return np.full(self.feature_dim, np.inf, dtype=np.float32)

    def _hour_index(self, time_seconds: float) -> int:
        if self.step_seconds <= 0:
            raise PriceProfileError("step_seconds 非法")
        idx = int(round(float(time_seconds) / self.step_seconds))
        if idx < 0 or idx >= self.annual_horizon_hours:
            # 尾段 wrap：最后一个窗口可覆盖
            idx = int(idx) % self.annual_horizon_hours
        return idx

    def prices_at(self, time_seconds: float) -> tuple[float, float]:
        """返回 (buy, sell) 元/kWh。"""
        i = self._hour_index(time_seconds)
        return float(self._buy[i]), float(self._sell[i])

    def features_at(self, time_seconds: float) -> np.ndarray:
        """观测用价（可为预测序列）；horizon-major: buy,sell。"""
        i0 = self._hour_index(time_seconds)
        out = np.empty(self.feature_dim, dtype=np.float32)
        k = 0
        n = self.annual_horizon_hours
        for h in range(self.horizon_hours):
            i = (i0 + h) % n
            out[k] = float(self._obs_buy[i] / self.buy_scale)
            out[k + 1] = float(self._obs_sell[i] / self.sell_scale)
            k += 2
        return out
