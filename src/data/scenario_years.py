"""情景年：按气象季节分桶，整周块 bootstrap。

设计要点
--------
- 一年 8760 h = 52×168 h + 余 24 h。整周重采样保持日内/周内自相关；
  同季节桶内抽取，避免把盛夏周贴进隆冬。
- ``year_000`` 恒为基准年恒等拷贝，作留出对照。
- 边界四通道与分时电价共用同一套 donor 周索引，保证「同一条替代年」一致。
- ``resource_predicted/`` 不在此重采样（需另跑预测模型）；perfect/noisy 模式
  直接读情景年真值即可。
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

WEEK_HOURS = 168
YEAR_HOURS = 8760
N_WEEKS = YEAR_HOURS // WEEK_HOURS  # 52
REMAINDER_HOURS = YEAR_HOURS - N_WEEKS * WEEK_HOURS  # 24
STEP_SECONDS = 3600.0

# 边界通道：文件名与 BoundaryProvider / forecast 配置对齐
BOUNDARY_FILES: tuple[tuple[str, str], ...] = (
    ("wind", "winds.csv"),
    ("irradiance", "Gstc.csv"),
    ("ambient_temperature", "environment.csv"),
    ("planned_load", "load.csv"),
)

SEASON_BY_MONTH = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}


class ScenarioYearError(ValueError):
    """情景年生成或加载契约错误。"""


def season_of_week(week_index: int, *, year: int = 2019) -> str:
    """非闰年 1 月 1 日起算，第 ``week_index`` 周所属气象季节。"""
    if not 0 <= week_index < N_WEEKS:
        raise ScenarioYearError(f"week_index 越界: {week_index}")
    dt = datetime(year, 1, 1) + timedelta(hours=week_index * WEEK_HOURS)
    return SEASON_BY_MONTH[dt.month]


def season_of_hour(hour_index: int, *, year: int = 2019) -> str:
    """第 ``hour_index`` 小时（0-based）所属气象季节。"""
    if not 0 <= hour_index < YEAR_HOURS:
        raise ScenarioYearError(f"hour_index 越界: {hour_index}")
    dt = datetime(year, 1, 1) + timedelta(hours=hour_index)
    return SEASON_BY_MONTH[dt.month]


def build_season_buckets(*, year: int = 2019) -> dict[str, list[int]]:
    """季节 -> 该季节内基准年周索引列表。"""
    buckets: dict[str, list[int]] = {
        "winter": [],
        "spring": [],
        "summer": [],
        "autumn": [],
    }
    for w in range(N_WEEKS):
        buckets[season_of_week(w, year=year)].append(w)
    for name, weeks in buckets.items():
        if not weeks:
            raise ScenarioYearError(f"季节 {name} 无可用周，无法 bootstrap")
    return buckets


@dataclass(frozen=True)
class SeriesSpec:
    """待重采样的序列规格。"""

    name: str
    path: Path
    columns: tuple[str, ...]  # 除 time 外的数值列；price 可有多列
    n_rows_expected: int  # 8761（边界含端点）或 8760（电价）


def _read_numeric_csv(path: Path, columns: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise ScenarioYearError(f"缺少源文件: {path}")
    times: list[float] = []
    cols: dict[str, list[float]] = {c: [] for c in columns}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"time", *columns}
        if reader.fieldnames is None or required - set(reader.fieldnames):
            raise ScenarioYearError(
                f"{path.name} 表头须含 {sorted(required)}，实际 {reader.fieldnames}"
            )
        for row_number, row in enumerate(reader, start=2):
            try:
                times.append(float(row["time"]))
                for c in columns:
                    cols[c].append(float(row[c]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ScenarioYearError(f"{path.name}:{row_number} 非法") from exc
    expected_t = [i * STEP_SECONDS for i in range(len(times))]
    if any(abs(t - e) > 1e-6 for t, e in zip(times, expected_t)):
        raise ScenarioYearError(f"{path.name} 时间轴不是从 0 起的严格小时网格")
    return {"time": np.asarray(times, dtype=np.float64), **{
        c: np.asarray(v, dtype=np.float64) for c, v in cols.items()
    }}


def _write_numeric_csv(
    path: Path,
    *,
    columns: tuple[str, ...],
    values: Mapping[str, np.ndarray],
    extra_columns: Mapping[str, list[Any]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(values[columns[0]])
    fieldnames = ["time", *columns]
    if extra_columns:
        fieldnames = ["time", *columns, *extra_columns.keys()]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n):
            row: dict[str, Any] = {"time": int(i * STEP_SECONDS)}
            for c in columns:
                row[c] = float(values[c][i])
            if extra_columns:
                for k, col in extra_columns.items():
                    row[k] = col[i]
            writer.writerow(row)


def _resample_body(
    body: np.ndarray,
    *,
    rng: np.random.Generator,
    buckets: Mapping[str, list[int]],
    week_seasons: Sequence[str] | list[str],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """对长度为 8760 的主体做整周 + 余日 bootstrap，返回 (8760,), donor 日志。"""
    if body.shape[0] < YEAR_HOURS:
        raise ScenarioYearError(f"序列长度 {body.shape[0]} < {YEAR_HOURS}")
    body = body[:YEAR_HOURS]
    pieces: list[np.ndarray] = []
    donors: list[dict[str, Any]] = []
    for slot, season in enumerate(week_seasons):
        donor = int(rng.choice(buckets[season]))
        lo = donor * WEEK_HOURS
        pieces.append(body[lo : lo + WEEK_HOURS])
        donors.append(
            {"slot": slot, "kind": "week", "season": season, "donor_week": donor}
        )
    # 余 24 h：从同季节某周中抽一整天
    rem_season = season_of_hour(N_WEEKS * WEEK_HOURS)
    donor = int(rng.choice(buckets[rem_season]))
    day = int(rng.integers(0, WEEK_HOURS // 24))
    lo = donor * WEEK_HOURS + day * 24
    pieces.append(body[lo : lo + REMAINDER_HOURS])
    donors.append(
        {
            "slot": "remainder",
            "kind": "day",
            "season": rem_season,
            "donor_week": donor,
            "donor_day": day,
        }
    )
    out = np.concatenate(pieces)
    if out.shape[0] != YEAR_HOURS:
        raise ScenarioYearError(f"内部错误：重采样长度 {out.shape[0]}")
    return out, donors


def _with_endpoint(body_8760: np.ndarray) -> np.ndarray:
    """边界 CSV 需要 8761 点：端点取最后一小时值（ConstantSegments 友好）。"""
    return np.concatenate([body_8760, body_8760[-1:]])


@dataclass
class ScenarioYearGenerator:
    """从基准 ``data/`` 生成情景年目录树。"""

    root: Path
    out_root: Path
    seed: int = 0
    n_years: int = 10
    calendar_year: int = 2019

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.out_root = Path(self.out_root)
        if self.n_years < 1:
            raise ScenarioYearError("n_years 必须 >= 1")

    def _boundary_specs(self) -> list[SeriesSpec]:
        return [
            SeriesSpec(name, self.root / "data" / filename, ("value",), YEAR_HOURS + 1)
            for name, filename in BOUNDARY_FILES
        ]

    def _price_spec(self) -> SeriesSpec | None:
        path = self.root / "data" / "price_tou.csv"
        if not path.is_file():
            return None
        return SeriesSpec(
            "price_tou",
            path,
            ("buy_yuan_per_kwh", "sell_yuan_per_kwh"),
            YEAR_HOURS,
        )

    def generate(self) -> dict[str, Any]:
        """生成全部情景年并写 manifest，返回 manifest 字典。"""
        buckets = build_season_buckets(year=self.calendar_year)
        week_seasons = [season_of_week(w, year=self.calendar_year) for w in range(N_WEEKS)]
        boundary_specs = self._boundary_specs()
        price_spec = self._price_spec()

        # 预读基准序列
        base_boundary: dict[str, dict[str, np.ndarray]] = {}
        for spec in boundary_specs:
            series = _read_numeric_csv(spec.path, spec.columns)
            if len(series["time"]) != spec.n_rows_expected:
                raise ScenarioYearError(
                    f"{spec.path.name} 期望 {spec.n_rows_expected} 行，"
                    f"实际 {len(series['time'])}"
                )
            base_boundary[spec.name] = series

        base_price = None
        price_bands: list[str] | None = None
        if price_spec is not None:
            base_price = _read_numeric_csv(price_spec.path, price_spec.columns)
            if len(base_price["time"]) != price_spec.n_rows_expected:
                raise ScenarioYearError(
                    f"{price_spec.path.name} 期望 {price_spec.n_rows_expected} 行，"
                    f"实际 {len(base_price['time'])}"
                )
            # band 列保留（非数值）
            price_bands = []
            with price_spec.path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames and "band" in reader.fieldnames:
                    price_bands = [str(row["band"]) for row in reader]

        self.out_root.mkdir(parents=True, exist_ok=True)
        year_entries: list[dict[str, Any]] = []

        for year_idx in range(self.n_years):
            year_id = f"year_{year_idx:03d}"
            year_dir = self.out_root / year_id
            if year_dir.exists():
                shutil.rmtree(year_dir)
            year_dir.mkdir(parents=True)

            if year_idx == 0:
                donors = [
                    {
                        "slot": w,
                        "kind": "week",
                        "season": week_seasons[w],
                        "donor_week": w,
                    }
                    for w in range(N_WEEKS)
                ] + [
                    {
                        "slot": "remainder",
                        "kind": "identity_tail",
                        "season": season_of_hour(N_WEEKS * WEEK_HOURS),
                        "hour_start": N_WEEKS * WEEK_HOURS,
                    }
                ]
                # 恒等：直接拷贝文件，避免浮点改写
                for _, filename in BOUNDARY_FILES:
                    shutil.copy2(self.root / "data" / filename, year_dir / filename)
                if price_spec is not None:
                    shutil.copy2(price_spec.path, year_dir / "price_tou.csv")
                kind = "identity"
                year_seed = None
            else:
                year_seed = int(self.seed) + year_idx * 1009
                rng = np.random.default_rng(year_seed)
                # 抽一套 donor，再重放到所有通道
                _, donors = _resample_body(
                    base_boundary["wind"]["value"][:YEAR_HOURS],
                    rng=rng,
                    buckets=buckets,
                    week_seasons=week_seasons,
                )
                for spec in boundary_specs:
                    rebuilt = _apply_donors(
                        base_boundary[spec.name]["value"][:YEAR_HOURS], donors
                    )
                    full = _with_endpoint(rebuilt)
                    _write_numeric_csv(
                        year_dir / dict(BOUNDARY_FILES)[spec.name],
                        columns=("value",),
                        values={"value": full},
                    )
                if price_spec is not None and base_price is not None:
                    buy = _apply_donors(base_price["buy_yuan_per_kwh"][:YEAR_HOURS], donors)
                    sell = _apply_donors(
                        base_price["sell_yuan_per_kwh"][:YEAR_HOURS], donors
                    )
                    extra = None
                    if price_bands is not None and len(price_bands) >= YEAR_HOURS:
                        band_body = _apply_donors_labels(
                            price_bands[:YEAR_HOURS], donors
                        )
                        extra = {"band": band_body}
                    _write_numeric_csv(
                        year_dir / "price_tou.csv",
                        columns=("buy_yuan_per_kwh", "sell_yuan_per_kwh"),
                        values={
                            "buy_yuan_per_kwh": buy,
                            "sell_yuan_per_kwh": sell,
                        },
                        extra_columns=extra,
                    )
                kind = "bootstrap"

            try:
                rel_path = str(year_dir.relative_to(self.root)).replace("\\", "/")
            except ValueError:
                rel_path = str(year_dir)
            meta = {
                "id": year_id,
                "kind": kind,
                "seed": year_seed,
                "path": rel_path,
                "donors": donors,
            }
            (year_dir / "year_meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            year_entries.append(meta)

        try:
            out_rel = str(self.out_root.relative_to(self.root)).replace("\\", "/")
        except ValueError:
            out_rel = str(self.out_root)
        manifest = {
            "version": 1,
            "method": "seasonal_week_bootstrap",
            "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "base_data": "data/",
            "out_root": out_rel,
            "seed": int(self.seed),
            "n_years": int(self.n_years),
            "calendar_year_for_seasons": int(self.calendar_year),
            "year_hours": YEAR_HOURS,
            "week_hours": WEEK_HOURS,
            "n_weeks": N_WEEKS,
            "remainder_hours": REMAINDER_HOURS,
            "season_buckets": {k: list(v) for k, v in buckets.items()},
            "week_seasons": week_seasons,
            "channels": {
                "boundaries": [f for _, f in BOUNDARY_FILES],
                "price": "price_tou.csv" if price_spec is not None else None,
            },
            "years": year_entries,
            "note": (
                "year_000 是基准年恒等拷贝；其余年为同季节整周 bootstrap。"
                "训练/评估时通过 OPTIMAL_DEMO_SCENARIO=year_00X 或 "
                "env_config.scenarios.active 切换；forecast 与 boundaries 必须指向同一情景。"
            ),
        }
        manifest_path = self.out_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return manifest


def _apply_donors(body: np.ndarray, donors: list[dict[str, Any]]) -> np.ndarray:
    """按已抽好的 donor 日志重放重采样（保证多通道同步）。"""
    pieces: list[np.ndarray] = []
    for entry in donors:
        if entry["kind"] == "week":
            donor = int(entry["donor_week"])
            lo = donor * WEEK_HOURS
            pieces.append(body[lo : lo + WEEK_HOURS])
        elif entry["kind"] == "day":
            donor = int(entry["donor_week"])
            day = int(entry["donor_day"])
            lo = donor * WEEK_HOURS + day * 24
            pieces.append(body[lo : lo + REMAINDER_HOURS])
        elif entry["kind"] == "identity_tail":
            lo = int(entry["hour_start"])
            pieces.append(body[lo : lo + REMAINDER_HOURS])
        else:
            raise ScenarioYearError(f"未知 donor kind: {entry}")
    out = np.concatenate(pieces)
    if out.shape[0] != YEAR_HOURS:
        raise ScenarioYearError(f"重放长度错误: {out.shape[0]}")
    return out


def _apply_donors_labels(labels: list[str], donors: list[dict[str, Any]]) -> list[str]:
    """对字符串标签列做与数值通道相同的 donor 重放。"""
    arr = list(labels)
    pieces: list[list[str]] = []
    for entry in donors:
        if entry["kind"] == "week":
            donor = int(entry["donor_week"])
            lo = donor * WEEK_HOURS
            pieces.append(arr[lo : lo + WEEK_HOURS])
        elif entry["kind"] == "day":
            donor = int(entry["donor_week"])
            day = int(entry["donor_day"])
            lo = donor * WEEK_HOURS + day * 24
            pieces.append(arr[lo : lo + REMAINDER_HOURS])
        elif entry["kind"] == "identity_tail":
            lo = int(entry["hour_start"])
            pieces.append(arr[lo : lo + REMAINDER_HOURS])
        else:
            raise ScenarioYearError(f"未知 donor kind: {entry}")
    out: list[str] = []
    for p in pieces:
        out.extend(p)
    return out


def resolve_scenario_dir(root: Path, scenario_id: str, *, scenarios_root: str = "data/scenarios") -> Path:
    """解析情景年目录；不存在则报错。"""
    path = Path(root) / scenarios_root / scenario_id
    if not path.is_dir():
        raise ScenarioYearError(f"情景年目录不存在: {path}")
    for _, filename in BOUNDARY_FILES:
        if not (path / filename).is_file():
            raise ScenarioYearError(f"情景年缺少 {filename}: {path}")
    return path


def apply_scenario_to_env_config(
    config: dict[str, Any],
    root: Path,
    scenario_id: str,
) -> dict[str, Any]:
    """深拷贝 env 配置，把 boundaries / forecast / market 路径改到情景年目录。

    predicted_sources 保持不动（预测残差需另训）；perfect/noisy 读情景真值。
    """
    import copy

    cfg = copy.deepcopy(config)
    scenarios_cfg = cfg.get("scenarios") or {}
    scenarios_root = str(scenarios_cfg.get("root", "data/scenarios"))
    year_dir = resolve_scenario_dir(root, scenario_id, scenarios_root=scenarios_root)
    rel = str(year_dir.relative_to(root)).replace("\\", "/")

    name_to_file = dict(BOUNDARY_FILES)
    for src in cfg.get("boundaries", {}).get("sources") or []:
        fname = name_to_file[str(src["name"])]
        src["path"] = f"{rel}/{fname}"
    for src in cfg.get("forecast", {}).get("sources") or []:
        fname = name_to_file[str(src["name"])]
        src["path"] = f"{rel}/{fname}"

    market = cfg.setdefault("market", {})
    price_file = year_dir / "price_tou.csv"
    if price_file.is_file():
        market["price_path"] = f"{rel}/price_tou.csv"
        # 观测价若未单独指定，跟随结算价
        if not market.get("obs_price_path"):
            market["obs_price_path"] = None

    cfg.setdefault("scenarios", {})["active"] = scenario_id
    return cfg
