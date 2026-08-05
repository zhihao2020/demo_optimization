#!/usr/bin/env python
"""构造风光/负荷「日前 DA + 实现 + 残差」序列，供 residual BiLSTM 训练。

realized = CSV 真值（与 FMU 边界同源）
DA       = 24h persistence（x_{t-24}）；t<24 用首日同时刻或序列均值
eps      = realized - DA

观测侧可预测 \\hat x = DA + \\hat eps；物理仍由 FMU 真值驱动。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

CHANNELS = (
    ("wind", "data/winds.csv"),
    ("irradiance", "data/Gstc.csv"),
    ("ambient_temperature", "data/environment.csv"),
    ("planned_load", "data/load.csv"),
)


def _read_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    vals: list[float] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["time", "value"]:
            raise ValueError(f"{path.name} 表头须为 time,value")
        for row in reader:
            times.append(float(row["time"]))
            vals.append(float(row["value"]))
    return np.asarray(times, dtype=np.float64), np.asarray(vals, dtype=np.float64)


def persistence_da(realized: np.ndarray, lag: int = 24) -> np.ndarray:
    """24h 持续性日前预报。"""
    n = len(realized)
    da = np.empty(n, dtype=np.float64)
    mean0 = float(np.mean(realized[: min(lag, n)]))
    for t in range(n):
        if t >= lag:
            da[t] = realized[t - lag]
        else:
            # 首日：用序列前 lag 均值，避免未来泄漏
            da[t] = mean0
    return da


def main() -> None:
    p = argparse.ArgumentParser(description="构造资源 DA/realized/residual 序列")
    p.add_argument("--out", type=str, default="data/resource_residual_series.csv")
    p.add_argument("--meta-out", type=str, default="data/resource_residual_meta.json")
    p.add_argument("--lag", type=int, default=24)
    args = p.parse_args()

    series: dict[str, dict[str, np.ndarray]] = {}
    times_ref: np.ndarray | None = None
    for name, rel in CHANNELS:
        path = ROOT / rel
        times, realized = _read_series(path)
        if times_ref is None:
            times_ref = times
        elif len(times) != len(times_ref) or not np.allclose(times, times_ref):
            raise ValueError(f"{rel} 时间轴与 winds 不一致")
        da = persistence_da(realized, lag=int(args.lag))
        eps = realized - da
        series[name] = {"da": da, "realized": realized, "eps": eps}

    assert times_ref is not None
    n = len(times_ref)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["time"]
    for name, _ in CHANNELS:
        fieldnames.extend([f"{name}_da", f"{name}_realized", f"{name}_eps"])

    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in range(n):
            row: dict[str, float | str] = {"time": float(times_ref[i])}
            for name, _ in CHANNELS:
                row[f"{name}_da"] = float(series[name]["da"][i])
                row[f"{name}_realized"] = float(series[name]["realized"][i])
                row[f"{name}_eps"] = float(series[name]["eps"][i])
            w.writerow(row)

    meta = {
        "da_definition": f"persistence-{args.lag}h",
        "n": n,
        "channels": [c[0] for c in CHANNELS],
        "sources": {c[0]: c[1] for c in CHANNELS},
        "note": (
            "realized = FMU-aligned CSV truth; DA = lag persistence; "
            "BiLSTM predicts eps; predicted obs = DA + eps_hat. Physics stays FMU."
        ),
        "mae_da": {
            name: float(np.mean(np.abs(series[name]["eps"])))
            for name, _ in CHANNELS
        },
    }
    meta_path = ROOT / args.meta_out
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": str(out), "meta": str(meta_path), "mae_da": meta["mae_da"]}, indent=2))


if __name__ == "__main__":
    main()
