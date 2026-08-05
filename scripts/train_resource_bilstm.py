#!/usr/bin/env python
"""训练资源残差 BiLSTM，导出 predicted 观测 CSV（原始物理单位）。"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forecast.resource_bilstm import (  # noqa: E402
    CHANNEL_NAMES,
    NONNEG_CHANNELS,
    ResourceBiLSTM,
    ResourceBiLSTMConfig,
)

# 导出文件名与 env_config sources 对齐
EXPORT_FILES = {
    "wind": "winds.csv",
    "irradiance": "Gstc.csv",
    "ambient_temperature": "environment.csv",
    "planned_load": "load.csv",
}


def read_residual_csv(path: Path) -> dict[str, np.ndarray]:
    cols: dict[str, list[float]] = {"time": []}
    for name in CHANNEL_NAMES:
        cols[f"{name}_da"] = []
        cols[f"{name}_realized"] = []
        cols[f"{name}_eps"] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cols["time"].append(float(row["time"]))
            for name in CHANNEL_NAMES:
                cols[f"{name}_da"].append(float(row[f"{name}_da"]))
                cols[f"{name}_realized"].append(float(row[f"{name}_realized"]))
                cols[f"{name}_eps"].append(float(row[f"{name}_eps"]))
    return {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}


def export_predictions(
    model: ResourceBiLSTM,
    data: dict[str, np.ndarray],
    out_dir: Path,
) -> dict:
    n = len(data["time"])
    c = len(CHANNEL_NAMES)
    da = np.column_stack([data[f"{name}_da"] for name in CHANNEL_NAMES])
    eps = np.column_stack([data[f"{name}_eps"] for name in CHANNEL_NAMES])
    realized = np.column_stack([data[f"{name}_realized"] for name in CHANNEL_NAMES])

    L, H = model.cfg.lookback, model.cfg.horizon
    hours = np.arange(n, dtype=np.float64)
    cal = ResourceBiLSTM.calendar_features(hours)
    feats = np.concatenate([da, eps, cal], axis=1).astype(np.float32)

    pred = da.copy()
    for t in range(L - 1, n - H):
        yhat = model.predict_eps(feats[t - L + 1 : t + 1]).reshape(H, c)
        # 只写 t+1 的一步（滚动），与 price 导出一致
        pred[t + 1] = da[t + 1] + yhat[0]
    pred[:L] = da[:L]

    for j, name in enumerate(CHANNEL_NAMES):
        if name in NONNEG_CHANNELS:
            pred[:, j] = np.maximum(pred[:, j], 0.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    step = float(data["time"][1] - data["time"][0]) if n > 1 else 3600.0
    for j, name in enumerate(CHANNEL_NAMES):
        path = out_dir / EXPORT_FILES[name]
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time", "value"])
            for i in range(n):
                w.writerow([i * step, f"{pred[i, j]:.8g}"])

    metrics: dict[str, float] = {}
    for j, name in enumerate(CHANNEL_NAMES):
        sl = slice(L, None)
        mae_p = float(np.mean(np.abs(pred[sl, j] - realized[sl, j])))
        mae_d = float(np.mean(np.abs(da[sl, j] - realized[sl, j])))
        metrics[f"mae_pred_{name}"] = mae_p
        metrics[f"mae_da_{name}"] = mae_d
        metrics[f"improvement_vs_da_{name}"] = mae_d - mae_p
    metrics["out_dir"] = str(out_dir)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/resource_residual_series.csv")
    ap.add_argument("--ckpt", default="data/forecast_models/resource_bilstm.pt")
    ap.add_argument("--pred-out-dir", default="data/resource_predicted")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lookback", type=int, default=168)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    data = read_residual_csv(ROOT / args.data)
    da = np.column_stack([data[f"{name}_da"] for name in CHANNEL_NAMES])
    eps = np.column_stack([data[f"{name}_eps"] for name in CHANNEL_NAMES])
    cfg = ResourceBiLSTMConfig(
        lookback=args.lookback,
        horizon=args.horizon,
        epochs=args.epochs,
        device=args.device,
        n_channels=len(CHANNEL_NAMES),
    )
    model = ResourceBiLSTM(cfg)
    X, y = model.build_arrays(da, eps)
    hist = model.fit(X, y)
    ckpt = ROOT / args.ckpt
    model.save(ckpt)
    metrics = export_predictions(model, data, ROOT / args.pred_out_dir)
    summary = {"train": hist, "export": metrics, "checkpoint": str(ckpt)}
    ckpt.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
