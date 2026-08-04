#!/usr/bin/env python
"""训练电价残差 BiLSTM，导出 predicted / realized 购售电价 CSV。"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forecast.price_bilstm import PriceBiLSTM, PriceBiLSTMConfig  # noqa: E402


def read_residual_csv(path: Path) -> dict[str, np.ndarray]:
    cols = {k: [] for k in ("buy_da", "sell_da", "buy_realized", "sell_realized", "eps_buy", "eps_sell")}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            for k in cols:
                cols[k].append(float(row[k]))
    return {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}


def export_predictions(model: PriceBiLSTM, data: dict[str, np.ndarray], out_path: Path) -> dict:
    L, H = model.cfg.lookback, model.cfg.horizon
    n = len(data["buy_da"])
    hours = np.arange(n, dtype=np.float64)
    cal = PriceBiLSTM.calendar_features(hours)
    feats = np.column_stack(
        [
            data["buy_da"],
            data["sell_da"],
            data["eps_buy"],
            data["eps_sell"],
            cal[:, 0],
            cal[:, 1],
        ]
    ).astype(np.float32)

    pred_buy = data["buy_da"].copy()
    pred_sell = data["sell_da"].copy()
    for t in range(L - 1, n - H):
        yhat = model.predict_eps(feats[t - L + 1 : t + 1])
        pred_buy[t + 1] = max(data["buy_da"][t + 1] + float(yhat[0]), 0.05)
        pred_sell[t + 1] = min(
            max(data["sell_da"][t + 1] + float(yhat[1]), 0.02),
            pred_buy[t + 1] - 0.02,
        )
    pred_buy[:L] = data["buy_da"][:L]
    pred_sell[:L] = data["sell_da"][:L]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "buy_yuan_per_kwh", "sell_yuan_per_kwh"])
        for i in range(n):
            w.writerow([i * 3600, f"{pred_buy[i]:.6f}", f"{pred_sell[i]:.6f}"])

    realized_path = out_path.with_name("price_realized.csv")
    with realized_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "buy_yuan_per_kwh", "sell_yuan_per_kwh"])
        for i in range(n):
            w.writerow(
                [i * 3600, f"{data['buy_realized'][i]:.6f}", f"{data['sell_realized'][i]:.6f}"]
            )

    mae_b = float(np.mean(np.abs(pred_buy[L:] - data["buy_realized"][L:])))
    mae_s = float(np.mean(np.abs(pred_sell[L:] - data["sell_realized"][L:])))
    mae_da_b = float(np.mean(np.abs(data["buy_da"][L:] - data["buy_realized"][L:])))
    return {
        "mae_pred_buy": mae_b,
        "mae_pred_sell": mae_s,
        "mae_da_buy": mae_da_b,
        "improvement_vs_da_buy": mae_da_b - mae_b,
        "predicted_csv": str(out_path),
        "realized_csv": str(realized_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/price_residual_series.csv")
    ap.add_argument("--ckpt", default="data/forecast_models/price_bilstm.pt")
    ap.add_argument("--pred-out", default="data/price_predicted.csv")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lookback", type=int, default=168)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    data = read_residual_csv(ROOT / args.data)
    cfg = PriceBiLSTMConfig(
        lookback=args.lookback, horizon=args.horizon, epochs=args.epochs, device=args.device
    )
    model = PriceBiLSTM(cfg)
    X, y = model.build_arrays(data["buy_da"], data["sell_da"], data["eps_buy"], data["eps_sell"])
    hist = model.fit(X, y)
    ckpt = ROOT / args.ckpt
    model.save(ckpt)
    metrics = export_predictions(model, data, ROOT / args.pred_out)
    summary = {"train": hist, "export": metrics, "checkpoint": str(ckpt)}
    ckpt.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
