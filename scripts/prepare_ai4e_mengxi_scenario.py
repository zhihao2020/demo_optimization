#!/usr/bin/env python
"""将 AI4E 蒙西竞赛数据预处理为小时级域 B 场景表。

输入（默认初赛 train）:
  E:/ai4Science/AI4E/初赛/src/data/train/mengxi_boundary_anon_filtered.csv
  E:/ai4Science/AI4E/初赛/src/data/train/mengxi_node_price_selected.csv

输出:
  data/ai4e_mengxi/hourly_merged.csv
  data/ai4e_mengxi/meta.json
  data/ai4e_mengxi/splits.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

COL_MAP = {
    "系统负荷实际值": "load_act",
    "系统负荷预测值": "load_fc",
    "风光总加实际值": "re_act",
    "风光总加预测值": "re_fc",
    "联络线实际值": "tie_act",
    "联络线预测值": "tie_fc",
    "风电实际值": "wind_act",
    "风电预测值": "wind_fc",
    "光伏实际值": "pv_act",
    "光伏预测值": "pv_fc",
    "水电实际值": "hydro_act",
    "水电预测值": "hydro_fc",
    "非市场化机组实际值": "nonmkt_act",
    "非市场化机组预测值": "nonmkt_fc",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--boundary",
        type=str,
        default=r"E:/ai4Science/AI4E/初赛/src/data/train/mengxi_boundary_anon_filtered.csv",
    )
    ap.add_argument(
        "--price",
        type=str,
        default=r"E:/ai4Science/AI4E/初赛/src/data/train/mengxi_node_price_selected.csv",
    )
    ap.add_argument("--out-dir", type=str, default="data/ai4e_mengxi")
    ap.add_argument("--train-end-month", type=int, default=9)
    args = ap.parse_args()

    bpath, ppath = Path(args.boundary), Path(args.price)
    if not bpath.is_file() or not ppath.is_file():
        raise FileNotFoundError(f"missing AI4E files:\n  {bpath}\n  {ppath}")

    boundary = pd.read_csv(bpath, parse_dates=["times"]).rename(columns=COL_MAP)
    price = pd.read_csv(ppath, parse_dates=["times"]).rename(columns={"A": "price_rt"})
    df = boundary.merge(price, on="times", how="inner").sort_values("times")
    df = df.drop_duplicates("times")
    for c in ("wind_act", "pv_act", "re_act", "wind_fc", "pv_fc", "re_fc"):
        if c in df.columns:
            df[c] = df[c].clip(lower=0.0)

    df = df.set_index("times")
    num_cols = list(df.columns)
    hourly = df[num_cols].resample("1h").mean().dropna(how="any").reset_index()
    hourly["date"] = hourly["times"].dt.date.astype(str)
    hourly["hour"] = hourly["times"].dt.hour.astype(int)
    hourly["month"] = hourly["times"].dt.month.astype(int)
    hourly["dow"] = hourly["times"].dt.dayofweek.astype(int)

    train_mask = hourly["month"] <= int(args.train_end_month)
    scales: dict[str, float] = {}
    for c in ("wind_act", "pv_act", "load_act", "re_act"):
        m = float(hourly.loc[train_mask, c].quantile(0.995))
        m = max(m, 1e-6)
        scales[c] = m
        hourly[f"{c}_cf"] = (hourly[c] / m).clip(0.0, 1.5)

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    out_csv = out / "hourly_merged.csv"
    hourly.to_csv(out_csv, index=False)

    train_days = sorted(hourly.loc[train_mask, "date"].unique())
    test_days = sorted(hourly.loc[~train_mask, "date"].unique())
    splits = {
        "train_end_month": int(args.train_end_month),
        "train_days": train_days,
        "test_days": test_days,
        "n_train_days": len(train_days),
        "n_test_days": len(test_days),
        "n_hours": int(len(hourly)),
    }
    (out / "splits.json").write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "source": "AI4E World Scientific Intelligence Competition (AI+Energy)",
        "region": "Mengxi system-level anonymized",
        "year": 2025,
        "native_resolution": "15min",
        "export_resolution": "1h mean",
        "price": "real-time nodal price A",
        "boundary_path": str(bpath),
        "price_path": str(ppath),
        "scales_train_q995": scales,
        "note": (
            "Domain B only — not FMU plant physics. "
            "Cross-domain robustness of forecast-then-constrained storage scheduling. "
            "Acknowledge AI4E organizers and TJ Weather (NWP) in paper."
        ),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "out_csv": str(out_csv),
                "n_hours": len(hourly),
                "n_train_days": len(train_days),
                "n_test_days": len(test_days),
                "price_mean": float(hourly["price_rt"].mean()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
