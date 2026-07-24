"""绘制一周 SOC 轨迹。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    """绘制轨迹 CSV 中四类 SOC 随 step 变化图。

    Raises:
        SystemExit: CSV 路径无效或 pandas 读入失败。
    """
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    series = [
        ("obs_battery_soc", "battery"),
        ("obs_caes_gas_soc", "caes_gas"),
        ("obs_caes_hot_soc", "caes_hot"),
        ("obs_caes_cold_soc", "caes_cold"),
    ]
    for ax, (col, title) in zip(axes.ravel(), series):
        if col not in df.columns:
            continue
        y = df[col].astype(float)
        ax.plot(df["step"], y, label="SOC")
        ax.axhline(y.iloc[0], color="C1", ls="--", label="initial")
        ax.axhline(y.iloc[-1], color="C2", ls=":", label="final")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = args.out or args.csv.with_suffix(".soc.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
