#!/usr/bin/env python
"""由山东分时表构造「日前 DA 价 + 实现价 + 残差」序列，供 BiLSTM 训练。

实现价 = DA 分时到户价 + 残差 ε。
ε 模拟市场化扰动：AR(1) + 峰段更大波动 + 弱负荷相关。
结算用 realized；策略观测可用 predicted = DA + ε_hat。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _read_tou(path: Path):
    times, buy, sell, bands = [], [], [], []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            times.append(float(row["time"]))
            buy.append(float(row["buy_yuan_per_kwh"]))
            sell.append(float(row["sell_yuan_per_kwh"]))
            bands.append(str(row.get("band", "F")))
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(buy, dtype=np.float64),
        np.asarray(sell, dtype=np.float64),
        bands,
    )


def _load_optional(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    vals = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            vals.append(float(row["value"]))
    return np.asarray(vals, dtype=np.float64)


def make_residual(n, bands, *, seed, sigma_base, rho, load):
    rng = np.random.default_rng(seed)
    scale = {"S": 1.8, "P": 1.4, "F": 1.0, "V": 0.7, "D": 0.5}
    eps = np.zeros(n, dtype=np.float64)
    z = 0.0
    load_z = None
    if load is not None and load.size >= n:
        x = load[:n]
        load_z = (x - x.mean()) / (x.std() + 1e-9)
    for t in range(n):
        sig = sigma_base * scale.get(bands[t], 1.0)
        innov = rng.normal(0.0, sig)
        if load_z is not None:
            innov += 0.02 * float(load_z[t])
        z = rho * z + innov
        eps[t] = z
    day = n // 24
    for d in range(day):
        sl = slice(d * 24, (d + 1) * 24)
        eps[sl] -= 0.3 * eps[sl].mean()
    return eps


def main() -> None:
    p = argparse.ArgumentParser(description="构造电价 DA/realized/residual 序列")
    p.add_argument("--tou", type=str, default="data/price_tou.csv")
    p.add_argument("--out", type=str, default="data/price_residual_series.csv")
    p.add_argument("--meta-out", type=str, default="data/price_residual_meta.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sigma", type=float, default=0.04)
    p.add_argument("--rho", type=float, default=0.85)
    args = p.parse_args()

    times, buy_da, sell_da, bands = _read_tou(ROOT / args.tou)
    n = len(times)
    load = _load_optional(ROOT / "data" / "load.csv")
    eps_buy = make_residual(n, bands, seed=args.seed, sigma_base=args.sigma, rho=args.rho, load=load)
    eps_sell = make_residual(
        n, bands, seed=args.seed + 7, sigma_base=args.sigma * 0.5, rho=args.rho, load=load
    )
    buy_rt = np.maximum(buy_da + eps_buy, 0.05)
    sell_rt = np.clip(sell_da + eps_sell, 0.02, buy_rt - 0.02)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "time",
                "band",
                "buy_da",
                "sell_da",
                "buy_realized",
                "sell_realized",
                "eps_buy",
                "eps_sell",
            ]
        )
        for i in range(n):
            w.writerow(
                [
                    int(times[i]),
                    bands[i],
                    f"{buy_da[i]:.6f}",
                    f"{sell_da[i]:.6f}",
                    f"{buy_rt[i]:.6f}",
                    f"{sell_rt[i]:.6f}",
                    f"{eps_buy[i]:.6f}",
                    f"{eps_sell[i]:.6f}",
                ]
            )

    meta = {
        "source_tou": str(args.tou),
        "n_hours": n,
        "seed": args.seed,
        "sigma_base": args.sigma,
        "rho": args.rho,
        "formula": "realized = max(da + eps, floor); eps ~ AR(1) scaled by band",
        "eps_buy_std": float(eps_buy.std()),
        "eps_sell_std": float(eps_sell.std()),
        "buy_realized_mean": float(buy_rt.mean()),
    }
    (ROOT / args.meta_out).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(out), **meta}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
