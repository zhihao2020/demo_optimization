"""域 B：AI4E 蒙西小时级储能套利（无 Modelica/FMU）。

竞赛风格（小时版）：一天最多一轮——连续 h_chg 小时充电 + 连续 h_dis 小时放电，或空闲。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DaySeries:
    date: str
    price: np.ndarray
    load_cf: np.ndarray
    wind_cf: np.ndarray
    pv_cf: np.ndarray
    load_fc: np.ndarray
    wind_fc: np.ndarray
    pv_fc: np.ndarray
    hour: np.ndarray
    dow: np.ndarray
    month: np.ndarray


def load_day_table(hourly_csv: str) -> dict[str, DaySeries]:
    df = pd.read_csv(hourly_csv, parse_dates=["times"])
    df["date"] = df["times"].dt.date.astype(str)
    out: dict[str, DaySeries] = {}
    for date, g in df.groupby("date"):
        g = g.sort_values("times")
        if len(g) < 24:
            continue
        g = g.iloc[:24]
        out[str(date)] = DaySeries(
            date=str(date),
            price=g["price_rt"].to_numpy(np.float64),
            load_cf=g["load_act_cf"].to_numpy(np.float64),
            wind_cf=g["wind_act_cf"].to_numpy(np.float64),
            pv_cf=g["pv_act_cf"].to_numpy(np.float64),
            load_fc=g["load_fc"].to_numpy(np.float64),
            wind_fc=g["wind_fc"].to_numpy(np.float64),
            pv_fc=g["pv_fc"].to_numpy(np.float64),
            hour=g["hour"].to_numpy(np.int32) if "hour" in g else np.arange(24, dtype=np.int32),
            dow=g["dow"].to_numpy(np.int32) if "dow" in g else np.zeros(24, dtype=np.int32),
            month=g["month"].to_numpy(np.int32) if "month" in g else np.ones(24, dtype=np.int32),
        )
    return out


def simulate_day_power(
    price: np.ndarray,
    power: np.ndarray,
    *,
    e_cap: float = 2.0,
) -> dict[str, Any]:
    price = np.asarray(price, dtype=np.float64)
    power = np.asarray(power, dtype=np.float64)
    # power: negative=charge (buy), positive=discharge (sell)
    # SOC increases when charging: ΔSOC = -power / e_cap
    soc = 0.0
    socs = []
    feasible = True
    for t in range(len(power)):
        p = float(power[t])
        if p < -1e-9 and soc >= 1.0 - 1e-9:  # charge but full
            feasible = False
        if p > 1e-9 and soc <= 1e-9:  # discharge but empty
            feasible = False
        soc = float(np.clip(soc - p / e_cap, 0.0, 1.0))
        socs.append(soc)
    revenue = float(np.dot(price, power))
    return {
        "revenue": revenue if feasible else float("-inf"),
        "revenue_raw": revenue,
        "feasible": feasible,
        "soc_traj": np.asarray(socs, dtype=np.float64),
        "power": power,
    }


def build_block_power(
    charge_start: int,
    discharge_start: int,
    *,
    h_chg: int = 2,
    h_dis: int = 2,
    p_rate: float = 1.0,
    n: int = 24,
) -> np.ndarray | None:
    if charge_start < 0 or discharge_start < 0:
        return np.zeros(n, dtype=np.float64)
    if charge_start + h_chg > n or discharge_start + h_dis > n:
        return None
    if discharge_start < charge_start + h_chg:
        return None
    power = np.zeros(n, dtype=np.float64)
    power[charge_start : charge_start + h_chg] = -p_rate
    power[discharge_start : discharge_start + h_dis] = p_rate
    return power


def oracle_best_windows(
    price: np.ndarray,
    *,
    h_chg: int = 2,
    h_dis: int = 2,
    p_rate: float = 1.0,
    e_cap: float = 2.0,
) -> dict[str, Any]:
    n = len(price)
    best: dict[str, Any] = {
        "revenue": 0.0,
        "charge_start": -1,
        "discharge_start": -1,
        "power": np.zeros(n),
        "feasible": True,
    }
    for cs in range(0, n - h_chg + 1):
        for ds in range(cs + h_chg, n - h_dis + 1):
            power = build_block_power(cs, ds, h_chg=h_chg, h_dis=h_dis, p_rate=p_rate, n=n)
            if power is None:
                continue
            sim = simulate_day_power(price, power, e_cap=e_cap)
            if not sim["feasible"]:
                continue
            if sim["revenue"] > best["revenue"]:
                best = {
                    "revenue": sim["revenue"],
                    "charge_start": cs,
                    "discharge_start": ds,
                    "power": power,
                    "feasible": True,
                }
    return best


def rule_from_price_path(
    price_hat: np.ndarray,
    *,
    h_chg: int = 2,
    h_dis: int = 2,
    p_rate: float = 1.0,
    e_cap: float = 2.0,
    min_spread: float = 0.0,
) -> dict[str, Any]:
    plan = oracle_best_windows(
        price_hat, h_chg=h_chg, h_dis=h_dis, p_rate=p_rate, e_cap=e_cap
    )
    if plan["charge_start"] < 0:
        return plan
    cs, ds = int(plan["charge_start"]), int(plan["discharge_start"])
    chg = float(price_hat[cs : cs + h_chg].mean())
    dis = float(price_hat[ds : ds + h_dis].mean())
    if dis - chg < min_spread:
        return {
            "revenue": 0.0,
            "charge_start": -1,
            "discharge_start": -1,
            "power": np.zeros(len(price_hat)),
            "feasible": True,
            "skipped_low_spread": True,
        }
    return plan
