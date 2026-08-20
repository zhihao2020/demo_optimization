"""从已有轨迹回归 CAES 三个库存的一步转移系数。

背景：``src/config/feasibility_margins.yaml`` 里 ``energy_model`` 的
``alpha_gas / alpha_hot / alpha_cold`` 是手填的，其中 alpha_cold 与 alpha_hot 同号。
物理上充电是「冷罐 -> 热罐」（压缩热存入热罐，冷罐供冷却水），放电反向，
所以冷罐系数应当与热罐反号。本脚本用真实轨迹把三个系数拟合出来。

用法::

    python scripts/fit_caes_transition.py
    python scripts/fit_caes_transition.py --csv runs/continuous_annual_b0/continuous_year.csv

输出每个库存在充电段 / 放电段各自的斜率，以及合并拟合值，可直接抄进
``feasibility_margins.yaml``。
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

# 与 FeasibilityOracle.predict_next_state 保持同一无量纲化：
# energy = u_caes * P_cap * dt / E_ref
P_CAP_W = 1.5e8
DT_S = 3600.0
E_REF_J = 5.4e12

INVENTORIES = ("gas", "hot", "cold")
SOC_COLUMNS = {name: f"obs_caes_{name}_soc" for name in INVENTORIES}

DEFAULT_GLOBS = (
    "runs/continuous_annual_*/continuous_year.csv",
    "runs/paper_dispatch_traj/*.csv",
)


def load_steps(paths: list[Path]) -> pd.DataFrame:
    """把多个轨迹 CSV 拼成逐步转移样本。

    每行 CSV 是「执行该步动作之后」的状态，因此第 i 步的增量为 obs[i] - obs[i-1]，
    对应动作是第 i 行的 decoded_u_caes。跨文件不做差分。
    """
    frames = []
    for path in paths:
        try:
            raw = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            print(f"  跳过 {path}: {exc}")
            continue
        needed = ["decoded_u_caes", *SOC_COLUMNS.values()]
        if any(col not in raw.columns for col in needed):
            print(f"  跳过 {path}: 缺少所需列")
            continue
        frame = raw[needed].apply(pd.to_numeric, errors="coerce")
        if "transition_valid" in raw.columns:
            valid = raw["transition_valid"].astype(str).str.lower().isin(("true", "1"))
        else:
            valid = pd.Series(True, index=raw.index)

        block = pd.DataFrame({"u_caes": frame["decoded_u_caes"]})
        for name, col in SOC_COLUMNS.items():
            block[f"d_{name}"] = frame[col].diff()
        # 只保留自身与前一步都是有效转移的样本
        block = block[valid & valid.shift(fill_value=False)]
        block = block.dropna()
        block["source"] = str(path)
        frames.append(block)
        print(f"  {path}: {len(block)} 个可用转移")

    if not frames:
        raise SystemExit("没有可用轨迹，检查 --csv 或 runs/ 下的产物")
    return pd.concat(frames, ignore_index=True)


def fit_slope(energy: np.ndarray, delta: np.ndarray) -> tuple[float, float, int]:
    """过原点最小二乘 delta = alpha * energy，返回 (alpha, R^2, 样本数)。"""
    if energy.size < 2 or not np.any(np.abs(energy) > 1e-12):
        return float("nan"), float("nan"), int(energy.size)
    alpha = float(energy @ delta / (energy @ energy))
    resid = delta - alpha * energy
    ss_tot = float(delta @ delta)
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    return alpha, r2, int(energy.size)


def report(samples: pd.DataFrame) -> dict[str, float]:
    energy_all = samples["u_caes"].to_numpy() * P_CAP_W * DT_S / E_REF_J
    charge = samples["u_caes"].to_numpy() > 1e-9
    discharge = samples["u_caes"].to_numpy() < -1e-9

    print(f"\n样本总数 {len(samples)}；充电 {int(charge.sum())}，放电 {int(discharge.sum())}，"
          f"idle {int((~charge & ~discharge).sum())}")
    print("\nenergy = u_caes * P_cap * dt / E_ref，拟合 d_soc = alpha * energy\n")
    header = f"{'库存':<6}{'充电 alpha':>14}{'放电 alpha':>14}{'合并 alpha':>14}{'合并 R^2':>12}{'样本':>8}"
    print(header)
    print("-" * len(header.encode("gbk", errors="replace").decode("gbk")))

    fitted: dict[str, float] = {}
    for name in INVENTORIES:
        delta = samples[f"d_{name}"].to_numpy()
        a_chg, _, _ = fit_slope(energy_all[charge], delta[charge])
        a_dis, _, _ = fit_slope(energy_all[discharge], delta[discharge])
        moving = charge | discharge
        a_all, r2_all, n_all = fit_slope(energy_all[moving], delta[moving])
        fitted[name] = a_all
        print(f"{name:<6}{a_chg:>14.4f}{a_dis:>14.4f}{a_all:>14.4f}{r2_all:>12.4f}{n_all:>8d}")

    # idle 段的自发漂移：决定「冷罐在待机时是否也在漏」
    idle = samples[~charge & ~discharge]
    if len(idle):
        print("\nidle 段每小时自发漂移（均值 / 中位数）：")
        for name in INVENTORIES:
            col = idle[f"d_{name}"]
            print(f"  {name:<6}mean={col.mean():+.6f}  median={col.median():+.6f}")
    return fitted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        action="append",
        default=None,
        help="轨迹 CSV 路径，可重复；默认扫描 runs/ 下的连续年与调度轨迹",
    )
    args = parser.parse_args()

    if args.csv:
        paths = [Path(p) for p in args.csv]
    else:
        paths = sorted({Path(p) for pattern in DEFAULT_GLOBS for p in glob.glob(pattern)})

    print("读取轨迹：")
    samples = load_steps(paths)
    fitted = report(samples)

    print("\n可写入 src/config/feasibility_margins.yaml 的 energy_model：")
    print("  energy_model:")
    print(f"    E_ref_J: {E_REF_J:.1e}")
    for name in INVENTORIES:
        print(f"    alpha_{name}: {fitted[name]:.3f}")

    if fitted["cold"] * fitted["hot"] < 0:
        print("\n结论：冷罐与热罐系数反号，证实充电抽冷罐/放电回灌冷罐。")
        print("      现有配置中 alpha_cold 与 alpha_hot 同号，是连续年在约 1276 h 崩溃的直接原因。")
    else:
        print("\n注意：冷罐与热罐系数未反号，需要人工复核后再改投影层。")


if __name__ == "__main__":
    main()
