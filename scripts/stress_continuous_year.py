#!/usr/bin/env python
"""连续年边界压力测试：随机策略 + 可行集投影，跑满 8760 h。

为什么需要它：B0 规则策略几乎不碰 CAES 边界，能跑完全年并不能证明投影层正确。
这里的策略只从 ``env.get_feasible_action_spec()`` 返回的可行集里采样，且刻意偏向
CAES 动作，所以「能否跑完全年」完全取决于投影层的一步预测是否正确。

验收判据：``fmu_failure_count == 0`` 且无 PostStepHardConstraintViolation。

``--legacy`` 通过配置把投影层退回修复前的行为（alpha_cold 与 alpha_hot 同号、
冷罐守卫失效、无生存性过滤），用于给出修复前后的对照。用法::

    python scripts/stress_continuous_year.py --hours 8760 --seed 0
    python scripts/stress_continuous_year.py --hours 8760 --seed 0 --legacy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config.paths import apply_process_cache_env, resolve_run_dir  # noqa: E402

apply_process_cache_env()

from actions.caes_u import (  # noqa: E402
    CHARGE_HI,
    CHARGE_LO,
    DISCHARGE_HI,
    DISCHARGE_LO,
    physical_dict,
)
from actions.mode_mask import ModeMask  # noqa: E402
from actions.types import CaesMode  # noqa: E402
from envs.power_system_env import PowerSystemEnv  # noqa: E402
from training.evaluate_td3 import evaluate_continuous_annual_policy  # noqa: E402

# 修复前的 energy_model：alpha_cold 与 alpha_hot 同号是 1276 h 崩溃的根因
LEGACY_ENERGY_MODEL = {
    "E_ref_J": 5.4e12,
    "alpha_gas": 1.0,
    "alpha_hot": 0.35,
    "alpha_cold": 0.35,
    "alpha_pressure_Pa_per_soc": 3.0e6,
    "alpha_temp_K_per_u": 2.0,
}


class BoundaryStressPolicy:
    """在可行集内随机采样，并刻意偏向 CAES 充放电以压迫库存边界。"""

    def __init__(self, env: PowerSystemEnv, seed: int, charge_bias: float):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.charge_bias = float(charge_bias)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> dict[str, Any]:
        _ = obs, deterministic
        feas = self.env.get_feasible_action_spec()
        u_tp = float(self.rng.uniform(feas.u_tp_low, feas.u_tp_high))
        u_bat = float(self.rng.uniform(feas.u_battery_low, feas.u_battery_high))

        mask = feas.mode_mask
        # 优先挑充/放，只有两个方向都被禁时才 idle：最大化边界压力
        choices: list[str] = []
        if mask.charge:
            choices.append("charge")
        if mask.discharge:
            choices.append("discharge")
        if not choices:
            return physical_dict(u_tp, u_bat, 0.0)

        if len(choices) == 2:
            pick = "charge" if self.rng.random() < self.charge_bias else "discharge"
        else:
            pick = choices[0]
        if pick == "charge":
            u_caes = float(self.rng.uniform(CHARGE_LO, CHARGE_HI))
        else:
            u_caes = float(self.rng.uniform(DISCHARGE_LO, DISCHARGE_HI))
        return physical_dict(u_tp, u_bat, u_caes)


def revert_alphas(env: PowerSystemEnv) -> None:
    """缺陷 1：换回符号错误的旧 energy_model，并让冷罐守卫失效。

    冷罐守卫的裕度设为 -1.0，使翻转后的守卫恒真，等效于修复前不守冷罐；
    同时关掉一步生存性过滤。全部通过配置实现，不改动生产代码。
    """
    caes = env.oracle.margins.setdefault("caes", {})
    caes["energy_model"] = dict(LEGACY_ENERGY_MODEL)
    chg = caes.setdefault("charge", {})
    chg["margin_cold"] = 0.0
    chg["residual_p99_cold_low"] = -1.0
    dis = caes.setdefault("discharge", {})
    dis["margin_cold"] = 0.0
    dis["residual_p99_cold_high"] = -1.0


def revert_mask(env: PowerSystemEnv) -> None:
    """缺陷 2：取消幅值子区间，并恢复 mag=1.0 的方向探针语义。

    旧实现用 ``u_from_mode_mag(mode, 1.0)`` 当最坏情况探针：充电取最强端、
    放电取最弱端。这里把方向判定还原为该语义，并把区间放宽回整条合法带，
    从而重现「方向被判合法但带内多数幅值会越界」的不一致。
    """
    oracle = env.oracle
    cm = oracle.margins.get("caes", {})

    def legacy_mask_and_intervals(outputs):
        base = oracle._caes_mode_mask_base(outputs)
        charge_ok = base.charge and oracle._caes_mode_feasible(
            outputs, CaesMode.CHARGE, 1.0, cm.get("charge", {}), "high"
        )
        mask = ModeMask(
            discharge=bool(base.discharge),
            idle=bool(base.idle),
            charge=bool(charge_ok),
        )
        return mask, {
            "charge": (CHARGE_LO, CHARGE_HI) if charge_ok else None,
            "discharge": (DISCHARGE_LO, DISCHARGE_HI) if base.discharge else None,
        }

    oracle._caes_mask_and_intervals = legacy_mask_and_intervals  # type: ignore[method-assign]


def revert_temp(env: PowerSystemEnv) -> None:
    """缺陷 3：把 gas_temp_min_K 重新当作硬约束。"""
    env.oracle.params["caes"]["gas_temp_min_is_numeric_clamp"] = False


def summarize_csv(csv_path: Path) -> dict[str, Any]:
    """从逐步 CSV 抽出库存极值与硬界触碰情况。"""
    import pandas as pd

    if not csv_path.is_file():
        return {}
    df = pd.read_csv(csv_path)
    out: dict[str, Any] = {"rows": int(len(df))}
    for name in ("caes_gas_soc", "caes_hot_soc", "caes_cold_soc", "battery_soc"):
        col = f"obs_{name}"
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            out[f"{name}_min"] = float(series.min())
            out[f"{name}_max"] = float(series.max())
    if "failure_type" in df.columns:
        counts = df["failure_type"].dropna().astype(str)
        counts = counts[counts.str.len() > 0]
        out["failure_types"] = counts.value_counts().to_dict()
    if "transition_valid" in df.columns:
        valid = df["transition_valid"].astype(str).str.lower().isin(("true", "1"))
        out["invalid_steps"] = int((~valid).sum())
    if "decoded_u_caes" in df.columns:
        u = pd.to_numeric(df["decoded_u_caes"], errors="coerce").fillna(0.0)
        out["caes_charge_steps"] = int((u > 1e-9).sum())
        out["caes_discharge_steps"] = int((u < -1e-9).sum())
        out["caes_idle_steps"] = int((u.abs() <= 1e-9).sum())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hours", type=int, default=8760)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--charge-bias",
        type=float,
        default=0.6,
        help="两方向都可行时选充电的概率；>0.5 更快压出冷罐耗尽",
    )
    p.add_argument("--legacy", action="store_true", help="同时退回下面三项")
    p.add_argument("--legacy-alphas", action="store_true", help="缺陷1：冷罐符号与标定")
    p.add_argument("--legacy-mask", action="store_true", help="缺陷2：取消幅值子区间")
    p.add_argument("--legacy-temp", action="store_true", help="缺陷3：温度钳位当硬约束")
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args()

    reverts = {
        "alphas": args.legacy or args.legacy_alphas,
        "mask": args.legacy or args.legacy_mask,
        "temp": args.legacy or args.legacy_temp,
    }
    active = [k for k, v in reverts.items() if v]
    tag = "fixed" if not active else "legacy_" + "_".join(active)
    out_dir = Path(args.out_dir or f"runs/stress_{tag}_s{args.seed}_{args.hours}h")
    out = resolve_run_dir(str(out_dir))
    out.mkdir(parents=True, exist_ok=True)

    env = PowerSystemEnv(
        run_id=f"stress_{tag}_s{args.seed}",
        forecast_enabled=True,
        episode_steps=int(args.hours),
    )
    try:
        if reverts["alphas"]:
            revert_alphas(env)
        if reverts["mask"]:
            revert_mask(env)
        if reverts["temp"]:
            revert_temp(env)
        em = env.oracle.margins["caes"]["energy_model"]
        print(
            f"=== stress {tag} seed={args.seed} hours={args.hours} "
            f"alpha_cold={em.get('alpha_cold')} reverted={active or ['none']} ===",
            flush=True,
        )
        policy = BoundaryStressPolicy(env, args.seed, args.charge_bias)
        t0 = time.perf_counter()
        res = evaluate_continuous_annual_policy(
            env,
            policy,
            annual_horizon_hours=int(args.hours),
            output_dir=out,
            start_time=0.0,
        )
        wall = time.perf_counter() - t0
    finally:
        env.close()

    row = {
        "arm": tag,
        "reverted_defects": active,
        "seed": args.seed,
        "charge_bias": args.charge_bias,
        "requested_hours": int(args.hours),
        "steps": res.get("steps"),
        "valid_steps": res.get("valid_steps"),
        "fmu_failure_count": res.get("fmu_failure_count"),
        "invalid_transition_count": res.get("invalid_transition_count"),
        "survived_full_year": int(res.get("steps") or 0) >= int(args.hours),
        "wall_s": round(wall, 1),
        "trajectory": summarize_csv(out / "continuous_year.csv"),
    }
    (out / "stress_summary.json").write_text(
        json.dumps(row, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps(row, indent=2, ensure_ascii=False, default=str), flush=True)
    verdict = "PASS" if row["survived_full_year"] and not row["fmu_failure_count"] else "FAIL"
    print(f"\n[{verdict}] steps={row['steps']}/{args.hours} "
          f"fmu_fail={row['fmu_failure_count']}", flush=True)


if __name__ == "__main__":
    main()
