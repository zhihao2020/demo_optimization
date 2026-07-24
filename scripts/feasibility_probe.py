"""Phase D.5 可行性探针：收集 FailureRecord、残差统计并写入 runs/feasibility_probe/。"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from envs.power_system_env import PowerSystemEnv
from training.hybrid_td3.buffer import FilteredReplayBuffer, SafetyDataset
from training.hybrid_td3.collector import ValidTransitionCollector
from training.hybrid_td3.train import RandomFeasiblePolicy


def residual_stats(residuals: list[dict], key: str) -> dict:
    """计算单字段残差的分位数摘要。

    Args:
        residuals: 含残差字段的行列表。
        key: 残差字段名(key)。

    Returns:
        含 n、mean、median、p90/p95/p99、max、min 的字典；无样本时 {"n": 0}。
    """
    vals = [float(r[key]) for r in residuals if key in r and np.isfinite(r[key])]
    if not vals:
        return {"n": 0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


def main(n_valid: int = 1000, seed: int = 0, run_dir: str = "runs/feasibility_probe") -> dict:
    """运行随机可行策略收集有效转移、失败记录与残差摘要。

    Args:
        n_valid: 目标有效转移数(n_valid)。
        seed: 初始 reset 种子。
        run_dir: 输出目录相对路径。

    Returns:
        含 residual_summary、suggested_margins 等的 summary 字典。
    """
    out = ROOT / run_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "train").mkdir(exist_ok=True)
    env = PowerSystemEnv(run_id="feasibility_probe")
    buf = FilteredReplayBuffer(capacity=50_000)
    safety = SafetyDataset()
    collector = ValidTransitionCollector(buf, safety_dataset=safety)
    policy = RandomFeasiblePolicy(env)
    oracle = env.oracle
    obs, _ = env.reset(seed=seed)
    valid = 0
    episode = 0
    residual_rows: list[dict] = []
    while valid < n_valid:
        try:
            action = policy.predict(obs)
        except Exception:
            episode += 1
            obs, _ = env.reset(seed=seed + episode)
            continue
        obs, reward, term, trunc, info = collector.step_and_store(env, action)
        if info.get("transition_valid"):
            valid += 1
            if info.get("residuals"):
                residual_rows.append(
                    {
                        **info["residuals"],
                        "mode": int(info.get("requested_caes_mode", 1)),
                        "u_battery": float(info.get("requested_u_battery", 0.0)),
                        "obs_battery_soc": float((info.get("observations") or {}).get("battery_soc", np.nan)),
                        "obs_caes_gas_soc": float((info.get("observations") or {}).get("caes_gas_soc", np.nan)),
                    }
                )
        if term or trunc:
            episode += 1
            obs, _ = env.reset(seed=seed + episode)
    failures = [r.to_dict() for r in env.failure_records]
    (out / "train" / "failure_records.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    safety.save(out / "train" / "safety_dataset.json")
    fine_counts = Counter(f.get("fine_failure_type", "unknown") for f in failures)
    mode_fail = Counter(int((f.get("hybrid_action") or {}).get("caes_mode", -1)) for f in failures)
    keys = [
        "battery_soc",
        "caes_gas_soc",
        "caes_hot_soc",
        "caes_cold_soc",
        "caes_gas_pressure",
        "caes_gas_temperature",
        "p_thermal",
        "p_grid",
    ]
    residual_summary = {k: residual_stats(residual_rows, k) for k in keys}
    by_mode = {}
    for mode in (0, 1, 2):
        subset = [r for r in residual_rows if r.get("mode") == mode]
        by_mode[mode] = {k: residual_stats(subset, k) for k in keys}
    bat_high = [r["battery_soc"] for r in residual_rows if r.get("u_battery", 0) > 0.05 and r.get("battery_soc", 0) is not None]
    bat_low = [r["battery_soc"] for r in residual_rows if r.get("u_battery", 0) < -0.05]
    suggested = {
        "battery_residual_p99_charge_high": float(np.percentile(bat_high, 99)) if bat_high else None,
        "battery_residual_p99_discharge_low_abs": float(np.percentile(np.abs(bat_low), 99)) if bat_low else None,
        "caes_gas_residual_p99_abs": float(np.percentile(np.abs([r["caes_gas_soc"] for r in residual_rows if "caes_gas_soc" in r]), 99))
        if residual_rows
        else None,
    }
    summary = {
        "n_valid": valid,
        "episodes": episode,
        "post_step_failures": collector.stats["post_step_constraint_failures"],
        "precheck_rejections": collector.stats["precheck_rejections"],
        "fine_failure_counts": dict(fine_counts),
        "mode_failure_counts": {str(k): v for k, v in mode_fail.items()},
        "collector_stats": collector.stats,
        "oracle_version": oracle.oracle_version,
        "residual_summary": residual_summary,
        "residual_by_mode": {str(k): v for k, v in by_mode.items()},
        "suggested_margins_from_p99": suggested,
        "failure_rate": collector.stats["post_step_constraint_failures"] / max(valid, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "residual_summary.yaml").write_text(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False), encoding="utf-8")
    env.close()
    print(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False))
    return summary


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    main(n_valid=n)
