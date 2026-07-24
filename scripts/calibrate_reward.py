"""用规则控制器标定 C_ref 与终端 SOC bonus/tolerance，并写回 reward_config.yaml。"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from controllers.rule_based_controller import RuleBasedController
from envs.power_system_env import PowerSystemEnv


def run_rule_costs(seeds=(0, 1, 2), max_steps=168) -> tuple[list[float], list[float], list[dict]]:
    """用规则控制器 rollout 收集逐步 raw 成本与 episode 摘要。

    Args:
        seeds: reset 随机种子列表。
        max_steps: 每 episode 最大步数。

    Returns:
        (raw_costs, week_norms, episodes) 三元组。
    """
    raw_costs: list[float] = []
    week_norms: list[float] = []
    soc_errors: list[float] = []
    episodes: list[dict] = []
    for seed in seeds:
        with PowerSystemEnv() as env:
            # 临时：若 C_ref 为空，RewardCalculator 用 1.0 作分母，raw 仍正确
            ctrl = RuleBasedController(env)
            obs, info = env.reset(seed=seed)
            week_raw = 0.0
            steps = 0
            for _ in range(max_steps):
                action = ctrl.predict(obs)
                obs, reward, terminated, truncated, step_info = env.step(action)
                terms = step_info.get("reward_terms") or {}
                if step_info.get("transition_valid"):
                    raw_costs.append(float(terms.get("raw_total_cost", 0.0)))
                    week_raw += float(terms.get("raw_total_cost", 0.0))
                    steps += 1
                if terminated or truncated:
                    err = float(terms.get("terminal_soc_l1_error", 0.0))
                    soc_errors.append(err)
                    episodes.append(
                        {
                            "seed": seed,
                            "valid_steps": steps,
                            "week_raw": week_raw,
                            "terminal_soc_l1": err,
                            "failure": step_info.get("failure_type"),
                        }
                    )
                    break
            # 若 C_ref 尚未设置，normalized_cost ≈ raw
            week_norms.append(week_raw)
    return raw_costs, week_norms, episodes


def main() -> None:
    """标定 C_ref 与终端 SOC bonus/tolerance 并写回 reward_config.yaml。

    Raises:
        SystemExit: 规则轨迹未产生任何有效成本样本。
    """
    raw_costs, week_norms, episodes = run_rule_costs()
    if not raw_costs:
        raise SystemExit("规则轨迹未产生有效成本样本")
    abs_costs = np.abs(np.asarray(raw_costs, dtype=np.float64))
    c_ref = float(np.percentile(abs_costs, 95))
    if c_ref <= 0:
        # 全零成本时使用火电满发一小时量级作为防护性分母
        c_ref = 400.0 * 150.0  # 元/MWh * MW * 1h
    mean_week = float(np.mean(week_norms))
    # bonus ≈ 5%–30% 一周成本量级；取 15%
    bonus = abs(mean_week / c_ref) * 0.15 if c_ref > 0 else 1.0
    # tolerance：规则轨迹 L1 误差的 medium 档（P75 与 0.05 取大）
    errs = [e["terminal_soc_l1"] for e in episodes if e.get("terminal_soc_l1") is not None]
    tol_medium = float(max(0.05, np.percentile(errs, 75))) if errs else 0.1
    tol_strict = float(max(0.02, np.percentile(errs, 25))) if errs else 0.05
    tol_loose = float(max(0.1, np.percentile(errs, 90))) if errs else 0.2

    cfg_path = ROOT / "src" / "config" / "reward_config.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["cost_reference"] = {
        "value": c_ref,
        "unit": "CNY_per_step",
        "source": "rule_controller_p95",
        "computed_at": str(date.today()),
        "trajectory": "scripts/calibrate_reward.py seeds=0,1,2",
        "n_samples": len(raw_costs),
        "mean_week_raw_cost": mean_week,
    }
    cfg["terminal_soc"]["bonus"] = float(bonus)
    cfg["terminal_soc"]["tolerance"] = tol_medium
    cfg["terminal_soc"]["tolerance_strict"] = tol_strict
    cfg["terminal_soc"]["tolerance_loose"] = tol_loose
    cfg["terminal_soc"]["mode"] = "binary_bonus"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    out = ROOT / "runs" / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(episodes[0].keys()))
        w.writeheader()
        w.writerows(episodes)
    summary = {
        "C_ref": c_ref,
        "bonus": bonus,
        "tolerance_medium": tol_medium,
        "tolerance_strict": tol_strict,
        "tolerance_loose": tol_loose,
        "episodes": episodes,
    }
    (out / "summary.yaml").write_text(yaml.safe_dump(summary, allow_unicode=True), encoding="utf-8")
    print(yaml.safe_dump(summary, allow_unicode=True))


if __name__ == "__main__":
    main()
