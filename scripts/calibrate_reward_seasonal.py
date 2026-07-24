"""跨季节 / 场景周采样重标定 C_ref（start_time 偏移覆盖 8760h FMU）。"""
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
# 代表性周起点（秒）：冬 / 夏 / 春秋 + 可选高低风光周
SEASONAL_STARTS = {
    "winter_week": 0,  # 1 月初
    "spring_week": 90 * 24 * 3600,
    "summer_week": 180 * 24 * 3600,
    "autumn_week": 270 * 24 * 3600,
    "high_wind_proxy": 30 * 24 * 3600,
    "low_wind_proxy": 210 * 24 * 3600,
}
def run_week(start_time: float, seed: int = 0, max_steps: int = 168) -> dict:
    """在指定 FMU 起始时刻运行规则控制器一周。

    Args:
        start_time: episode 起始 simulation_time(秒)。
        seed: env.reset 种子。
        max_steps: 最大步数。

    Returns:
        含 week_raw、valid_steps、terminal_soc_l1 等的 episode 字典。
    """
    with PowerSystemEnv() as env:
        ctrl = RuleBasedController(env)
        obs, info = env.reset(seed=seed, options={"start_time": float(start_time)})
        raw_costs = []
        week_raw = 0.0
        steps = 0
        last_info = info
        for _ in range(max_steps):
            action = ctrl.predict(obs)
            obs, reward, terminated, truncated, step_info = env.step(action)
            last_info = step_info
            terms = step_info.get("reward_terms") or {}
            if step_info.get("transition_valid"):
                c = float(terms.get("raw_total_cost", 0.0))
                raw_costs.append(c)
                week_raw += c
                steps += 1
            if terminated or truncated:
                break
        return {
            "start_time": start_time,
            "valid_steps": steps,
            "week_raw": week_raw,
            "raw_costs": raw_costs,
            "terminal_soc_l1": float((last_info.get("reward_terms") or {}).get("terminal_soc_l1_error", 0.0)),
            "failure": last_info.get("failure_type"),
        }
def main(update_config: bool = False) -> dict:
    """跨季节周采样重标定 C_ref，可选写回 reward_config.yaml。

    Args:
        update_config: 为 True 且变化 >5% 时更新配置文件。

    Returns:
        标定摘要字典。
    """
    episodes = []
    all_costs: list[float] = []
    for name, start in SEASONAL_STARTS.items():
        ep = run_week(start, seed=0)
        ep["scenario"] = name
        # 不把完整 raw_costs 塞进 yaml
        costs = ep.pop("raw_costs")
        all_costs.extend(costs)
        episodes.append(ep)
        print(f"{name}: steps={ep['valid_steps']} week_raw={ep['week_raw']:.2f} fail={ep['failure']}")
    abs_costs = np.abs(np.asarray(all_costs, dtype=np.float64))
    c_ref_new = float(np.percentile(abs_costs, 95)) if len(abs_costs) else None
    mean_week = float(np.mean([e["week_raw"] for e in episodes])) if episodes else None
    cfg_path = ROOT / "src" / "config" / "reward_config.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    c_ref_old = float((cfg.get("cost_reference") or {}).get("value") or 0.0)
    material = c_ref_new is not None and (c_ref_old <= 0 or abs(c_ref_new - c_ref_old) / max(c_ref_old, 1.0) > 0.05)
    summary = {
        "C_ref_old": c_ref_old,
        "C_ref_new_p95": c_ref_new,
        "mean_week_raw_cost": mean_week,
        "n_step_samples": len(all_costs),
        "material_change_gt_5pct": material,
        "episodes": episodes,
        "computed_at": str(date.today()),
        "updated_config": False,
    }
    if update_config and material and c_ref_new:
        bonus = abs(mean_week / c_ref_new) * 0.15 if c_ref_new > 0 else cfg["terminal_soc"]["bonus"]
        cfg["cost_reference"] = {
            "value": c_ref_new,
            "unit": "CNY_per_step",
            "source": "rule_controller_seasonal_p95",
            "computed_at": str(date.today()),
            "trajectory": "scripts/calibrate_reward_seasonal.py",
            "n_samples": len(all_costs),
            "mean_week_raw_cost": mean_week,
            "previous_value": c_ref_old,
        }
        cfg["terminal_soc"]["bonus"] = float(bonus)
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        summary["updated_config"] = True
        summary["bonus_new"] = float(bonus)
    out = ROOT / "runs" / "calibration_seasonal"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.yaml").write_text(yaml.safe_dump(summary, allow_unicode=True), encoding="utf-8")
    with (out / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(episodes[0].keys()))
        w.writeheader()
        w.writerows(episodes)
    # docs note
    doc = ROOT / "docs" / "cref_seasonal_recalibration.md"
    doc.write_text(
        "\n".join(
            [
                "# C_ref 跨季节重标定",
                "",
                f"- 旧 C_ref（单周 3 种子）: {c_ref_old:.4f}",
                f"- 新 C_ref（季节周 P95）: {c_ref_new:.4f}" if c_ref_new else "- 新 C_ref: n/a",
                f"- 相对变化 >5%: {material}",
                f"- 配置已更新: {summary['updated_config']}",
                f"- 样本步数: {len(all_costs)}",
                "",
                "场景周起点见 `scripts/calibrate_reward_seasonal.py` 中 `SEASONAL_STARTS`。",
                "若差异不显著，保持冻结旧值；否则以季节 P95 回写 `reward_config.yaml`。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(yaml.safe_dump(summary, allow_unicode=True))
    return summary
if __name__ == "__main__":
    update = "--update" in sys.argv
    main(update_config=update)
