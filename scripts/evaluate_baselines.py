"""对比规则与混合随机可行策略（不再启动普通 Box TD3）。"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from controllers.rule_based_controller import RuleBasedController
from envs.power_system_env import PowerSystemEnv
from training.evaluate_td3 import evaluate_policy
from training.hybrid_td3.train import RandomFeasiblePolicy
from training.train_td3 import LEGACY_ERROR, run_smoke as legacy_td3_smoke


run_dir = Path("runs/baseline_comparison")
(run_dir / "trajectories").mkdir(parents=True, exist_ok=True)
results = {}

with PowerSystemEnv() as env:
    results["rule"] = evaluate_policy(env, RuleBasedController(env), run_dir / "trajectories" / "rule.csv")

with PowerSystemEnv() as env:
    results["random_feasible"] = evaluate_policy(
        env, RandomFeasiblePolicy(env), run_dir / "trajectories" / "random_feasible.csv"
    )

legacy = legacy_td3_smoke(allow_legacy_smoke=False)
results["legacy_box_td3"] = {"blocked": True, "error": legacy.get("error", LEGACY_ERROR)}

(run_dir / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
