"""规则控制器 rollout：评估一周并写入 runs/rule_controller/ 轨迹。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from controllers.rule_based_controller import RuleBasedController
from envs.power_system_env import PowerSystemEnv
from training.evaluate_td3 import evaluate_policy

with PowerSystemEnv() as env:
    result = evaluate_policy(
        env,
        RuleBasedController(env),
        Path("runs/rule_controller/trajectories/rollout.csv"),
    )
print(result)
