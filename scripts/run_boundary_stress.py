"""运行 BoundaryStressTester 边界应力测试（默认 2000 步 smoke；Phase E 门控要求 ≥20000）。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from boundary_stress import BoundaryStressTester
from envs.power_system_env import PowerSystemEnv


def main(n: int = 2000) -> dict:
    """执行边界应力测试并写入 runs/boundary_stress/ 摘要。

    Args:
        n: 尝试步数(n)；Phase E 正式门控建议 ≥20000。

    Returns:
        BoundaryStressResult.to_dict() 扩展后的 payload 字典。
    """
    out = ROOT / "runs" / "boundary_stress"
    out.mkdir(parents=True, exist_ok=True)
    env = PowerSystemEnv(run_id="boundary_stress")
    tester = BoundaryStressTester(oracle=env.oracle, seed=0)
    result = tester.run(env, n_actions=n, reset_every=40)
    env.close()
    payload = result.to_dict()
    payload["target_for_phase_e"] = 20000
    (out / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "summary.yaml").write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    if result.failures:
        (out / "failures.json").write_text(json.dumps(result.failures[:200], indent=2, ensure_ascii=False), encoding="utf-8")
    print(yaml.safe_dump(payload, allow_unicode=True))
    return payload


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    main(n)
