"""C_ref 测试。"""

from pathlib import Path

import yaml

from envs.reward_calculator import RewardCalculator, IncompleteRewardConfigError
import pytest


def test_cost_ref_from_config_file_shape():
    """验证 reward_config 中 cost_reference 结构与 require_complete 行为。"""
    root = Path(__file__).resolve().parents[1]
    with (root / "src/config/reward_config.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cref = cfg.get("cost_reference") or {}
    assert "unit" in cref and cref["unit"] == "CNY_per_step"
    assert "source" in cref
    # value 可能尚未标定；正式训练 require_complete 会 fail
    cfg["decision_interval_seconds"] = 3600
    cfg["episode_steps"] = 168
    if cref.get("value") is None:
        with pytest.raises(IncompleteRewardConfigError):
            RewardCalculator(cfg, require_complete=True)
    else:
        assert float(cref["value"]) > 0
        RewardCalculator(cfg, require_complete=True)
