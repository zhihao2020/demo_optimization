"""Executed-action jsonl is the GiveSafe-canonical physical action."""
from __future__ import annotations

import json

import numpy as np

from test_env_reset import FakeAdapter, _action
from actions import CaesMode
from envs.power_system_env import PowerSystemEnv


def test_successful_step_appends_executed_prefix(tmp_path):
    adapter = FakeAdapter()
    env = PowerSystemEnv(adapter=adapter, forecast_enabled=False)
    log = tmp_path / "executed_actions.jsonl"
    env.executed_log_path = log
    env.reset(seed=0)
    env.step(_action(CaesMode.CHARGE, mag=1.0))
    env.close()
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["u_caes_executed"] > 0.8
    assert row["u_caes_readback"] == row["u_caes_executed"]
    assert row["caes_mode"] == int(CaesMode.CHARGE)
    assert row["previous_u_caes"] is None
    assert "cold_soc_before" in row
    assert "cold_soc_after" in row
