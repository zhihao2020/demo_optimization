"""Story A lock-CAES counterfactual (no FMU)."""
from __future__ import annotations

from actions import ModeMask, PhysicalFmuAction
from envs.power_system_env import caes_locked_from_config


def test_lock_from_config_flag():
    assert caes_locked_from_config({"caes": {"locked": True}}) is True
    assert caes_locked_from_config({"caes": {"locked": False}}) is False
    assert caes_locked_from_config({}) is False


def test_lock_env_var_overrides_config(monkeypatch):
    monkeypatch.setenv("OPTIMAL_DEMO_LOCK_CAES", "1")
    assert caes_locked_from_config({"caes": {"locked": False}}) is True
    monkeypatch.setenv("OPTIMAL_DEMO_LOCK_CAES", "0")
    assert caes_locked_from_config({"caes": {"locked": True}}) is False


def test_forced_idle_action_zeros_u_caes():
    raw = PhysicalFmuAction(u_tp=0.8, u_battery=-0.2, u_caes=0.95)
    locked = PhysicalFmuAction(u_tp=raw.u_tp, u_battery=raw.u_battery, u_caes=0.0)
    assert locked.u_caes == 0.0
    assert locked.u_tp == 0.8
    mask = ModeMask(discharge=False, idle=True, charge=False)
    assert mask.idle and not mask.charge and not mask.discharge
