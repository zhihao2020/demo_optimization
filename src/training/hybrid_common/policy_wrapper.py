"""评估/探索用策略包装（与算法解耦）。"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from actions import CaesMode
from envs.failures import FeasibleSetEmpty
from envs.power_system_env import PowerSystemEnv
from safety import GiveSafeController


class SupportsSelectAction(Protocol):
    def select_action(self, obs, feasible, deterministic: bool = False) -> dict: ...


class RandomFeasiblePolicy:
    def __init__(self, env: PowerSystemEnv):
        self.env = env

    def predict(self, _obs, deterministic: bool = False) -> dict:
        feasible = self.env.get_feasible_action_spec()
        modes = [
            m
            for m, ok in zip(
                (CaesMode.DISCHARGE, CaesMode.IDLE, CaesMode.CHARGE),
                (feasible.mode_mask.discharge, feasible.mode_mask.idle, feasible.mode_mask.charge),
            )
            if ok
        ]
        if not modes:
            raise FeasibleSetEmpty("无可选 CAES 模式")
        mode = modes[int(np.random.randint(len(modes)))]
        u_tp = float(np.random.uniform(feasible.u_tp_low, feasible.u_tp_high))
        u_bat = float(np.random.uniform(feasible.u_battery_low, feasible.u_battery_high))
        mag = 0.0 if mode == CaesMode.IDLE else float(np.random.uniform(0.0, 1.0))
        return {
            "u_tp": np.asarray([u_tp], dtype=np.float32),
            "u_battery": np.asarray([u_bat], dtype=np.float32),
            "caes_mode": int(mode),
            "caes_magnitude": np.asarray([mag], dtype=np.float32),
        }


class HybridGiveSafePolicyWrapper:
    """评估用：经 GiveSafeController 采样；绝不调用规则 fallback。"""

    def __init__(
        self,
        agent: SupportsSelectAction,
        env: PowerSystemEnv,
        controller: GiveSafeController,
        deterministic: bool = True,
    ):
        self.agent = agent
        self.env = env
        self.controller = controller
        self.deterministic = deterministic

    def predict(self, obs, deterministic: bool | None = None):
        det = self.deterministic if deterministic is None else deterministic

        def propose():
            feasible = self.env.get_feasible_action_spec()
            return self.agent.select_action(obs, feasible, deterministic=det)

        gs = self.controller.select_safe_action(
            self.env.last_outputs,
            self.env.previous_thermal,
            propose,
            deterministic=det,
        )
        return gs.safe_action

    def on_episode_reset(self, info: dict[str, Any]) -> None:
        if self.controller.shadow is not None:
            self.controller.shadow.on_episode_reset(float(info.get("time", 0.0) or 0.0))

    def on_transition(self, info: dict[str, Any]) -> None:
        if self.controller.shadow is None or not info.get("transition_valid"):
            return
        self.controller.shadow.on_physical_success(
            {
                "u_tp": float(info["decoded_u_tp"]),
                "u_battery": float(info["decoded_u_battery"]),
                "u_caes": float(info["decoded_u_caes"]),
            }
        )
