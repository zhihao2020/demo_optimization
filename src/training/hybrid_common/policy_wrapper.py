"""评估/探索用策略包装（与具体 RL 算法解耦）。"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from actions import CaesMode
from envs.failures import FeasibleSetEmpty
from envs.power_system_env import PowerSystemEnv
from safety import GiveSafeController


class SupportsSelectAction(Protocol):
    """支持按可行域采样动作的智能体协议。"""

    def select_action(self, obs, feasible, deterministic: bool = False) -> dict:
        """从观测与可行域采样混合动作。

        Args:
            obs: 环境观测向量。
            feasible: 当前步可行动作规格。
            deterministic: 是否确定性采样。

        Returns:
            混合动作字典。
        """
        ...


class RandomFeasiblePolicy:
    """随机可行动作策略(RandomFeasiblePolicy)：在动态边界内均匀采样，用于早期探索。"""

    def __init__(self, env: PowerSystemEnv):
        """绑定环境以查询可行域。

        Args:
            env: 电力系统环境实例。
        """
        self.env = env

    def predict(self, _obs, deterministic: bool = False) -> dict:
        """在可行 CAES 模式与连续边界内随机采样动作。

        Args:
            _obs: 观测（未使用）。
            deterministic: 忽略；始终随机。

        Returns:
            含 u_tp、u_battery、caes_mode、caes_magnitude 的动作字典。

        Raises:
            FeasibleSetEmpty: 无可选 CAES 模式时抛出。
        """
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
    """GiveSafe 评估策略包装(HybridGiveSafePolicyWrapper)：经 GiveSafeController 采样，绝不调用规则 fallback。"""

    def __init__(
        self,
        agent: SupportsSelectAction,
        env: PowerSystemEnv,
        controller: GiveSafeController,
        deterministic: bool = True,
    ):
        """组装评估用安全策略环。

        Args:
            agent: 实现 ``select_action`` 的训练智能体。
            env: 评估环境。
            controller: GiveSafe 安全控制器。
            deterministic: 默认是否确定性动作。
        """
        self.agent = agent
        self.env = env
        self.controller = controller
        self.deterministic = deterministic

    def predict(self, obs, deterministic: bool | None = None):
        """通过 GiveSafe 环选择安全动作并返回。

        Args:
            obs: 当前观测。
            deterministic: 覆盖实例默认可选；None 时使用构造时的值。

        Returns:
            经 GiveSafe 验证的安全混合动作字典。
        """
        det = self.deterministic if deterministic is None else deterministic

        def propose():
            """按当前可行域向智能体请求动作。"""
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
        """回合重置时通知影子 FMU 校验器。

        Args:
            info: reset 返回的 info，含仿真时间。

        Returns:
            无。
        """
        if self.controller.shadow is not None:
            self.controller.shadow.on_episode_reset(float(info.get("time", 0.0) or 0.0))

    def on_transition(self, info: dict[str, Any]) -> None:
        """物理步成功后更新影子 FMU 状态。

        Args:
            info: step 返回的 info，需含解码后动作与 transition_valid。

        Returns:
            无。
        """
        if self.controller.shadow is None or not info.get("transition_valid"):
            return
        self.controller.shadow.on_physical_success(
            {
                "u_tp": float(info["decoded_u_tp"]),
                "u_battery": float(info["decoded_u_battery"]),
                "u_caes": float(info["decoded_u_caes"]),
            }
        )
