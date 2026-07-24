"""PPO 物理步进 rollout 缓冲(RolloutBuffer) + GAE 优势估计。"""

from __future__ import annotations

import numpy as np


class RolloutBuffer:
    """PPO rollout 缓冲：仅存物理有效步，含动态边界与 old log_prob。"""

    def __init__(self, capacity: int, obs_dim: int):
        """预分配固定容量数组。

        Args:
            capacity: 最大 rollout 步数。
            obs_dim: 观测维度。
        """
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.reset()

    def reset(self) -> None:
        """清空写入指针并重置数组（不重新分配内存）。

        Returns:
            无。
        """
        n = self.capacity
        d = self.obs_dim
        self.obs = np.zeros((n, d), dtype=np.float32)
        self.next_obs = np.zeros((n, d), dtype=np.float32)
        self.u_tp = np.zeros(n, dtype=np.float32)
        self.u_battery = np.zeros(n, dtype=np.float32)
        self.caes_mode = np.zeros(n, dtype=np.int64)
        self.caes_magnitude = np.zeros(n, dtype=np.float32)
        self.reward = np.zeros(n, dtype=np.float32)
        self.done = np.zeros(n, dtype=np.float32)
        self.log_prob = np.zeros(n, dtype=np.float32)
        self.value = np.zeros(n, dtype=np.float32)
        self.advantage = np.zeros(n, dtype=np.float32)
        self.return_ = np.zeros(n, dtype=np.float32)
        self.mode_mask = np.zeros((n, 3), dtype=np.bool_)
        self.u_tp_low = np.zeros(n, dtype=np.float32)
        self.u_tp_high = np.zeros(n, dtype=np.float32)
        self.u_bat_low = np.zeros(n, dtype=np.float32)
        self.u_bat_high = np.zeros(n, dtype=np.float32)
        self.pos = 0

    def __len__(self) -> int:
        """已写入步数。

        Returns:
            当前 pos 指针值。
        """
        return self.pos

    def add(
        self,
        *,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: dict,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        mode_mask: np.ndarray,
        bounds: dict[str, float],
    ) -> None:
        """追加一步物理转移。

        Args:
            obs: 步前观测。
            next_obs: 步后观测。
            action: 混合动作字典。
            reward: 环境奖励。
            done: 是否 episode 结束。
            log_prob: 行为策略 log_prob。
            value: V(s) 估计。
            mode_mask: CAES 模式掩码。
            bounds: 动态动作边界字典。

        Raises:
            RuntimeError: 超出 capacity 时抛出。
        """
        if self.pos >= self.capacity:
            raise RuntimeError("RolloutBuffer 已满")
        i = self.pos
        self.obs[i] = np.asarray(obs, dtype=np.float32).reshape(-1)
        self.next_obs[i] = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        self.u_tp[i] = float(np.asarray(action["u_tp"]).reshape(-1)[0])
        self.u_battery[i] = float(np.asarray(action["u_battery"]).reshape(-1)[0])
        self.caes_mode[i] = int(action["caes_mode"])
        self.caes_magnitude[i] = float(np.asarray(action["caes_magnitude"]).reshape(-1)[0])
        self.reward[i] = float(reward)
        self.done[i] = float(done)
        self.log_prob[i] = float(log_prob)
        self.value[i] = float(value)
        self.mode_mask[i] = np.asarray(mode_mask, dtype=np.bool_).reshape(3)
        self.u_tp_low[i] = float(bounds["u_tp_low"])
        self.u_tp_high[i] = float(bounds["u_tp_high"])
        self.u_bat_low[i] = float(bounds["u_battery_low"])
        self.u_bat_high[i] = float(bounds["u_battery_high"])
        self.pos += 1

    def compute_gae(self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        """反向计算 GAE 优势与回报。

        Args:
            last_value: 末步 bootstrap 价值 V(s_T)。
            gamma: 折扣因子。
            gae_lambda: GAE λ。

        Returns:
            无；结果写入 advantage 与 return_ 数组。
        """
        n = self.pos
        adv = 0.0
        for t in reversed(range(n)):
            next_nonterminal = 1.0 - self.done[t]
            next_v = last_value if t == n - 1 else self.value[t + 1]
            delta = self.reward[t] + gamma * next_v * next_nonterminal - self.value[t]
            adv = delta + gamma * gae_lambda * next_nonterminal * adv
            self.advantage[t] = adv
            self.return_[t] = adv + self.value[t]

    def get_batches(self, batch_size: int):
        """随机打乱索引并按 batch_size 产出训练小批。

        Args:
            batch_size: 小批大小。

        Yields:
            含 obs、动作、log_prob、advantage、return_、边界等的字典。
        """
        n = self.pos
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            b = idx[start : start + batch_size]
            yield {
                "obs": self.obs[b],
                "u_tp": self.u_tp[b],
                "u_battery": self.u_battery[b],
                "caes_mode": self.caes_mode[b],
                "caes_magnitude": self.caes_magnitude[b],
                "log_prob": self.log_prob[b],
                "advantage": self.advantage[b],
                "return_": self.return_[b],
                "mode_mask": self.mode_mask[b],
                "u_tp_low": self.u_tp_low[b],
                "u_tp_high": self.u_tp_high[b],
                "u_bat_low": self.u_bat_low[b],
                "u_bat_high": self.u_bat_high[b],
            }
