"""FS-HSAC replay: Bellman (physical FMU) vs Feasibility (accept/reject labels)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from training.fs_hsac.action_support import feasible_to_support_dict


@dataclass
class FSHSACTransition:
    observation: np.ndarray
    next_observation: np.ndarray
    u_tp: float
    u_battery: float
    u_caes: float
    reward: float
    terminated: bool
    truncated: bool
    support: dict[str, float | bool]
    next_support: dict[str, float | bool]
    physically_valid: bool = True
    feasibility_label: int | None = None  # 1=safe, 0=unsafe; None=not for feas replay
    failure_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _batch_from_transitions(batch: list[FSHSACTransition], *, bellman: bool) -> dict[str, np.ndarray]:
    n = len(batch)
    obs = np.stack([t.observation for t in batch]).astype(np.float32)
    next_obs = np.stack([t.next_observation for t in batch]).astype(np.float32)
    u_tp = np.asarray([t.u_tp for t in batch], dtype=np.float32)
    u_bat = np.asarray([t.u_battery for t in batch], dtype=np.float32)
    u_caes = np.asarray([t.u_caes for t in batch], dtype=np.float32)
    reward = np.asarray([t.reward for t in batch], dtype=np.float32)
    done = np.asarray([float(t.terminated) for t in batch], dtype=np.float32)

    def pack(support_attr: str, pfx: str) -> dict[str, np.ndarray]:
        mode_mask = np.zeros((n, 3), dtype=np.bool_)
        u_tp_low = np.zeros(n, dtype=np.float32)
        u_tp_high = np.zeros(n, dtype=np.float32)
        u_bat_low = np.zeros(n, dtype=np.float32)
        u_bat_high = np.zeros(n, dtype=np.float32)
        dis_lo = np.zeros(n, dtype=np.float32)
        dis_hi = np.zeros(n, dtype=np.float32)
        chg_lo = np.zeros(n, dtype=np.float32)
        chg_hi = np.zeros(n, dtype=np.float32)
        for i, t in enumerate(batch):
            s = getattr(t, support_attr)
            if any(k not in s for k in ("u_caes_discharge_low", "u_caes_charge_low")):
                raise KeyError(f"support missing CAES interval fields: keys={sorted(s)}")
            mode_mask[i, 0] = bool(s["mode_discharge"])
            mode_mask[i, 1] = bool(s["mode_idle"])
            mode_mask[i, 2] = bool(s["mode_charge"])
            u_tp_low[i] = float(s["u_tp_low"])
            u_tp_high[i] = float(s["u_tp_high"])
            u_bat_low[i] = float(s["u_battery_low"])
            u_bat_high[i] = float(s["u_battery_high"])
            dis_lo[i] = float(s["u_caes_discharge_low"])
            dis_hi[i] = float(s["u_caes_discharge_high"])
            chg_lo[i] = float(s["u_caes_charge_low"])
            chg_hi[i] = float(s["u_caes_charge_high"])
        return {
            f"{pfx}mode_mask": mode_mask,
            f"{pfx}u_tp_low": u_tp_low,
            f"{pfx}u_tp_high": u_tp_high,
            f"{pfx}u_bat_low": u_bat_low,
            f"{pfx}u_bat_high": u_bat_high,
            f"{pfx}dis_lo": dis_lo,
            f"{pfx}dis_hi": dis_hi,
            f"{pfx}chg_lo": chg_lo,
            f"{pfx}chg_hi": chg_hi,
        }

    out = {
        "obs": obs,
        "next_obs": next_obs,
        "u_tp": u_tp,
        "u_battery": u_bat,
        "u_caes": u_caes,
        "reward": reward,
        "done": done,
    }
    out.update(pack("support", ""))
    out.update(pack("next_support", "next_"))
    if not bellman:
        labels = np.asarray([int(t.feasibility_label) for t in batch], dtype=np.float32)
        out["feasibility_label"] = labels
        out["failure_type"] = np.asarray([t.failure_type or "" for t in batch], dtype=object)
    return out


class _Ring:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._storage: list[FSHSACTransition] = []
        self._pos = 0

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, tr: FSHSACTransition) -> None:
        if len(self._storage) < self.capacity:
            self._storage.append(tr)
        else:
            self._storage[self._pos] = tr
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int) -> list[FSHSACTransition]:
        n = len(self._storage)
        if n == 0:
            raise RuntimeError("replay empty")
        idx = np.random.randint(0, n, size=batch_size)
        return [self._storage[i] for i in idx]


class FSHSACReplayBuffer:
    """Split Bellman and feasibility memories."""

    def __init__(self, capacity: int = 100_000):
        self.bellman = _Ring(capacity)
        self.feasibility = _Ring(capacity)
        self.rejected_to_bellman_attempts = 0

    def __len__(self) -> int:
        return len(self.bellman)

    @property
    def bellman_size(self) -> int:
        return len(self.bellman)

    @property
    def feasibility_size(self) -> int:
        return len(self.feasibility)

    def add_physical(
        self,
        *,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        feasible,
        next_feasible,
    ) -> None:
        support = feasible_to_support_dict(feasible)
        next_support = feasible_to_support_dict(next_feasible)
        u_tp = float(np.asarray(action["u_tp"]).reshape(-1)[0])
        u_bat = float(np.asarray(action["u_battery"]).reshape(-1)[0])
        u_caes = float(np.asarray(action["u_caes"]).reshape(-1)[0])
        tr = FSHSACTransition(
            observation=np.asarray(obs, dtype=np.float32),
            next_observation=np.asarray(next_obs, dtype=np.float32),
            u_tp=u_tp,
            u_battery=u_bat,
            u_caes=u_caes,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            support=support,
            next_support=next_support,
            physically_valid=True,
            feasibility_label=1,
            failure_type=None,
        )
        self.bellman.add(tr)
        self.feasibility.add(tr)

    def add_rejection(
        self,
        *,
        obs: np.ndarray,
        action: dict[str, Any],
        feasible,
        failure_type: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Oracle/shadow rejection: feasibility label 0 only — never Bellman."""
        self.rejected_to_bellman_attempts += 0  # explicit: do not add to bellman
        support = feasible_to_support_dict(feasible)
        u_tp = float(np.asarray(action["u_tp"]).reshape(-1)[0])
        u_bat = float(np.asarray(action["u_battery"]).reshape(-1)[0])
        u_caes = float(np.asarray(action["u_caes"]).reshape(-1)[0])
        tr = FSHSACTransition(
            observation=np.asarray(obs, dtype=np.float32),
            next_observation=np.asarray(obs, dtype=np.float32),  # unused for feas
            u_tp=u_tp,
            u_battery=u_bat,
            u_caes=u_caes,
            reward=0.0,
            terminated=False,
            truncated=False,
            support=support,
            next_support=dict(support),
            physically_valid=False,
            feasibility_label=0,
            failure_type=failure_type,
            metadata=dict(metadata or {}),
        )
        self.feasibility.add(tr)

    def add_post_step_failure(
        self,
        *,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: dict[str, Any],
        feasible,
        next_feasible,
        failure_type: str | None = None,
    ) -> None:
        """FMU advanced but hard-failed: not Bellman; feasibility label 0."""
        support = feasible_to_support_dict(feasible)
        next_support = feasible_to_support_dict(next_feasible)
        u_tp = float(np.asarray(action["u_tp"]).reshape(-1)[0])
        u_bat = float(np.asarray(action["u_battery"]).reshape(-1)[0])
        u_caes = float(np.asarray(action["u_caes"]).reshape(-1)[0])
        tr = FSHSACTransition(
            observation=np.asarray(obs, dtype=np.float32),
            next_observation=np.asarray(next_obs, dtype=np.float32),
            u_tp=u_tp,
            u_battery=u_bat,
            u_caes=u_caes,
            reward=0.0,
            terminated=True,
            truncated=False,
            support=support,
            next_support=next_support,
            physically_valid=False,
            feasibility_label=0,
            failure_type=failure_type,
        )
        self.feasibility.add(tr)

    def sample_bellman(self, batch_size: int) -> dict[str, np.ndarray]:
        batch = self.bellman.sample(batch_size)
        for t in batch:
            if not t.physically_valid or t.feasibility_label == 0 and t.failure_type:
                # hard guard: rejection must never appear
                if not t.physically_valid:
                    raise RuntimeError("rejection leaked into Bellman replay")
        return _batch_from_transitions(batch, bellman=True)

    def sample_feasibility(self, batch_size: int) -> dict[str, np.ndarray]:
        return _batch_from_transitions(self.feasibility.sample(batch_size), bellman=False)
