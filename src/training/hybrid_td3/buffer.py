"""经济 replay 与安全数据集分离。
EconomicReplayBuffer / FilteredReplayBuffer：仅 physically_valid 经济转移。
SafetyDataset：safe + post-step fails + residuals + failure types；禁止把 reward=-1e9 写入经济 buffer。
"""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
@dataclass
class Transition:
    observation: np.ndarray
    hybrid_action: dict[str, Any]
    decoded_fmu_action: dict[str, float]
    reward: float
    next_observation: np.ndarray
    terminated: bool
    valid_mode_mask: np.ndarray
    dynamic_action_bounds: dict[str, float]
    reward_terms: dict[str, float]
    constraint_metadata: dict[str, Any] = field(default_factory=dict)
    next_valid_mode_mask: np.ndarray | None = None
    next_dynamic_action_bounds: dict[str, float] | None = None
    physically_valid: bool = True
    oracle_predicted_next_state: dict[str, float] | None = None
    residuals: dict[str, float] | None = None
    distance_to_physical_boundary: dict[str, float] | None = None
    distance_to_safe_boundary: dict[str, float] | None = None
    safety_probability: float | None = None
    safety_threshold: float | None = None
    oracle_version: str | None = None
    transition_type: str = "physical"  # physical | givesafe_rejection
    truncated: bool = False
class FilteredReplayBuffer:
    """经济 replay：拒绝 invalid；永不接受 reward=-1e9 伪标签。"""
    FORBIDDEN_ECONOMIC_REWARD = -1e9
    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self._storage: list[Transition] = []
        self._pos = 0
        self.rejected_count = 0
        self.invalid_attempt_count = 0
    def __len__(self) -> int:
        return len(self._storage)
    def add(self, transition: Transition) -> bool:
        if not transition.physically_valid:
            self.rejected_count += 1
            self.invalid_attempt_count += 1
            return False
        if abs(float(transition.reward) - self.FORBIDDEN_ECONOMIC_REWARD) < 1.0:
            self.rejected_count += 1
            self.invalid_attempt_count += 1
            return False
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
        else:
            self._storage[self._pos] = transition
        self._pos = (self._pos + 1) % self.capacity
        return True
    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        n = len(self._storage)
        if n == 0:
            raise RuntimeError("replay buffer 为空")
        idx = np.random.randint(0, n, size=batch_size)
        batch = [self._storage[i] for i in idx]
        obs = np.stack([t.observation for t in batch]).astype(np.float32)
        next_obs = np.stack([t.next_observation for t in batch]).astype(np.float32)
        u_tp = np.asarray([t.hybrid_action["u_tp"] for t in batch], dtype=np.float32)
        u_bat = np.asarray([t.hybrid_action["u_battery"] for t in batch], dtype=np.float32)
        mode = np.asarray([t.hybrid_action["caes_mode"] for t in batch], dtype=np.int64)
        mag = np.asarray([t.hybrid_action["caes_magnitude"] for t in batch], dtype=np.float32)
        reward = np.asarray([t.reward for t in batch], dtype=np.float32)
        done = np.asarray([t.terminated for t in batch], dtype=np.float32)
        mask = np.stack([t.valid_mode_mask for t in batch]).astype(np.bool_)
        next_mask = np.stack(
            [(t.next_valid_mode_mask if t.next_valid_mode_mask is not None else t.valid_mode_mask) for t in batch]
        ).astype(np.bool_)
        def bounds_arr(key: str, next_b: bool = False):
            vals = []
            for t in batch:
                src = t.next_dynamic_action_bounds if next_b and t.next_dynamic_action_bounds else t.dynamic_action_bounds
                vals.append(float(src[key]))
            return np.asarray(vals, dtype=np.float32)
        return {
            "obs": obs,
            "next_obs": next_obs,
            "u_tp": u_tp,
            "u_battery": u_bat,
            "caes_mode": mode,
            "caes_magnitude": mag,
            "reward": reward,
            "done": done,
            "mode_mask": mask,
            "next_mode_mask": next_mask,
            "u_tp_low": bounds_arr("u_tp_low"),
            "u_tp_high": bounds_arr("u_tp_high"),
            "u_bat_low": bounds_arr("u_battery_low"),
            "u_bat_high": bounds_arr("u_battery_high"),
            "next_u_tp_low": bounds_arr("u_tp_low", True),
            "next_u_tp_high": bounds_arr("u_tp_high", True),
            "next_u_bat_low": bounds_arr("u_battery_low", True),
            "next_u_bat_high": bounds_arr("u_battery_high", True),
        }
# 显式别名：经济 buffer
EconomicReplayBuffer = FilteredReplayBuffer
@dataclass
class SafetySample:
    label_safe: bool
    fine_failure_type: str | None
    previous_observation: dict[str, float]
    hybrid_action: dict[str, Any]
    decoded_fmu_action: dict[str, float] | None
    oracle_predicted_next_state: dict[str, float] | None
    actual_fmu_outputs: dict[str, float] | None
    residuals: dict[str, float] | None
    dangerous_residual: dict[str, float] | None
    distance_to_physical_boundary: dict[str, float] | None
    distance_to_safe_boundary: dict[str, float] | None
    failure_reason: str | None = None
    oracle_version: str | None = None
    episode: int | None = None
    step: int | None = None
    run_id: str | None = None
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
class SafetyDataset:
    """存储 safe / post-step-fail / residual；与经济 Critic 数据隔离。"""
    def __init__(self):
        self.samples: list[SafetySample] = []
    def __len__(self) -> int:
        return len(self.samples)
    def add(self, sample: SafetySample) -> None:
        self.samples.append(sample)
    def add_from_failure_record(self, rec: dict[str, Any]) -> None:
        self.add(
            SafetySample(
                label_safe=False,
                fine_failure_type=rec.get("fine_failure_type"),
                previous_observation=dict(rec.get("previous_observation") or rec.get("last_valid_state") or {}),
                hybrid_action=dict(rec.get("hybrid_action") or {}),
                decoded_fmu_action=rec.get("decoded_fmu_action"),
                oracle_predicted_next_state=rec.get("oracle_predicted_next_state"),
                actual_fmu_outputs=rec.get("actual_fmu_outputs"),
                residuals=rec.get("residuals"),
                dangerous_residual=rec.get("dangerous_residual"),
                distance_to_physical_boundary=rec.get("distance_to_physical_boundary"),
                distance_to_safe_boundary=rec.get("distance_to_safe_boundary"),
                failure_reason=rec.get("triggering_constraint") or rec.get("modelica_assert_message"),
                oracle_version=rec.get("oracle_version"),
                episode=rec.get("episode"),
                step=rec.get("step"),
                run_id=rec.get("run_id"),
            )
        )
    def add_safe_transition(
        self,
        *,
        previous_observation: dict[str, float],
        hybrid_action: dict[str, Any],
        decoded_fmu_action: dict[str, float],
        predicted: dict[str, float] | None,
        actual: dict[str, float] | None,
        residuals: dict[str, float] | None,
        distances_physical: dict[str, float] | None,
        distances_safe: dict[str, float] | None,
        oracle_version: str | None = None,
    ) -> None:
        self.add(
            SafetySample(
                label_safe=True,
                fine_failure_type=None,
                previous_observation=previous_observation,
                hybrid_action=hybrid_action,
                decoded_fmu_action=decoded_fmu_action,
                oracle_predicted_next_state=predicted,
                actual_fmu_outputs=actual,
                residuals=residuals,
                dangerous_residual=None,
                distance_to_physical_boundary=distances_physical,
                distance_to_safe_boundary=distances_safe,
                oracle_version=oracle_version,
            )
        )
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [s.to_dict() for s in self.samples]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    @classmethod
    def load(cls, path: str | Path) -> "SafetyDataset":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        ds = cls()
        for item in data:
            ds.samples.append(SafetySample(**item))
        return ds
    def split_safe_fail(self) -> tuple[list[dict], list[dict]]:
        safe = [s.to_dict() for s in self.samples if s.label_safe]
        fail = [s.to_dict() for s in self.samples if not s.label_safe]
        return safe, fail
