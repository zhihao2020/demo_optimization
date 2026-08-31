"""经济 replay 与安全数据集分离。

EconomicReplayBuffer / FilteredReplayBuffer：仅 physically_valid 经济转移。
SafetyDataset：safe + post-step fails + residuals + failure types；
禁止把 reward=-1e9 写入经济 buffer。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Transition:
    """单步混合动作转移(Transition)记录。"""

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
    """过滤经济 replay(FilteredReplayBuffer)：拒绝 invalid；永不接受 reward=-1e9 伪标签。"""

    FORBIDDEN_ECONOMIC_REWARD = -1e9

    def __init__(self, capacity: int = 100_000):
        """初始化环形缓冲区。

        Args:
            capacity: 最大存储转移数。
        """
        self.capacity = capacity
        self._storage: list[Transition] = []
        self._pos = 0
        self.rejected_count = 0
        self.invalid_attempt_count = 0

    def __len__(self) -> int:
        """当前有效样本数。

        Returns:
            缓冲区长度。
        """
        return len(self._storage)

    def add(self, transition: Transition) -> bool:
        """尝试写入转移；无效或伪奖励被拒绝。

        Args:
            transition: 待写入转移。

        Returns:
            True 表示成功写入，False 表示拒绝。
        """
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
        """均匀随机采样一批训练数据。

        Args:
            batch_size: 批大小。

        Returns:
            含 obs、动作、reward、动态边界与 mode_mask 的字典。

        Raises:
            RuntimeError: 缓冲区为空时抛出。
        """
        n = len(self._storage)
        if n == 0:
            raise RuntimeError("replay buffer 为空")
        idx = np.random.randint(0, n, size=batch_size)
        batch = [self._storage[i] for i in idx]
        obs = np.stack([t.observation for t in batch]).astype(np.float32)
        next_obs = np.stack([t.next_observation for t in batch]).astype(np.float32)
        u_tp = np.asarray([t.hybrid_action["u_tp"] for t in batch], dtype=np.float32)
        u_bat = np.asarray([t.hybrid_action["u_battery"] for t in batch], dtype=np.float32)
        u_caes = np.asarray([t.hybrid_action["u_caes"] for t in batch], dtype=np.float32)
        from actions.caes_u import mag_from_u, mode_from_u

        caes_mode = np.asarray(
            [
                int(t.hybrid_action.get("caes_mode", int(mode_from_u(float(t.hybrid_action["u_caes"])))))
                for t in batch
            ],
            dtype=np.int64,
        )
        caes_magnitude = np.asarray(
            [
                float(
                    t.hybrid_action.get("caes_magnitude", mag_from_u(float(t.hybrid_action["u_caes"])))
                )
                for t in batch
            ],
            dtype=np.float32,
        )
        reward = np.asarray([t.reward for t in batch], dtype=np.float32)
        done = np.asarray([t.terminated for t in batch], dtype=np.float32)
        mask = np.stack([t.valid_mode_mask for t in batch]).astype(np.bool_)
        next_mask = np.stack(
            [(t.next_valid_mode_mask if t.next_valid_mode_mask is not None else t.valid_mode_mask) for t in batch]
        ).astype(np.bool_)

        from actions.caes_u import CHARGE_HI, CHARGE_LO, DISCHARGE_HI, DISCHARGE_LO

        _fb = {
            "u_caes_discharge_low": DISCHARGE_LO,
            "u_caes_discharge_high": DISCHARGE_HI,
            "u_caes_charge_low": CHARGE_LO,
            "u_caes_charge_high": CHARGE_HI,
            "grid_residual_W": 0.0,
            "grid_g_min_W": -5.0e8,
            "grid_g_max_W": 5.0e8,
            "p_cap_thermal_W": 1.5e8,
            "p_cap_battery_W": 1.0e8,
            "p_cap_caes_W": 1.5e8,
        }

        def bounds_arr(key: str, next_b: bool = False):
            """从批次转移中提取动态界数组。"""
            vals = []
            for t in batch:
                src = t.next_dynamic_action_bounds if next_b and t.next_dynamic_action_bounds else t.dynamic_action_bounds
                vals.append(float(src.get(key, _fb.get(key, 0.0))))
            return np.asarray(vals, dtype=np.float32)

        return {
            "obs": obs,
            "next_obs": next_obs,
            "u_tp": u_tp,
            "u_battery": u_bat,
            "u_caes": u_caes,
            "caes_mode": caes_mode,
            "caes_magnitude": caes_magnitude,
            "reward": reward,
            "done": done,
            "mode_mask": mask,
            "next_mode_mask": next_mask,
            "u_tp_low": bounds_arr("u_tp_low"),
            "u_tp_high": bounds_arr("u_tp_high"),
            "u_bat_low": bounds_arr("u_battery_low"),
            "u_bat_high": bounds_arr("u_battery_high"),
            "dis_lo": bounds_arr("u_caes_discharge_low"),
            "dis_hi": bounds_arr("u_caes_discharge_high"),
            "chg_lo": bounds_arr("u_caes_charge_low"),
            "chg_hi": bounds_arr("u_caes_charge_high"),
            "next_u_tp_low": bounds_arr("u_tp_low", True),
            "next_u_tp_high": bounds_arr("u_tp_high", True),
            "next_u_bat_low": bounds_arr("u_battery_low", True),
            "next_u_bat_high": bounds_arr("u_battery_high", True),
            "next_dis_lo": bounds_arr("u_caes_discharge_low", True),
            "next_dis_hi": bounds_arr("u_caes_discharge_high", True),
            "next_chg_lo": bounds_arr("u_caes_charge_low", True),
            "next_chg_hi": bounds_arr("u_caes_charge_high", True),
            "grid_residual_W": bounds_arr("grid_residual_W"),
            "grid_g_min_W": bounds_arr("grid_g_min_W"),
            "grid_g_max_W": bounds_arr("grid_g_max_W"),
            "p_cap_thermal_W": bounds_arr("p_cap_thermal_W"),
            "p_cap_battery_W": bounds_arr("p_cap_battery_W"),
            "p_cap_caes_W": bounds_arr("p_cap_caes_W"),
            "next_grid_residual_W": bounds_arr("grid_residual_W", True),
            "next_grid_g_min_W": bounds_arr("grid_g_min_W", True),
            "next_grid_g_max_W": bounds_arr("grid_g_max_W", True),
        }


# 显式别名：经济 buffer
EconomicReplayBuffer = FilteredReplayBuffer


@dataclass
class SafetySample:
    """安全校准样本(SafetySample)：含标签、残差与边界距离。"""

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
        """序列化为字典。

        Returns:
            dataclass 字段字典。
        """
        return asdict(self)


class SafetyDataset:
    """安全数据集(SafetyDataset)：存储 safe / post-step-fail / residual；与经济 Critic 数据隔离。"""

    def __init__(self):
        """初始化空样本列表。"""
        self.samples: list[SafetySample] = []

    def __len__(self) -> int:
        """样本数量。

        Returns:
            列表长度。
        """
        return len(self.samples)

    def add(self, sample: SafetySample) -> None:
        """追加一条安全样本。

        Args:
            sample: 安全样本实例。

        Returns:
            无。
        """
        self.samples.append(sample)

    def add_from_failure_record(self, rec: dict[str, Any]) -> None:
        """从环境 failure_record 构造失败样本并写入。

        Args:
            rec: 失败记录字典。

        Returns:
            无。
        """
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
        """写入标签为 safe 的成功物理转移样本。

        Args:
            previous_observation: 步前 FMU 输出状态。
            hybrid_action: 混合动作字典。
            decoded_fmu_action: 解码后 FMU 动作。
            predicted: Oracle 预测下一状态。
            actual: 实际 FMU 输出。
            residuals: 预测残差。
            distances_physical: 到物理边界的距离。
            distances_safe: 到安全边界的距离。
            oracle_version: Oracle 版本号。

        Returns:
            无。
        """
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
        """将样本列表写入 JSON 文件。

        Args:
            path: 输出路径。

        Returns:
            无。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [s.to_dict() for s in self.samples]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SafetyDataset":
        """从 JSON 文件加载安全数据集。

        Args:
            path: JSON 文件路径。

        Returns:
            重建的 SafetyDataset 实例。
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        ds = cls()
        for item in data:
            ds.samples.append(SafetySample(**item))
        return ds

    def split_safe_fail(self) -> tuple[list[dict], list[dict]]:
        """按 safe / fail 标签拆分样本。

        Returns:
            (safe 列表, fail 列表) 元组，元素为字典。
        """
        safe = [s.to_dict() for s in self.samples if s.label_safe]
        fail = [s.to_dict() for s in self.samples if not s.label_safe]
        return safe, fail
