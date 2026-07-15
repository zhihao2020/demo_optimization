"""SafetyClassifier / FeasibilityCalibrator：与经济 Critic 分离，优化 unsafe recall。"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
FEATURE_KEYS = (
    "battery_soc",
    "caes_gas_soc",
    "caes_hot_soc",
    "caes_cold_soc",
    "caes_gas_pressure_norm",
    "caes_gas_temperature_norm",
    "u_tp",
    "u_battery",
    "mode_discharge",
    "mode_idle",
    "mode_charge",
    "caes_magnitude",
    "dist_bat_min",
    "dist_bat_max",
    "dist_gas_min",
    "dist_gas_max",
)
@dataclass
class SafetyMetrics:
    unsafe_recall: float
    unsafe_precision: float
    false_safe_rate: float
    false_unsafe_rate: float
    per_failure_type_recall: dict[str, float]
    n_safe: int
    n_unsafe: int
    threshold: float
    model_version: str
    def to_dict(self) -> dict[str, Any]:
        return {
            "unsafe_recall": self.unsafe_recall,
            "unsafe_precision": self.unsafe_precision,
            "false_safe_rate": self.false_safe_rate,
            "false_unsafe_rate": self.false_unsafe_rate,
            "per_failure_type_recall": self.per_failure_type_recall,
            "n_safe": self.n_safe,
            "n_unsafe": self.n_unsafe,
            "threshold": self.threshold,
            "model_version": self.model_version,
            "gate_metric": "false_safe_rate",
        }
class SafetyClassifier:
    """逻辑回归风格安全分类器：P(safe | s,a)。
    标签：1=post-step 成功（safe），0=post-step 硬失败（unsafe）。
    门控指标：false_safe_rate = FN_unsafe / n_unsafe（把危险判成安全）。
    """
    def __init__(
        self,
        *,
        threshold: float = 0.99,
        model_version: str = "d5.0-logistic",
        weights: np.ndarray | None = None,
        bias: float = 0.0,
    ):
        self.threshold = float(threshold)
        self.model_version = model_version
        self.weights = weights if weights is not None else np.zeros(len(FEATURE_KEYS), dtype=np.float64)
        self.bias = float(bias)
        self._fitted = weights is not None
    def featurize(
        self,
        outputs: Mapping[str, float],
        action: Mapping[str, Any],
        distances: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        mode = int(action.get("caes_mode", 1))
        u_tp = float(action["u_tp"][0] if hasattr(action.get("u_tp"), "__len__") else action.get("u_tp", 1.0))
        u_bat = float(action["u_battery"][0] if hasattr(action.get("u_battery"), "__len__") else action.get("u_battery", 0.0))
        mag = float(
            action["caes_magnitude"][0]
            if hasattr(action.get("caes_magnitude"), "__len__")
            else action.get("caes_magnitude", 0.0)
        )
        dist = distances or {}
        feats = np.asarray(
            [
                float(outputs.get("battery_soc", 0.5)),
                float(outputs.get("caes_gas_soc", 0.8)),
                float(outputs.get("caes_hot_soc", 0.5)),
                float(outputs.get("caes_cold_soc", 0.5)),
                float(outputs.get("caes_gas_pressure", 8.5e6)) / 1e7,
                float(outputs.get("caes_gas_temperature", 300.0)) / 500.0,
                u_tp,
                u_bat,
                1.0 if mode == 0 else 0.0,
                1.0 if mode == 1 else 0.0,
                1.0 if mode == 2 else 0.0,
                mag,
                float(dist.get("battery_soc_to_min", outputs.get("battery_soc", 0.5) - 0.1)),
                float(dist.get("battery_soc_to_max", 0.9 - float(outputs.get("battery_soc", 0.5)))),
                float(dist.get("caes_gas_soc_to_min", float(outputs.get("caes_gas_soc", 0.8)) - 0.6)),
                float(dist.get("caes_gas_soc_to_max", 1.0 - float(outputs.get("caes_gas_soc", 0.8)))),
            ],
            dtype=np.float64,
        )
        return feats
    def predict_proba(self, features: np.ndarray) -> float:
        z = float(np.dot(self.weights, features) + self.bias)
        # stable sigmoid
        if z >= 0:
            ez = np.exp(-z)
            return float(1.0 / (1.0 + ez))
        ez = np.exp(z)
        return float(ez / (1.0 + ez))
    def is_safe(self, outputs: Mapping[str, float], action: Mapping[str, Any], distances: Mapping[str, float] | None = None) -> tuple[bool, float]:
        p = self.predict_proba(self.featurize(outputs, action, distances))
        return p >= self.threshold, p
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        lr: float = 0.05,
        epochs: int = 400,
        l2: float = 1e-3,
        class_weight_unsafe: float = 5.0,
    ) -> "SafetyClassifier":
        """y: 1=safe, 0=unsafe。加重 unsafe 权重以提高 recall。"""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        w = np.zeros(d, dtype=np.float64)
        b = 0.0
        for _ in range(epochs):
            logits = X @ w + b
            # sigmoid
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            weights = np.where(y < 0.5, class_weight_unsafe, 1.0)
            err = (probs - y) * weights
            grad_w = (X.T @ err) / n + l2 * w
            grad_b = float(np.mean(err))
            w -= lr * grad_w
            b -= lr * grad_b
        self.weights = w
        self.bias = b
        self._fitted = True
        return self
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        failure_types: Sequence[str] | None = None,
        threshold: float | None = None,
    ) -> SafetyMetrics:
        thr = float(self.threshold if threshold is None else threshold)
        probs = np.asarray([self.predict_proba(x) for x in X], dtype=np.float64)
        pred_safe = probs >= thr
        y = np.asarray(y, dtype=np.float64)
        # safe=1, unsafe=0
        unsafe = y < 0.5
        safe = ~unsafe
        # predicted safe among unsafe = false safe
        false_safe = pred_safe & unsafe
        false_unsafe = (~pred_safe) & safe
        true_unsafe = (~pred_safe) & unsafe
        n_unsafe = int(np.sum(unsafe))
        n_safe = int(np.sum(safe))
        unsafe_recall = float(np.sum(true_unsafe) / max(n_unsafe, 1))
        pred_unsafe = ~pred_safe
        unsafe_precision = float(np.sum(true_unsafe) / max(int(np.sum(pred_unsafe)), 1))
        false_safe_rate = float(np.sum(false_safe) / max(n_unsafe, 1))
        false_unsafe_rate = float(np.sum(false_unsafe) / max(n_safe, 1))
        per_type: dict[str, float] = {}
        if failure_types is not None:
            ft = list(failure_types)
            for name in sorted(set(ft)):
                if name in ("", "None", None):
                    continue
                idx = np.asarray([i for i, t in enumerate(ft) if t == name], dtype=int)
                if len(idx) == 0:
                    continue
                # among this failure type (all unsafe), recall of flagging unsafe
                per_type[str(name)] = float(np.mean(~pred_safe[idx]))
        return SafetyMetrics(
            unsafe_recall=unsafe_recall,
            unsafe_precision=unsafe_precision,
            false_safe_rate=false_safe_rate,
            false_unsafe_rate=false_unsafe_rate,
            per_failure_type_recall=per_type,
            n_safe=n_safe,
            n_unsafe=n_unsafe,
            threshold=thr,
            model_version=self.model_version,
        )
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_version": self.model_version,
            "threshold": self.threshold,
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "feature_keys": list(FEATURE_KEYS),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    @classmethod
    def load(cls, path: str | Path) -> "SafetyClassifier":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            threshold=float(data.get("threshold", 0.99)),
            model_version=str(data.get("model_version", "d5.0-logistic")),
            weights=np.asarray(data["weights"], dtype=np.float64),
            bias=float(data.get("bias", 0.0)),
        )
class FeasibilityCalibrator:
    """从 SafetyDataset 训练 SafetyClassifier。"""
    def __init__(self, classifier: SafetyClassifier | None = None):
        self.classifier = classifier or SafetyClassifier()
    def fit_from_records(
        self,
        safe_records: Sequence[Mapping[str, Any]],
        fail_records: Sequence[Mapping[str, Any]],
        **fit_kwargs,
    ) -> SafetyClassifier:
        X_list = []
        y_list = []
        for rec in safe_records:
            outputs = rec.get("last_valid_state") or rec.get("previous_observation") or {}
            action = rec.get("hybrid_action") or {}
            X_list.append(self.classifier.featurize(outputs, action, rec.get("distance_to_physical_boundary")))
            y_list.append(1.0)
        for rec in fail_records:
            outputs = rec.get("last_valid_state") or rec.get("previous_observation") or {}
            action = rec.get("hybrid_action") or {}
            X_list.append(self.classifier.featurize(outputs, action, rec.get("distance_to_physical_boundary")))
            y_list.append(0.0)
        if not X_list:
            raise ValueError("无样本可训练 SafetyClassifier")
        X = np.stack(X_list)
        y = np.asarray(y_list, dtype=np.float64)
        self.classifier.fit(X, y, **fit_kwargs)
        return self.classifier
