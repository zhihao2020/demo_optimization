"""从 SafetyDataset / failure_records 训练 SafetyClassifier，并报告 false-safe rate。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from actions.safety_classifier import FeasibilityCalibrator, SafetyClassifier
from training.hybrid_td3.buffer import SafetyDataset
def main(
    dataset_path: str = "runs/feasibility_probe/train/safety_dataset.json",
    out_dir: str = "runs/safety_classifier",
    threshold: float = 0.99,
) -> dict:
    ds_path = ROOT / dataset_path
    if not ds_path.exists():
        raise SystemExit(f"缺少 SafetyDataset: {ds_path}；请先运行 scripts/feasibility_probe.py")
    ds = SafetyDataset.load(ds_path)
    safe, fail = ds.split_safe_fail()
    if len(fail) == 0:
        # 合成少量近界 unsafe 以便报告管线可跑（仅当真实失败为 0）
        print("WARNING: 无 post-step fail 样本；classifier 将主要拟合 safe 分布")
    calibrator = FeasibilityCalibrator(SafetyClassifier(threshold=threshold, model_version="d5.1-logistic"))
    clf = calibrator.fit_from_records(safe, fail, epochs=500, class_weight_unsafe=8.0)
    X, y, ftypes = [], [], []
    for rec in safe:
        X.append(clf.featurize(rec.get("previous_observation") or {}, rec.get("hybrid_action") or {}, rec.get("distance_to_physical_boundary")))
        y.append(1.0)
        ftypes.append("")
    for rec in fail:
        X.append(clf.featurize(rec.get("previous_observation") or {}, rec.get("hybrid_action") or {}, rec.get("distance_to_physical_boundary")))
        y.append(0.0)
        ftypes.append(str(rec.get("fine_failure_type") or "unknown"))
    X_arr = np.stack(X) if X else np.zeros((0, 16))
    y_arr = np.asarray(y, dtype=np.float64)
    metrics = clf.evaluate(X_arr, y_arr, failure_types=ftypes, threshold=threshold)
    out = ROOT / out_dir
    out.mkdir(parents=True, exist_ok=True)
    clf.save(out / "safety_classifier.json")
    report = metrics.to_dict()
    report["n_safe_train"] = len(safe)
    report["n_fail_train"] = len(fail)
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "metrics.yaml").write_text(yaml.safe_dump(report, allow_unicode=True), encoding="utf-8")
    print(yaml.safe_dump(report, allow_unicode=True))
    return report
if __name__ == "__main__":
    main()
