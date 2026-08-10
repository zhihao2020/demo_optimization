"""训练子包：HMSD/GHTD3 主线、Hybrid 基线、评估与运行报告。

主线：``training.ghtd3``（连续 CAES z→mode/mag + 目标条件层次 TD3）。
基线：``training.hybrid_td3``、``training.hybrid_sac``（共享 ``hybrid_common``）。
评估：``training.evaluate_td3``；报告：``training.report_policy_run``。
"""
