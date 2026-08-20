"""训练子包：扁平混合 SAC/TD3 主线，GHTD3 为可选分支。

主线：``training.hybrid_sac``（压空 mode+mag 参数化动作）。
对照：``training.hybrid_td3``（同动作表示，确定性）；``training.ghtd3`` 仅当 168 h 仍 abort。
评估：``training.evaluate_td3``；报告：``training.report_policy_run``。
"""
