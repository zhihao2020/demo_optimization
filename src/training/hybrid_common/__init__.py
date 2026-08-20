"""Hybrid 算法共用模块：随机 Actor、评估落盘、策略包装。

导出:
    HybridStochasticActor — 混合随机 Actor
    HybridGiveSafePolicyWrapper — GiveSafe 评估策略包装
    SoftShellGiveSafePolicy — 可选软外壳的 GiveSafe 包装别名
    RandomFeasiblePolicy — 随机可行动作探索策略
    prepare_run_dir — 创建运行目录并复制配置
    finalize_training_run — 训练后评估与 summary 落盘
    write_summary_and_report — 写 summary 并生成可读报告
"""

from .eval_and_save import (
    finalize_training_run,
    parameter_profile_fields,
    prepare_run_dir,
    write_summary_and_report,
)
from .policy_wrapper import (
    HybridGiveSafePolicyWrapper,
    RandomFeasiblePolicy,
    SoftShellGiveSafePolicy,
)
from .stochastic_actor import HybridStochasticActor

__all__ = [
    "HybridStochasticActor",
    "HybridGiveSafePolicyWrapper",
    "SoftShellGiveSafePolicy",
    "RandomFeasiblePolicy",
    "prepare_run_dir",
    "parameter_profile_fields",
    "finalize_training_run",
    "write_summary_and_report",
]
