"""控制器包：规则基线策略与 compare 开环调度到 FMU 的映射工具。"""

from .compare_schedule import build_plan_for_scheme, load_strategy, to_dispatch_plan, to_fmu_actions
from .rule_based_controller import RuleBasedController

__all__ = [
    "RuleBasedController",  # 规则基线控制器(RuleBasedController)
    "load_strategy",  # 加载 compare 方案序列(load_strategy)
    "to_fmu_actions",  # compare 序列符号翻转并校验(to_fmu_actions)
    "to_dispatch_plan",  # 构造 FMU 调度计划(to_dispatch_plan)
    "build_plan_for_scheme",  # 一站式生成 DispatchPlan(build_plan_for_scheme)
]
