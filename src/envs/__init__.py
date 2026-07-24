"""Gymnasium 电力系统强化学习环境包。

提供混合动作空间、动态可行域与 FMU 物理仿真的 ``PowerSystemEnv``。
"""

__all__ = ["PowerSystemEnv"]  # 电力系统 Gymnasium 环境（延迟导入）


def __getattr__(name: str):
    """延迟导入 ``PowerSystemEnv``，避免循环依赖。"""
    if name == "PowerSystemEnv":
        from .power_system_env import PowerSystemEnv
        return PowerSystemEnv
    raise AttributeError(name)
