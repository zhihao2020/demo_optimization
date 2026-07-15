"""Gymnasium 电力系统环境包（对外主类：PowerSystemEnv）。"""

__all__ = ["PowerSystemEnv"]


def __getattr__(name: str):
    if name == "PowerSystemEnv":
        from .power_system_env import PowerSystemEnv
        return PowerSystemEnv
    raise AttributeError(name)
