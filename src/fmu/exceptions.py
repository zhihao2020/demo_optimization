"""FMU 求解与生命周期相关异常。"""


class FmuSolverError(RuntimeError):
    """FMU 求解器错误(FmuSolverError)。

    涵盖 FMU 生命周期、通信步或读输出失败。
    """
