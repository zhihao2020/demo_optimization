"""FMI 3.0 固定步长会话：Python 与 FMU 之间的唯一执行路径。

职责边界：
- FMU：物理状态演化与真实物理量输出（功率、SOC、压力、温度等）。
- Python（本模块）：写输入前校验上下限，读输出后检测数值/物理异常。
- 不做市场结算、reward 或经济惩罚（后续模块再接）。

典型流程：reset →（循环：set_inputs → doStep → read）→ close。
校验发生在写 FMU 之前、读 FMU 之后。

输入/输出边界见 docs/fmu_input_bounds.md 与 Modelica 顶层接口。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from fmpy import extract, read_model_description
from fmpy.fmi3 import FMU3Slave

from .types import DispatchPlan, SimulationResult
from .validate import validate_inputs, validate_outputs

# 调度输入：无量纲指令，直接对应 Modelica RealInput。
ACTION_NAMES = ("u_tp", "u_battery", "u_caes")

# 外部边界输入：物理单位；与 actions 严格分离，不得混入动作字典。
# 只写顶层四个端口；Sysplorer 导出里那些内部同名 input（如 wind.v_in）一律不碰。
BOUNDARY_NAMES = (
    "v_wind_in",
    "g_irradiance_in",
    "t_air_in",
    "p_load_plan_in",
)

# 顶层物理及累计经济输出（与 PowerSystem_8760h RealOutput 一一对应；全部必需）。
# 功率单位 W；压力 Pa；温度 K；SOC ∈ [0,1]。
# 符号约定：发电为负、用电/充电为正（与电气接口一致）。
DEFAULT_OUTPUTS = (
    # 功率不平衡：拆分后的非负弃电 / 缺供
    "p_curtailment",
    "p_unserved",
    # 储能状态
    "battery_soc",
    "caes_gas_soc",
    "caes_hot_soc",
    "caes_cold_soc",
    # 设备实际功率
    "p_thermal",
    "p_battery",
    "p_caes",
    "p_grid",
    # 风光可用（P_plan）与并网实际（P_act）
    "p_wind_available",
    "p_wind_actual",
    "p_pv_available",
    "p_pv_actual",
    # 负荷实际供给
    "p_load_actual",
    # CAES 热力状态（observation / 边界诊断）
    "caes_gas_pressure",
    "caes_gas_temperature",
    "caes_hot_temperature",
    "caes_cold_temperature",
    # Modelica 累计现金流：正=收益、负=成本，Python 只做差分/审计。
    "economic_cashflow_total",
    "economic_cashflow_wind",
    "economic_cashflow_pv",
    "economic_cashflow_thermal",
    "economic_cashflow_battery",
    "economic_cashflow_caes",
    "economic_cashflow_load",
    "economic_cashflow_grid",
)

# 内嵌表参考输出：仅回归验收用，不进 RL observation。
BOUNDARY_REF_OUTPUTS = (
    "v_wind_ref",
    "g_irradiance_ref",
    "t_air_ref",
    "p_load_plan_ref",
)

# 与模型 start 一致的默认初值（须落在允许输入集合内）
DEFAULT_INITIAL_INPUTS = {"u_tp": 1.0, "u_battery": 0.0, "u_caes": 0.0}
DEFAULT_INITIAL_BOUNDARIES = {
    "v_wind_in": 1.45,
    "g_irradiance_in": 0.0,
    "t_air_in": 262.4,
    "p_load_plan_in": 2.14e8,
}


def fmu_platform_supported(fmu_path: Path) -> bool:
    """判断当前 OS 是否能在 FMU 包中找到对应原生二进制。

    Args:
        fmu_path: ``.fmu`` 压缩包路径。

    Returns:
        存在匹配 ``binaries/`` 下 Windows ``.dll`` 或 Linux ``.so`` 时为 ``True``。
    """
    required = ".dll" if sys.platform.startswith("win") else ".so"
    marker = "windows" if sys.platform.startswith("win") else "linux"
    with zipfile.ZipFile(fmu_path) as archive:
        return any(
            name.startswith("binaries/")
            and marker in name
            and name.endswith(required)
            for name in archive.namelist()
        )


class FmuSession:
    """FMI 会话(FmuSession)：固定步长 Co-Simulation 的唯一执行路径。

    构造时解析 modelDescription 并锁定 VR；每次 ``reset()`` 重新实例化，
    在初始化模式写入 ``initial_inputs``。
    """

    def __init__(
        self,
        fmu_path: Path,
        step_size: float = 3600.0,
        initial_inputs: dict[str, float] | None = None,
        initial_boundaries: dict[str, float] | None = None,
        outputs: tuple[str, ...] | list[str] = DEFAULT_OUTPUTS,
        *,
        require_boundaries: bool = True,
    ) -> None:
        """创建会话并校验 FMU 与初始输入。

        Args:
            fmu_path: ``.fmu`` 文件路径。
            step_size: 通信步长（秒）。
            initial_inputs: 初始调度输入；默认 ``DEFAULT_INITIAL_INPUTS``。
            initial_boundaries: 初始边界输入；默认 ``DEFAULT_INITIAL_BOUNDARIES``。
            outputs: 每步读取的 FMU 输出名列表。
            require_boundaries: 为 True 时要求 FMU 含四个边界输入（新导出）；
                旧 FMU 可设 False 以兼容过渡期。

        Raises:
            FileNotFoundError: FMU 文件不存在。
            RuntimeError: 当前平台无原生二进制或不支持 Co-Simulation。
            KeyError: 请求的动作/输出变量在 FMU 中缺失。
            ValueError: 初始输入越界（由 ``validate_inputs`` 抛出）。
        """
        self.fmu_path = Path(fmu_path)
        if not self.fmu_path.exists():
            raise FileNotFoundError(self.fmu_path)
        if not fmu_platform_supported(self.fmu_path):
            host = "Windows" if sys.platform.startswith("win") else "Linux"
            raise RuntimeError(
                f"FMU has no native {host} binary: {self.fmu_path}. "
                "Build/export this FMU for the server platform before running it."
            )
        self.step_size = float(step_size)
        self.initial_inputs = dict(initial_inputs or DEFAULT_INITIAL_INPUTS)
        self.initial_boundaries = dict(
            initial_boundaries or DEFAULT_INITIAL_BOUNDARIES
        )
        self.require_boundaries = bool(require_boundaries)
        # 初始输入也必须合法，避免带着越界值进入仿真
        validate_inputs(self.initial_inputs)

        requested_outputs = tuple(outputs)
        self._md = read_model_description(str(self.fmu_path))
        if self._md.coSimulation is None:
            raise RuntimeError("FMU does not support Co-Simulation")
        self._vrs = {
            variable.name: variable.valueReference
            for variable in self._md.modelVariables
        }

        missing = [name for name in ACTION_NAMES if name not in self._vrs]
        missing.extend(name for name in requested_outputs if name not in self._vrs)
        if self.require_boundaries:
            missing.extend(name for name in BOUNDARY_NAMES if name not in self._vrs)
        if missing:
            raise KeyError(f"FMU variables missing: {missing}")

        self._has_boundaries = all(name in self._vrs for name in BOUNDARY_NAMES)
        self.outputs = requested_outputs
        self._read_vrs = [self._vrs[name] for name in self.outputs]
        self._unzipdir: str | None = None
        self._fmu: FMU3Slave | None = None
        self.time = 0.0

    def _ensure_extracted(self) -> str:
        """解压 FMU 包（懒加载，仅一次）。

        解压到 **本会话私有** 目录（在 OPTIMAL_DEMO_TMP / job TEMP 下 mkdtemp），
        避免多进程共写同一解压树；配合 resolve_fmu_path 的每 job .fmu 副本。

        Returns:
            解压目录路径字符串。
        """
        if self._unzipdir is None:
            # 优先 job 隔离 TMP，否则系统临时目录
            base = os.environ.get("OPTIMAL_DEMO_TMP") or os.environ.get("TEMP") or os.environ.get("TMP")
            parent = Path(base) if base else Path(tempfile.gettempdir())
            parent = parent / "fmu_unzip"
            parent.mkdir(parents=True, exist_ok=True)
            unzipdir = tempfile.mkdtemp(prefix="fmu_", dir=str(parent))
            self._unzipdir = str(extract(str(self.fmu_path), unzipdir=unzipdir))
        return self._unzipdir

    def reset(
        self,
        start_time: float = 0.0,
        boundaries: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """重新实例化 FMU，写入初始动作与边界，返回 t=start_time 的输出快照。

        Args:
            start_time: 仿真起始时刻（秒）。
            boundaries: 可选，覆盖 ``initial_boundaries`` 的边界字典。

        Returns:
            请求输出名 -> 标量值的字典（已通过 ``validate_outputs``）。

        Raises:
            RuntimeError: FMI 实例化或初始化失败（由 fmpy 抛出）。
            ValueError: 读出的输出不合理。
        """
        self._release_instance()
        self._fmu = FMU3Slave(
            guid=self._md.guid,
            unzipDirectory=self._ensure_extracted(),
            modelIdentifier=self._md.coSimulation.modelIdentifier,
            instanceName="power_dispatch_session",
        )
        self._fmu.instantiate()
        self._fmu.enterInitializationMode(startTime=float(start_time))
        self.set_inputs(self.initial_inputs)
        self.set_boundaries(boundaries if boundaries is not None else self.initial_boundaries)
        self._fmu.exitInitializationMode()
        self.time = float(start_time)
        return self.read()

    def set_inputs(self, action: dict[str, float]) -> None:
        """写调度输入到 FMU；先校验上下限。不含边界。

        Args:
            action: 含 ``u_tp``、``u_battery``、``u_caes`` 的字典。

        Raises:
            RuntimeError: 未先 ``reset()``。
            ValueError: 输入越界或非有限。
        """
        if self._fmu is None:
            raise RuntimeError("call reset() before set_inputs()")
        validate_inputs(action)
        for name in ACTION_NAMES:
            self._fmu.setFloat64([self._vrs[name]], [float(action[name])])

    def set_boundaries(self, boundaries: dict[str, float]) -> None:
        """写外部边界输入；与动作严格分离。

        Args:
            boundaries: 含四个边界 FMU 名的字典；允许缺省（无边界端口的旧 FMU）。

        Raises:
            RuntimeError: 未先进入实例化，或新 FMU 缺少边界却要求写入。
            ValueError: 边界非有限或缺键。
        """
        if self._fmu is None:
            raise RuntimeError("call reset()/enterInitializationMode before set_boundaries()")
        if not self._has_boundaries:
            if self.require_boundaries:
                raise RuntimeError("FMU 缺少边界输入端口，无法写入")
            return
        missing = [name for name in BOUNDARY_NAMES if name not in boundaries]
        if missing:
            raise ValueError(f"边界字典缺少键: {missing}")
        for name in BOUNDARY_NAMES:
            value = float(boundaries[name])
            if not np.isfinite(value):
                raise ValueError(f"边界 {name}={value} 非有限")
            self._fmu.setFloat64([self._vrs[name]], [value])

    def read(self) -> dict[str, float]:
        """读物理输出并校验数值/物理合理性。

        Returns:
            请求输出名 -> 标量值的字典。

        Raises:
            RuntimeError: 未先 ``reset()``。
            ValueError: 非有限或物理不合理。
        """
        if self._fmu is None:
            raise RuntimeError("call reset() before read()")
        values = self._fmu.getFloat64(self._read_vrs)
        result = dict(zip(self.outputs, (float(value) for value in values)))
        validate_outputs(result)
        return result

    def step(
        self,
        action: dict[str, float],
        boundaries: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """执行一步：写动作与边界 → doStep → 读并校验输出。

        Args:
            action: 本通信步的调度输入。
            boundaries: 本通信步的边界输入；新 FMU 必填。

        Returns:
            步后 FMU 输出字典。

        Raises:
            RuntimeError: 未先 ``reset()``。
            ValueError: 输入或输出校验失败。
        """
        if self._fmu is None:
            raise RuntimeError("call reset() before step()")
        self.set_inputs(action)
        if boundaries is not None:
            self.set_boundaries(boundaries)
        elif self._has_boundaries and self.require_boundaries:
            raise ValueError("新 FMU 每步必须提供 boundaries，否则边界会冻在上一值")
        self._fmu.doStep(
            currentCommunicationPoint=self.time,
            communicationStepSize=self.step_size,
        )
        self.time += self.step_size
        return self.read()

    def rollout(
        self,
        plan: DispatchPlan,
        horizon_hours: int | None = None,
        start_time: float | None = None,
        boundaries_at=None,
    ) -> SimulationResult:
        """按计划滚动仿真；单步失败时记录 metadata，不向外抛出。

        Args:
            plan: 调度计划(DispatchPlan)。
            horizon_hours: 仿真小时数；默认 ``len(plan.time) - 1``。
            start_time: 起始时刻；默认 ``plan.time[0]``。
            boundaries_at: 可选，``Callable[[float], dict[str, float]]``，
                按当前通信点返回边界；新 FMU 必填。

        Returns:
            含时间序列、变量轨迹与执行 metadata 的 SimulationResult。
        """
        hours = int(horizon_hours if horizon_hours is not None else len(plan.time) - 1)
        start = float(plan.time[0] if start_time is None else start_time)
        b0 = boundaries_at(start) if boundaries_at is not None else None
        initial = self.reset(start, boundaries=b0)
        records = {name: [float(initial[name])] for name in self.outputs}
        times = [start]
        simulation_failed = False
        error: str | None = None
        completed = 0
        try:
            for index in range(hours):
                t = float(self.time)
                boundaries = boundaries_at(t) if boundaries_at is not None else None
                out = self.step(
                    {
                        "u_tp": float(plan.u_tp[index]),
                        "u_battery": float(plan.u_battery[index]),
                        "u_caes": float(plan.u_caes[index]),
                    },
                    boundaries=boundaries,
                )
                completed += 1
                times.append(self.time)
                for name in self.outputs:
                    records[name].append(float(out[name]))
        except Exception as exc:
            simulation_failed = True
            error = str(exc)
        return SimulationResult(
            time=np.asarray(times, dtype=float),
            variables={
                name: np.asarray(values, dtype=float) for name, values in records.items()
            },
            metadata={
                "execution": "FmuSession",
                "initial_inputs": dict(self.initial_inputs),
                "hours_done": completed,
                "horizon_hours": hours,
                "simulation_failed": simulation_failed,
                "error": error,
            },
        )

    def _release_instance(self) -> None:
        """终止并释放当前 FMI 实例（忽略 terminate 异常）。"""
        if self._fmu is None:
            return
        try:
            self._fmu.terminate()
        except Exception:
            pass
        self._fmu.freeInstance()
        self._fmu = None

    def close(self) -> None:
        """释放 FMI 实例并清理解压目录。"""
        self._release_instance()
        if self._unzipdir is not None:
            shutil.rmtree(self._unzipdir, ignore_errors=True)
            self._unzipdir = None

    def __enter__(self) -> "FmuSession":
        """上下文管理器入口。

        Returns:
            自身实例。
        """
        return self

    def __exit__(self, *_args) -> None:
        """上下文管理器退出时关闭会话。"""
        self.close()
