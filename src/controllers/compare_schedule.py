"""将 compare/ 开环调度序列映射为 FMU DispatchPlan。

compare 注释约定：电池/CAES 负充正放；本仓库 FMU 为正充负放。
映射时对 battery_seq、caes_seq 取反；落在禁区则直接报错，不做投影。
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from fmu.types import DispatchPlan
from fmu.validate import validate_inputs

SchemeName = Literal["output1", "output2"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEME_MODULES: dict[SchemeName, str] = {
    "output1": "compare.output1",
    "output2": "compare.output2",
}


def _ensure_repo_on_path() -> None:
    import sys

    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_strategy(scheme: SchemeName, steps: int = 8760) -> dict[str, list[float]]:
    """加载 compare 方案，返回 {caes_seq, battery_seq, tp_seq}。"""
    if scheme not in _SCHEME_MODULES:
        raise ValueError(f"未知 scheme={scheme!r}，可选: {sorted(_SCHEME_MODULES)}")
    _ensure_repo_on_path()
    module = importlib.import_module(_SCHEME_MODULES[scheme])
    caes_seq, battery_seq, tp_seq = module.generate_energy_strategy(steps=int(steps))
    return {
        "caes_seq": list(caes_seq),
        "battery_seq": list(battery_seq),
        "tp_seq": list(tp_seq),
    }


def to_fmu_actions(
    caes_seq: Sequence[float],
    battery_seq: Sequence[float],
    tp_seq: Sequence[float],
    *,
    scheme: str | None = None,
) -> dict[str, np.ndarray]:
    """符号翻转后逐小时 validate_inputs；非法则 ValueError。"""
    n = len(tp_seq)
    if len(caes_seq) != n or len(battery_seq) != n:
        raise ValueError(
            f"序列长度不一致: caes={len(caes_seq)}, battery={len(battery_seq)}, tp={n}"
        )
    u_tp = np.asarray(tp_seq, dtype=float)
    # compare: 负充正放 → FMU: 正充负放
    u_battery = -np.asarray(battery_seq, dtype=float)
    u_caes = -np.asarray(caes_seq, dtype=float)

    prefix = f"scheme={scheme}: " if scheme else ""
    for h in range(n):
        action = {
            "u_tp": float(u_tp[h]),
            "u_battery": float(u_battery[h]),
            "u_caes": float(u_caes[h]),
        }
        try:
            validate_inputs(action)
        except ValueError as exc:
            raise ValueError(
                f"{prefix}hour={h}: compare 动作映射后不合法 "
                f"(caes_seq={float(caes_seq[h])}, battery_seq={float(battery_seq[h])}, "
                f"tp_seq={float(tp_seq[h])} → "
                f"u_tp={action['u_tp']}, u_battery={action['u_battery']}, "
                f"u_caes={action['u_caes']}): {exc}"
            ) from exc

    return {"u_tp": u_tp, "u_battery": u_battery, "u_caes": u_caes}


def to_dispatch_plan(
    strategy: Mapping[str, Sequence[float]],
    *,
    scheme: str | None = None,
    hours: int | None = None,
    step_seconds: float = 3600.0,
) -> DispatchPlan:
    """校验通过后构造 DispatchPlan；hours 可截断前缀小时数。"""
    caes = list(strategy["caes_seq"])
    battery = list(strategy["battery_seq"])
    tp = list(strategy["tp_seq"])
    n = len(tp)
    use_hours = n if hours is None else int(hours)
    if use_hours <= 0:
        raise ValueError(f"hours 必须为正，得到 {use_hours}")
    if use_hours > n:
        raise ValueError(f"hours={use_hours} 超过序列长度 {n}")
    caes = caes[:use_hours]
    battery = battery[:use_hours]
    tp = tp[:use_hours]

    actions = to_fmu_actions(caes, battery, tp, scheme=scheme)
    time = np.arange(0, use_hours + 1, dtype=float) * float(step_seconds)
    return DispatchPlan(
        time=time,
        u_tp=actions["u_tp"],
        u_battery=actions["u_battery"],
        u_caes=actions["u_caes"],
    )


def build_plan_for_scheme(
    scheme: SchemeName,
    *,
    hours: int = 8760,
    step_seconds: float = 3600.0,
) -> tuple[DispatchPlan, dict[str, Any]]:
    """一站式：加载 → 截断 → 校验 → DispatchPlan。

    默认按 8760h 生成全年策略再取前缀；``hours > 8760`` 时按更长步数生成。
    """
    hours = int(hours)
    gen_steps = max(hours, 8760)
    strategy = load_strategy(scheme, steps=gen_steps)
    plan = to_dispatch_plan(
        strategy, scheme=scheme, hours=hours, step_seconds=step_seconds
    )
    meta = {
        "scheme": scheme,
        "hours": hours,
        "step_seconds": float(step_seconds),
        "generated_steps": gen_steps,
    }
    return plan, meta
