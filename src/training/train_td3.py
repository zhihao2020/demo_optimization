"""LEGACY：普通 Box + SB3 TD3。不得用于正式训练。

CAES 合法集合非凸，普通连续 Box 无法表达。请使用 hybrid action policy
（src/training/hybrid_td3）。
"""

from __future__ import annotations

import json
from pathlib import Path

LEGACY_ERROR = (
    "普通连续 Box 无法表达 CAES 非凸动作集合，请使用 hybrid action policy。"
)


def _assert_not_nonconvex_caes_formal(allow_legacy_smoke: bool = False) -> None:
    """检测到非凸 CAES 动作集合时禁止正式普通 TD3。"""
    if allow_legacy_smoke:
        return
    raise RuntimeError(LEGACY_ERROR)


def run_smoke(*_args, allow_legacy_smoke: bool = False, **_kwargs) -> dict:
    """保留入口但默认 fail-fast；仅显式 allow_legacy_smoke=True 时提示已禁用。"""
    _assert_not_nonconvex_caes_formal(allow_legacy_smoke=allow_legacy_smoke)
    # 即便允许 smoke 标记，也因动作空间已改为 Dict 而无法启动 SB3 TD3
    result = {
        "status": "blocked_legacy_box_td3",
        "error": LEGACY_ERROR,
        "note": "PowerSystemEnv.action_space 现为 Dict 混合动作；正式算法见 training.hybrid_td3",
    }
    run_dir = Path(_kwargs.get("run_dir", "runs/td3_smoke_legacy"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_formal_training(*_args, **_kwargs) -> None:
    """正式训练入口：始终拒绝。"""
    raise RuntimeError(LEGACY_ERROR)
