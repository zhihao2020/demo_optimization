"""LEGACY：普通 Box + SB3 TD3。不得用于正式训练。

CAES 合法集合非凸，普通连续 Box 无法表达。请使用混合动作策略
（``src/training/hybrid_td3``）。
"""

from __future__ import annotations

import json
from pathlib import Path

LEGACY_ERROR = (
    "普通连续 Box 无法表达 CAES 非凸动作集合，请使用 hybrid action policy。"
)


def run_smoke(*_args, allow_legacy_smoke: bool = False, **_kwargs) -> dict:
    """保留入口但始终返回 blocked 状态，不启动 SB3 Box TD3。

    Args:
        allow_legacy_smoke: 是否标记允许遗留冒烟（不影响 blocked 结果）。
        **_kwargs: 可选 ``run_dir`` 等，写入 summary。

    Returns:
        含 ``status``、``error`` 与 ``note`` 的字典。
    """
    # 即便 allow_legacy_smoke=True，Dict 混合动作空间也无法启动 SB3 TD3
    result = {
        "status": "blocked_legacy_box_td3",
        "error": LEGACY_ERROR,
        "note": "PowerSystemEnv.action_space 现为 Dict 混合动作；正式算法见 training.hybrid_td3",
        "allow_legacy_smoke": bool(allow_legacy_smoke),
    }
    run_dir = Path(_kwargs.get("run_dir", "runs/td3_smoke_legacy"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
