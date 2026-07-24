"""pytest 配置(conftest)：将 src/ 加入 sys.path，使 `import fmu` 等包可用。

本文件在 pytest 收集测试前自动加载；无需手动设置 PYTHONPATH。
运行方式：在仓库根目录执行 `pytest tests/`。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
