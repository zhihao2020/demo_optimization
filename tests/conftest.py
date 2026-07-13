"""pytest 配置：把 src/ 加入 path，使 `import fmu` 可用。

运行：`pytest tests/`（在仓库根目录）。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
