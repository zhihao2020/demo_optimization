"""项目路径：默认把 runs / 临时计算缓存放到 E: 以减轻 D: 占用。

环境变量（可选覆盖）：
  OPTIMAL_DEMO_CACHE   缓存根，默认 E:/optimal_demo_cache
  OPTIMAL_DEMO_RUNS    runs 根，默认 <CACHE>/runs
  OPTIMAL_DEMO_TMP     临时目录，默认 <CACHE>/tmp
"""

from __future__ import annotations

import os
from pathlib import Path

# 仓库根（src/config → parents[2]）
REPO_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_CACHE = Path("E:/optimal_demo_cache")


def cache_root() -> Path:
    raw = os.environ.get("OPTIMAL_DEMO_CACHE", "").strip()
    root = Path(raw) if raw else _DEFAULT_CACHE
    root.mkdir(parents=True, exist_ok=True)
    return root


def runs_root() -> Path:
    raw = os.environ.get("OPTIMAL_DEMO_RUNS", "").strip()
    if raw:
        p = Path(raw)
    else:
        # 若仓库内 runs 是 junction 指向 E:，优先用仓库相对路径 runs/
        local = REPO_ROOT / "runs"
        if local.exists():
            p = local
        else:
            p = cache_root() / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def tmp_root() -> Path:
    raw = os.environ.get("OPTIMAL_DEMO_TMP", "").strip()
    p = Path(raw) if raw else (cache_root() / "tmp")
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_run_dir(path: str | Path | None, default_name: str = "run") -> Path:
    """相对路径默认写到 runs_root()；绝对路径原样使用。"""
    if path is None:
        out = runs_root() / default_name
    else:
        p = Path(path)
        if p.is_absolute():
            out = p
        else:
            # "runs/foo" → runs_root()/foo ； "foo" → runs_root()/foo
            parts = p.parts
            if parts and parts[0] in ("runs", "./runs"):
                out = runs_root().joinpath(*parts[1:]) if len(parts) > 1 else runs_root()
            else:
                out = runs_root() / p
    out.mkdir(parents=True, exist_ok=True)
    return out


def apply_process_cache_env() -> dict[str, str]:
    """为本进程设置 TMP/TEMP/pycache 等到 E:（训练/基准脚本入口可调用）。"""
    c = cache_root()
    tmp = tmp_root()
    pyc = c / "pycache"
    torch_dir = c / "torch"
    pip_dir = c / "pip"
    for d in (tmp, pyc, torch_dir, pip_dir, c / "fmu_work", c / "logs"):
        d.mkdir(parents=True, exist_ok=True)
    updates = {
        "TEMP": str(tmp),
        "TMP": str(tmp),
        "TMPDIR": str(tmp),
        "PYTHONPYCACHEPREFIX": str(pyc),
        "TORCH_HOME": str(torch_dir),
        "PIP_CACHE_DIR": str(pip_dir),
        "OPTIMAL_DEMO_CACHE": str(c),
        "OPTIMAL_DEMO_RUNS": str(runs_root()),
        "OPTIMAL_DEMO_TMP": str(tmp),
    }
    for k, v in updates.items():
        os.environ[k] = v
    return updates
