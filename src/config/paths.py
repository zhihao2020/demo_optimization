"""项目路径：默认把 runs / 临时计算缓存放到 E: 以减轻 D: 占用。

环境变量（可选覆盖）：
  OPTIMAL_DEMO_CACHE   缓存根，默认 E:/optimal_demo_cache
  OPTIMAL_DEMO_RUNS    runs 根，默认 <CACHE>/runs
  OPTIMAL_DEMO_TMP     临时目录，默认 <CACHE>/tmp
  OPTIMAL_DEMO_JOB_ID  并行 job 标签（如 td3_s0），用于 FMU 副本与 TMP 隔离
  OPTIMAL_DEMO_FMU_ISOLATE  默认 1：每 job 复制一份 .fmu，避免多进程争用同一 ZIP
  OPTIMAL_DEMO_FMU_PATH     显式指定本进程使用的 .fmu（优先于 isolate 复制）
"""

from __future__ import annotations

import os
import shutil
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


def job_id() -> str:
    """并行训练 job 标签；未设置时用 pid，保证进程级隔离。"""
    raw = os.environ.get("OPTIMAL_DEMO_JOB_ID", "").strip()
    if raw:
        # 文件名安全
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in raw)[:80]
    return f"pid{os.getpid()}"


def tmp_root() -> Path:
    raw = os.environ.get("OPTIMAL_DEMO_TMP", "").strip()
    if raw:
        p = Path(raw)
    else:
        # 默认按 job 分子目录，多进程不挤同一 TEMP
        p = cache_root() / "tmp" / job_id()
    p.mkdir(parents=True, exist_ok=True)
    return p


def fmu_copies_root() -> Path:
    p = cache_root() / "fmu_copies"
    p.mkdir(parents=True, exist_ok=True)
    return p


def fmu_isolate_enabled() -> bool:
    raw = os.environ.get("OPTIMAL_DEMO_FMU_ISOLATE", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def resolve_fmu_path(src: str | Path) -> Path:
    """解析本进程应打开的 FMU 路径。

    优先级：
      1. OPTIMAL_DEMO_FMU_PATH（显式副本）
      2. OPTIMAL_DEMO_FMU_ISOLATE=1 时复制到 cache/fmu_copies/<job_id>/
      3. 原始 src

    多 job 并行时复制可避免同时读同一 ZIP、以及杀软对单文件锁。
    """
    override = os.environ.get("OPTIMAL_DEMO_FMU_PATH", "").strip()
    if override:
        p = Path(override)
        if not p.is_file():
            raise FileNotFoundError(f"OPTIMAL_DEMO_FMU_PATH not found: {p}")
        return p.resolve()

    src_p = Path(src)
    if not src_p.is_file():
        # 允许调用方再拼 root；此处不强制
        return src_p

    if not fmu_isolate_enabled():
        return src_p.resolve()

    dest_dir = fmu_copies_root() / job_id()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src_p.name
    try:
        need = (not dest.is_file()) or (dest.stat().st_mtime < src_p.stat().st_mtime) or (
            dest.stat().st_size != src_p.stat().st_size
        )
    except OSError:
        need = True
    if need:
        # 先写临时名再 replace，避免半截文件被并发打开
        tmp = dest.parent / f"{dest.name}.partial.{os.getpid()}"
        shutil.copy2(src_p, tmp)
        os.replace(tmp, dest)
    return dest.resolve()


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
    """为本进程设置 TMP/TEMP/pycache 等到缓存盘（训练/基准脚本入口可调用）。"""
    c = cache_root()
    tmp = tmp_root()
    pyc = c / "pycache"
    torch_dir = c / "torch"
    pip_dir = c / "pip"
    for d in (tmp, pyc, torch_dir, pip_dir, c / "fmu_work", c / "fmu_copies", c / "logs"):
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
        "OPTIMAL_DEMO_FMU_ISOLATE": os.environ.get("OPTIMAL_DEMO_FMU_ISOLATE", "1"),
    }
    if "OPTIMAL_DEMO_JOB_ID" in os.environ:
        updates["OPTIMAL_DEMO_JOB_ID"] = os.environ["OPTIMAL_DEMO_JOB_ID"]
    for k, v in updates.items():
        os.environ[k] = v
    return updates
