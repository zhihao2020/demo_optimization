"""FMU isolate copies are content-addressed and fail-fast on sha256."""
from __future__ import annotations

from pathlib import Path

import pytest

from config.paths import file_sha256, resolve_fmu_path
from training.hybrid_td3.train import _git_commit


def test_isolate_copy_uses_sha_not_mtime(tmp_path, monkeypatch):
    src = tmp_path / "model.fmu"
    src.write_bytes(b"0831-fmu-bytes")
    want = file_sha256(src)
    cache = tmp_path / "cache"
    monkeypatch.setenv("OPTIMAL_DEMO_CACHE", str(cache))
    monkeypatch.setenv("OPTIMAL_DEMO_JOB_ID", "jobA")
    monkeypatch.setenv("OPTIMAL_DEMO_FMU_ISOLATE", "1")
    monkeypatch.delenv("OPTIMAL_DEMO_FMU_PATH", raising=False)
    monkeypatch.delenv("OPTIMAL_DEMO_FMU_SHA256", raising=False)
    dest = resolve_fmu_path(src, expected_sha256=want)
    assert dest.is_file()
    assert file_sha256(dest) == want
    master = cache / "fmu_copies" / want[:16] / "model.fmu"
    assert master.is_file()
    assert file_sha256(master) == want


def test_wrong_expected_sha_raises(tmp_path, monkeypatch):
    src = tmp_path / "model.fmu"
    src.write_bytes(b"0831-fmu-bytes")
    cache = tmp_path / "cache"
    monkeypatch.setenv("OPTIMAL_DEMO_CACHE", str(cache))
    monkeypatch.setenv("OPTIMAL_DEMO_JOB_ID", "jobB")
    monkeypatch.setenv("OPTIMAL_DEMO_FMU_ISOLATE", "1")
    monkeypatch.delenv("OPTIMAL_DEMO_FMU_PATH", raising=False)
    monkeypatch.delenv("OPTIMAL_DEMO_FMU_SHA256", raising=False)
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        resolve_fmu_path(src, expected_sha256="0" * 64)


def test_git_commit_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIMAL_DEMO_GIT_COMMIT", "abc123def456")
    assert _git_commit(tmp_path) == "abc123def456"
