"""CPU device pin for FS-HSAC seasonal jobs."""

from __future__ import annotations

from training.fs_hsac.compute import configure_torch_compute


def test_configure_torch_compute_cpu(monkeypatch):
    monkeypatch.setenv("OPTIMAL_DEMO_DEVICE", "cpu")
    monkeypatch.setenv("OPTIMAL_DEMO_TORCH_THREADS", "2")
    spec = configure_torch_compute()
    assert spec["device"] == "cpu"
    assert int(spec["threads"]) == 2
