"""
Unit tests for the diarization device selection shared by every backend:
"auto" takes CUDA only when it exists and works, "cuda" that is missing or
broken falls back to the CPU with a warning and never raises, "cpu" never
touches the GPU.
"""

import builtins
from unittest.mock import MagicMock

import pytest

import src.shared.utils.diarization_backend.diarization_backend as backend_module
from src.shared.logger import Logger
from src.shared.utils.diarization_backend import select_device
from src.shared.utils.diarization_backend.diarization_backend import (
    resolve_device_preference,
)


@pytest.fixture(name="log")
def log_fixture():
    """A logger stub"""
    return MagicMock(spec=Logger)


def test_cpu_is_taken_without_looking_at_cuda(log, monkeypatch):
    """Cpu is taken without looking at cuda"""
    probe = MagicMock()
    monkeypatch.setattr(
        backend_module,
        "cuda_available",
        MagicMock(side_effect=AssertionError("must not be called")),
    )

    selection = select_device("cpu", log, probe=probe)

    assert selection.device == "cpu"
    assert selection.fallback is False
    probe.assert_not_called()
    log.warning.assert_not_called()


def test_auto_without_cuda_is_cpu_without_a_warning(log, monkeypatch):
    """Auto without cuda is cpu without a warning"""
    monkeypatch.setattr(backend_module, "cuda_available", lambda: False)

    selection = select_device("auto", log, probe=MagicMock())

    assert selection.device == "cpu"
    assert selection.fallback is False
    log.warning.assert_not_called()
    assert selection.as_runtime_info()["configured_device"] == "auto"


def test_auto_with_a_working_cuda_device_is_cuda(log, monkeypatch):
    """Auto with a working cuda device is cuda"""
    monkeypatch.setattr(backend_module, "cuda_available", lambda: True)
    probe = MagicMock()

    selection = select_device("auto", log, probe=probe)

    assert selection.device == "cuda"
    assert selection.fallback is False
    probe.assert_called_once_with("cuda")


def test_cuda_requested_but_unavailable_falls_back_with_a_warning(
    log, monkeypatch
):
    """Cuda requested but unavailable falls back with a warning"""
    monkeypatch.setattr(backend_module, "cuda_available", lambda: False)

    selection = select_device("cuda", log, probe=MagicMock())

    assert selection.device == "cpu"
    assert selection.fallback is True
    assert "is_available" in (selection.reason or "")
    log.warning.assert_called_once()
    assert "falling back to CPU" in log.warning.call_args[0][0]
    info = selection.as_runtime_info()
    assert info["device_fallback"] is True
    assert info["device"] == "cpu"
    assert info["configured_device"] == "cuda"


@pytest.mark.parametrize("configured", ["auto", "cuda"])
def test_a_cuda_device_that_fails_its_probe_falls_back(
    log, monkeypatch, configured
):
    """A cuda device that fails its probe falls back"""
    monkeypatch.setattr(backend_module, "cuda_available", lambda: True)

    def broken(device):
        raise RuntimeError(f"CUDA error: out of memory on {device}")

    selection = select_device(configured, log, probe=broken)

    assert selection.device == "cpu"
    assert selection.fallback is True
    assert "out of memory" in (selection.reason or "")
    log.warning.assert_called_once()


def test_resolve_device_preference_reports_what_auto_would_pick(monkeypatch):
    """Resolve device preference reports what auto would pick"""
    monkeypatch.setattr(backend_module, "cuda_available", lambda: False)
    assert resolve_device_preference("auto") == "cpu"
    monkeypatch.setattr(backend_module, "cuda_available", lambda: True)
    assert resolve_device_preference("auto") == "cuda"
    assert resolve_device_preference("cpu") == "cpu"
    assert resolve_device_preference("cuda") == "cuda"


def test_cuda_available_is_false_when_torch_is_missing(monkeypatch):
    """Cuda available is false when torch is missing"""
    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    assert backend_module.cuda_available() is False
