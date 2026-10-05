"""
Unit tests for PyannoteDiarizationContext's start-up behaviour without
loading a model: where the model is loaded from (baked directory, local
path, HuggingFace cache, HuggingFace with a token), the fail-fast messages
for a missing install, a missing token and a model that does not load, and
the device selection plus CUDA fallback reported to the pool.
"""

# pylint: disable=protected-access

import builtins
import sys
import types
from unittest.mock import MagicMock

import pytest

import src.shared.utils.diarization_backend.diarization_backend as backend_module
from src.shared.logger import Logger
from src.shared.utils.diarization_backend import DiarizationContextInterface
from src.transcription_contexts.pyannote_diarization_context import (
    PyannoteDiarizationContext,
    PyannoteDiarizationService,
)
from src.transcription_contexts.pyannote_diarization_context import (
    pyannote_diarization_context as context_module,
)

TAG = ["pyannote_diarization"]


@pytest.fixture(name="log")
def log_fixture():
    """A logger stub"""
    return MagicMock(spec=Logger)


@pytest.fixture(name="no_token", autouse=True)
def no_token_fixture(monkeypatch):
    """No token, no baked model, no offline mode unless a test sets them"""
    monkeypatch.delenv("HUGGINGFACE_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv(context_module.MODEL_DIR_ENV_VAR, raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)


@pytest.fixture(name="fake_pyannote")
def fake_pyannote_fixture(monkeypatch):
    """
    A `pyannote.audio` module whose Pipeline.from_pretrained is a mock, so
    `create` runs without the real package or a model.
    """
    pipeline = MagicMock(name="pipeline")
    pipeline_class = MagicMock(name="Pipeline")
    pipeline_class.from_pretrained.return_value = pipeline
    audio = types.ModuleType("pyannote.audio")
    setattr(audio, "Pipeline", pipeline_class)
    package = types.ModuleType("pyannote")
    setattr(package, "audio", audio)
    monkeypatch.setitem(sys.modules, "pyannote", package)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio)
    # No GPU in the test environment, whatever the host has.
    monkeypatch.setattr(backend_module, "cuda_available", lambda: False)
    return pipeline_class, pipeline


def _baked_dir(tmp_path):
    model_dir = tmp_path / "model"
    (model_dir / "segmentation").mkdir(parents=True)
    (model_dir / "embedding").mkdir()
    (model_dir / "config.yaml").write_text("pipeline: {}\n", encoding="utf-8")
    return model_dir


def test_it_is_a_diarization_context_with_the_cpu_default():
    """It is a diarization context with the cpu default"""
    context = PyannoteDiarizationContext({}, TAG)
    assert isinstance(context, DiarizationContextInterface)
    assert context.configured_device == "cpu"
    assert context.device == "cpu"


def test_device_accepts_auto_cpu_cuda_only():
    """Device accepts auto cpu cuda only"""
    for device in ("auto", "cpu", "cuda"):
        assert PyannoteDiarizationContext({"device": device}, TAG)
    with pytest.raises(ValueError):
        PyannoteDiarizationContext({"device": "mps"}, TAG)


def test_auto_reports_the_device_it_would_pick(monkeypatch):
    """Auto reports the device it would pick"""
    monkeypatch.setattr(backend_module, "cuda_available", lambda: False)
    assert PyannoteDiarizationContext({"device": "auto"}, TAG).device == "cpu"
    monkeypatch.setattr(backend_module, "cuda_available", lambda: True)
    assert PyannoteDiarizationContext({"device": "auto"}, TAG).device == "cuda"


def test_model_source_prefers_a_configured_local_directory(tmp_path):
    """Model source prefers a configured local directory"""
    model_dir = _baked_dir(tmp_path)
    context = PyannoteDiarizationContext({"model": str(model_dir)}, TAG)
    assert context.resolve_model_source() == (str(model_dir), "local_dir", None)


def test_model_source_uses_the_baked_directory_for_the_default_model(
    tmp_path, monkeypatch
):
    """Model source uses the baked directory for the default model"""
    model_dir = _baked_dir(tmp_path)
    monkeypatch.setenv(context_module.MODEL_DIR_ENV_VAR, str(model_dir))
    context = PyannoteDiarizationContext({}, TAG)
    assert context.resolve_model_source() == (str(model_dir), "local_dir", None)
    # Another model id ignores the baked directory and needs the token.
    other = PyannoteDiarizationContext({"model": "pyannote/other"}, TAG)
    assert other.resolve_model_source() == (
        "pyannote/other",
        "huggingface",
        "HUGGINGFACE_ACCESS_TOKEN",
    )


def test_a_baked_directory_without_the_pipeline_fails_clearly(
    tmp_path, monkeypatch
):
    """A baked directory without the pipeline fails clearly"""
    monkeypatch.setenv(context_module.MODEL_DIR_ENV_VAR, str(tmp_path))
    with pytest.raises(RuntimeError, match="holds no config.yaml"):
        PyannoteDiarizationContext({}, TAG).resolve_model_source()


def test_model_source_is_the_offline_cache_under_hf_hub_offline(monkeypatch):
    """Model source is the offline cache under hf hub offline"""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert PyannoteDiarizationContext({}, TAG).resolve_model_source() == (
        context_module.DEFAULT_MODEL,
        "hf_cache_offline",
        None,
    )


def test_missing_pyannote_install_fails_with_the_install_hint(log, monkeypatch):
    """Missing pyannote install fails with the install hint"""
    real_import = builtins.__import__

    def no_pyannote(name, *args, **kwargs):
        if name.startswith("pyannote"):
            raise ImportError("No module named 'pyannote'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pyannote)
    monkeypatch.delitem(sys.modules, "pyannote.audio", raising=False)
    monkeypatch.delitem(sys.modules, "pyannote", raising=False)

    with pytest.raises(RuntimeError) as error:
        PyannoteDiarizationContext({}, TAG).create(log)
    message = str(error.value)
    assert "pyannote.audio is not installed" in message
    assert "uv sync --extra pyannote-diarization" in message
    assert "diarization image" in message


def test_missing_token_for_a_huggingface_model_fails_before_loading(
    log, fake_pyannote
):
    """Missing token for a huggingface model fails before loading"""
    pipeline_class, _ = fake_pyannote
    with pytest.raises(RuntimeError) as error:
        PyannoteDiarizationContext({}, TAG).create(log)
    message = str(error.value)
    assert "HUGGINGFACE_ACCESS_TOKEN" in message
    assert context_module.MODEL_TERMS_URL in message
    assert context_module.MODEL_DIR_ENV_VAR in message
    pipeline_class.from_pretrained.assert_not_called()


def test_a_pipeline_that_does_not_load_names_the_fix(
    log, fake_pyannote, monkeypatch
):
    """A pipeline that does not load names the fix"""
    pipeline_class, _ = fake_pyannote
    monkeypatch.setenv("HUGGINGFACE_ACCESS_TOKEN", "hf_test")
    pipeline_class.from_pretrained.side_effect = OSError("401 Client Error")

    with pytest.raises(RuntimeError) as error:
        PyannoteDiarizationContext({"nice": 0}, TAG).create(log)
    message = str(error.value)
    assert "accept the model terms" in message
    assert "401 Client Error" in message


def test_a_broken_baked_directory_names_what_is_missing(
    log, fake_pyannote, monkeypatch, tmp_path
):
    """A broken baked directory names what is missing"""
    pipeline_class, _ = fake_pyannote
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("pipeline: {}\n", encoding="utf-8")
    monkeypatch.setenv(context_module.MODEL_DIR_ENV_VAR, str(model_dir))
    pipeline_class.from_pretrained.side_effect = FileNotFoundError(
        "segmentation/pytorch_model.bin"
    )

    with pytest.raises(RuntimeError) as error:
        PyannoteDiarizationContext({"nice": 0}, TAG).create(log)
    message = str(error.value)
    assert "missing segmentation, embedding" in message
    assert "rebuild the diarization image" in message


def test_create_loads_the_baked_model_offline_without_a_token(
    log, fake_pyannote, monkeypatch, tmp_path
):
    """Create loads the baked model offline without a token"""
    pipeline_class, pipeline = fake_pyannote
    model_dir = _baked_dir(tmp_path)
    monkeypatch.setenv(context_module.MODEL_DIR_ENV_VAR, str(model_dir))

    context = PyannoteDiarizationContext({"nice": 0}, TAG)
    service = context.create(log)

    pipeline_class.from_pretrained.assert_called_once_with(
        str(model_dir), token=None
    )
    assert isinstance(service, PyannoteDiarizationService)
    assert service.device == "cpu"
    info = context.runtime_info(service)
    assert info["model_source"] == "local_dir"
    assert info["device"] == "cpu"
    assert info["device_fallback"] is False
    assert info["model_load_sec"] >= 0.0
    pipeline.to.assert_not_called()


def test_create_falls_back_to_cpu_when_cuda_is_requested_but_absent(
    log, fake_pyannote, monkeypatch, tmp_path
):
    """Create falls back to cpu when cuda is requested but absent"""
    _, pipeline = fake_pyannote
    model_dir = _baked_dir(tmp_path)
    monkeypatch.setenv(context_module.MODEL_DIR_ENV_VAR, str(model_dir))

    context = PyannoteDiarizationContext({"device": "cuda", "nice": 0}, TAG)
    service = context.create(log)

    assert service.device == "cpu"
    info = context.runtime_info(service)
    assert info["configured_device"] == "cuda"
    assert info["device_fallback"] is True
    assert info["device_fallback_reason"]
    assert any(
        "falling back to CPU" in call.args[0]
        for call in log.warning.call_args_list
    )
    pipeline.to.assert_not_called()


def test_create_falls_back_when_the_pipeline_cannot_move_to_cuda(
    log, fake_pyannote, monkeypatch, tmp_path
):
    """Create falls back when the pipeline cannot move to cuda"""
    _, pipeline = fake_pyannote
    model_dir = _baked_dir(tmp_path)
    monkeypatch.setenv(context_module.MODEL_DIR_ENV_VAR, str(model_dir))
    monkeypatch.setattr(backend_module, "cuda_available", lambda: True)
    monkeypatch.setattr(backend_module, "_probe_cuda", lambda device: None)
    fake_torch = types.ModuleType("torch")
    fake_torch.device = lambda name: name
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    moves = []

    def to(device):
        moves.append(device)
        if device == "cuda":
            raise RuntimeError("CUDA error: device-side assert")

    pipeline.to.side_effect = to

    context = PyannoteDiarizationContext({"device": "auto", "nice": 0}, TAG)
    service = context.create(log)

    assert moves == ["cuda", "cpu"]
    assert service.device == "cpu"
    info = context.runtime_info(service)
    assert info["device_fallback"] is True
    assert "device-side assert" in info["device_fallback_reason"]
