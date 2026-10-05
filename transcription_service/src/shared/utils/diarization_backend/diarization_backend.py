"""
Defines the interface a speaker-diarization backend presents to the
diarization job, independent of the model behind it, and the device
selection every backend shares.

Today the one backend is the pyannote pipeline
(`PyannoteDiarizationService`). A streaming-native model such as NVIDIA's
Streaming Sortformer would be a second `DiarizationBackend` behind a second
`DiarizationContextInterface`; the job, the reconciler and the provider's
start-up checks only see these two types.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, runtime_checkable

import numpy as np

from src.shared.logger import Logger
from src.shared.utils.speaker_reconciler import SpeakerSegment
from src.shared.utils.worker_pool import JobContextInterface

DevicePreference = Literal["auto", "cpu", "cuda"]
DEVICE_CHOICES: tuple[DevicePreference, ...] = ("auto", "cpu", "cuda")


@dataclass
class DiarizationPass:
    """
    Everything one diarization pass reports about a stretch of audio

    Properties:
        segments            - Speaker turns to attribute words against,
                                with timestamps relative to the audio passed
                                in (the exclusive turns unless the backend is
                                configured overlap-aware)
        embeddings          - Raw label -> speaker embedding, in the
                                embedding model's own scale: the reconciler
                                normalises for cosine scoring and a PLDA
                                shipped with a model needs the raw vector. A
                                label whose centroid could not be computed
                                is absent
        overlap_segments    - The overlap-aware turns, always, so a benchmark
                                can score both conventions from one pass
    """

    segments: list[SpeakerSegment]
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    overlap_segments: list[SpeakerSegment] = field(default_factory=list)


def usable_embedding(vector: np.ndarray) -> bool:
    """
    Whether an embedding carries information: a zero row is a padded
    missing centroid and a NaN row a stage that failed on it
    """
    norm = float(np.linalg.norm(vector))
    return bool(np.isfinite(norm) and norm > 0.0)


@runtime_checkable
class DiarizationBackend(Protocol):
    """
    What the diarization job needs from a loaded diarization model
    """

    @property
    def device(self) -> str:
        """
        The device the backend is running inference on ("cpu" or "cuda")
        """

    @property
    def num_threads(self) -> int | None:
        """
        Torch intra-op threads every `diarize` call runs with, or None to
        leave the process-wide setting alone
        """

    @property
    def overlap_aware(self) -> bool:
        """
        Whether `segments` of every pass carry overlapping speech turns
        """

    @property
    def track_clusterer(
        self,
    ) -> Callable[[np.ndarray, np.ndarray], np.ndarray] | None:
        """
        Optional session-level re-clustering over a session's track
        embeddings (`clusterer(embeddings (n, d), seconds (n,)) -> cluster
        index (n,)`), or None when the backend has none
        """

    def diarize(
        self,
        samples: np.ndarray,
        sample_rate: int,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> DiarizationPass:
        """
        Run speaker diarization over mono float32 audio samples
        """


class DiarizationContextInterface(JobContextInterface[Any]):
    """
    Marker base for worker-pool contexts that create a `DiarizationBackend`.
    The whisper-streaming provider checks at start-up that the context its
    `diarization_context_tag` resolves to is one of these, so a tag pointed at
    the wrong context fails before the first session instead of inside it.
    """

    @property
    @abstractmethod
    def device(self) -> str:
        """
        The device this context will run on once created: the configured
        value with "auto" resolved against the hardware visible to this
        process. The worker re-checks it when loading the model and may fall
        back to the CPU; the pool reports the device it ended up on
        """

    @property
    def configured_device(self) -> DevicePreference:
        """
        The device preference as configured ("auto", "cpu" or "cuda")
        """
        return "cpu"


@dataclass
class DeviceSelection:
    """
    Outcome of selecting the inference device for a backend

    Properties:
        configured  - What the config asked for
        device      - What the backend runs on
        fallback    - True when `device` differs from a usable reading of
                        `configured` because CUDA was unavailable or failed
        reason      - Why it fell back, for the log and for telemetry
    """

    configured: DevicePreference
    device: str
    fallback: bool = False
    reason: str | None = None

    def as_runtime_info(self) -> dict[str, Any]:
        """
        The selection as the flat dict the worker reports to the pool
        """
        return {
            "configured_device": self.configured,
            "device": self.device,
            "device_fallback": self.fallback,
            "device_fallback_reason": self.reason,
        }


def cuda_available() -> bool:
    """
    Whether torch reports a usable CUDA device in this process. False when
    torch is not installed at all, so a config check can run without it.
    """
    try:
        # pylint: disable=import-outside-toplevel
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def resolve_device_preference(configured: DevicePreference) -> str:
    """
    Resolves "auto" to the device this process would pick, without touching
    the GPU: what the main process reports before the worker has loaded
    anything

    Args:
        configured  - The configured preference

    Returns:
        "cpu" or "cuda"
    """
    if configured == "auto":
        return "cuda" if cuda_available() else "cpu"
    return configured


def select_device(
    configured: DevicePreference,
    log: Logger,
    probe: Callable[[str], None] | None = None,
) -> DeviceSelection:
    """
    Picks the inference device inside the worker that is about to load a
    model, and proves it works before anything is placed on it.

    `auto` takes CUDA when torch sees a device and a small allocation on it
    succeeds, otherwise the CPU. `cuda` is tried the same way; if it is
    unavailable or the probe fails, the backend falls back to the CPU with a
    warning rather than failing the worker: labels are optional and a
    session must never die because a GPU driver did. The fallback is
    reported to the pool (`DeviceSelection.fallback`) so telemetry shows a
    deployment that asked for CUDA and is not getting it.

    Args:
        configured  - The configured preference
        log         - Worker logger; the fallback is a warning here
        probe       - Optional override of the CUDA probe (tests); receives
                        the device string and raises when it does not work

    Returns:
        The selection
    """
    if configured == "cpu":
        return DeviceSelection(configured=configured, device="cpu")

    if not cuda_available():
        if configured == "auto":
            log.info("Diarization device auto: CUDA not available, using CPU")
            return DeviceSelection(configured=configured, device="cpu")
        reason = "torch.cuda.is_available() is false"
        log.warning(
            "Diarization device 'cuda' requested but CUDA is not available; "
            "falling back to CPU",
            context={"reason": reason},
        )
        return DeviceSelection(
            configured=configured, device="cpu", fallback=True, reason=reason
        )

    try:
        (probe or _probe_cuda)("cuda")
    except Exception as error:  # pylint: disable=broad-exception-caught
        reason = f"CUDA probe failed: {type(error).__name__}: {error}"
        log.warning(
            f"Diarization device '{configured}' resolved to CUDA but the "
            "device does not work; falling back to CPU",
            context={"reason": reason},
        )
        return DeviceSelection(
            configured=configured, device="cpu", fallback=True, reason=reason
        )

    log.info(f"Diarization device '{configured}' resolved to CUDA")
    return DeviceSelection(configured=configured, device="cuda")


def _probe_cuda(device: str) -> None:
    """
    A small allocation and a kernel on the device, synchronised, so a broken
    driver or an out-of-memory card fails here and not on the first pass
    """
    # pylint: disable=import-outside-toplevel
    import torch

    tensor = torch.ones(8, device=device)
    (tensor * 2).sum().item()
    torch.cuda.synchronize()
