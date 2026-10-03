"""
Defines PyannoteDiarizationContext for caching a pyannote speaker
diarization pipeline in WorkerProcess
"""

# pylint: disable=import-outside-toplevel
# torch and pyannote are only imported when a diarization context is used,
# so deployments without the pyannote-diarization extra never load them

import os
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, TypeAdapter

from src.shared.logger import Logger
from src.shared.utils.speaker_reconciler import SpeakerSegment
from src.shared.utils.worker_pool import JobContextInterface


class PyannoteDiarizationService:
    """
    Service for speaker diarization backed by a pyannote Pipeline
    """

    def __init__(self, pipeline: Any, num_threads: int | None = None):
        self._pipeline = pipeline
        self._num_threads = num_threads

    @property
    def num_threads(self) -> int | None:
        """
        Torch intra-op threads every `diarize` call runs with, or None to
        leave the process-wide setting alone
        """
        return self._num_threads

    def diarize(
        self,
        samples: np.ndarray,
        sample_rate: int,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[SpeakerSegment]:
        """
        Run speaker diarization over mono float32 audio samples

        Args:
            samples         - Mono float32 audio samples
            sample_rate     - Sample rate of provided samples
            min_speakers    - Optional lower bound on speaker count
            max_speakers    - Optional upper bound on speaker count

        Returns:
            List of SpeakerSegments with timestamps relative to provided audio
        """
        import torch

        # Set on every call, not only at load: torch's thread count is
        # process-wide and the Silero context sets it to 1 in its own
        # `create()`, which lands on this process too whenever the two share
        # a worker. Cheap, and it makes the thread budget this context was
        # configured with the one that is actually in force.
        if self._num_threads is not None:
            torch.set_num_threads(self._num_threads)

        waveform = torch.from_numpy(
            np.ascontiguousarray(samples, dtype=np.float32)
        ).unsqueeze(0)

        kwargs: dict[str, int] = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        output = self._pipeline(
            {"waveform": waveform, "sample_rate": sample_rate}, **kwargs
        )

        # Newer pyannote pipelines wrap the annotation in an output object
        diarization = getattr(
            output,
            "exclusive_speaker_diarization",
            getattr(output, "speaker_diarization", output),
        )

        return [
            SpeakerSegment(
                start=float(turn.start),
                end=float(turn.end),
                speaker=str(speaker),
            )
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]


PyannoteDiarizationModelType = PyannoteDiarizationService


class PyannoteDiarizationContextConfig(BaseModel):
    """
    Configuration schema for PyannoteDiarizationContext
    """

    model: str = "pyannote/speaker-diarization-community-1"
    device: Literal["cuda"] | Literal["cpu"] = "cpu"
    token_env_var: str = "HUGGINGFACE_ACCESS_TOKEN"
    # Torch intra-op threads for inference on this context. Capped at 1 by
    # default so diarization cannot take the cores Whisper is using: on the
    # 4-CPU reference container the embedding stage barely speeds up with
    # more threads while the caption worker slows down measurably. Raise it
    # only on a machine with cores to spare, and never above the cores left
    # after Whisper's `cpu_threads`.
    num_threads: int | None = 1
    # `os.nice` increment applied to the worker process that creates this
    # context, so the OS scheduler prefers the caption worker whenever the
    # two compete for a core. 0 disables it. Applies to the whole worker, so
    # the context belongs on a worker of its own (see the provider's
    # start-up check); on a shared worker it would slow captions too.
    nice: int = 10
    # pyannote's segmentation sliding-window step as a ratio of its window
    # (the pipeline's own `segmentation_step`, default 0.1 = 1 s). The number
    # of embedding windows per pass is (window - 10 s) / step + 1, so this is
    # the second cost lever after the window length. None keeps the model's
    # default; the Phase 2a sweep in docs/speaker_diarization.md records
    # what each value costs and buys.
    segmentation_step: float | None = None


pyannote_diarization_context_config_adapter = TypeAdapter(
    PyannoteDiarizationContextConfig
)


class PyannoteDiarizationContext(
    JobContextInterface[PyannoteDiarizationModelType]
):
    """
    Job context definition for managing pyannote diarization pipeline lifecycle
    """

    def __init__(self, context_config: Any, tags: list[str]):
        super().__init__(tags)
        self._config = (
            pyannote_diarization_context_config_adapter.validate_python(
                context_config
            )
        )

    @property
    def device(self) -> str | None:
        # Reported on /metrics/status through the provider registry's
        # tag-to-device map, the same way the whisper context reports its own.
        return self._config.device

    def create(self, log: Logger) -> PyannoteDiarizationModelType:
        # pylint: disable=import-outside-toplevel
        # Only import pyannote when a diarization context is configured
        from pyannote.audio import Pipeline

        token = os.environ.get(self._config.token_env_var)
        if not token:
            raise RuntimeError(
                f"Environment variable '{self._config.token_env_var}' must be "
                "set to load pyannote diarization model"
            )

        log.info(
            f"Loading {self._config.model} diarization model on device: "
            f"{self._config.device}"
        )
        pipeline = Pipeline.from_pretrained(self._config.model, token=token)
        if pipeline is None:
            raise RuntimeError(
                f"Pipeline.from_pretrained returned nothing for "
                f"{self._config.model}; accept the model terms on HuggingFace "
                f"and check {self._config.token_env_var}"
            )

        if self._config.device == "cuda":
            import torch

            pipeline.to(torch.device("cuda"))

        if self._config.segmentation_step is not None:
            applied = self._apply_segmentation_step(pipeline)
            log.info(
                f"Pyannote segmentation step set to "
                f"{self._config.segmentation_step} of the segmentation "
                f"window ({applied:.2f}s)"
            )

        self._lower_scheduling_priority(log)

        log.info(
            "Pyannote diarization model loaded successfully "
            f"(torch threads per pass: {self._config.num_threads})"
        )
        return PyannoteDiarizationService(pipeline, self._config.num_threads)

    def _apply_segmentation_step(self, pipeline: Any) -> float:
        """
        Re-points the loaded pipeline's segmentation inference at the
        configured step. The step is a constructor argument of the pyannote
        pipeline that `from_pretrained` does not expose, but it is applied
        through `Inference.step`, which the pipeline reads on every call.

        Returns:
            The step in seconds
        """
        inference = pipeline._segmentation  # pylint: disable=protected-access
        step_sec = float(self._config.segmentation_step) * float(
            inference.duration
        )
        inference.step = step_sec
        pipeline.segmentation_step = self._config.segmentation_step
        return step_sec

    def _lower_scheduling_priority(self, log: Logger) -> None:
        """
        Lowers this worker process's OS scheduling priority by the configured
        nice increment, so the caption worker wins any contention for a core.
        Best effort: a platform without `os.nice` or a refused change is
        logged and ignored, since labels are optional and captions are not.
        """
        if self._config.nice <= 0:
            return
        try:
            resulting = os.nice(self._config.nice)
            log.info(
                "Lowered diarization worker scheduling priority "
                f"(nice {resulting})"
            )
        except (AttributeError, OSError) as error:
            log.warning(f"Could not lower diarization worker priority: {error}")

    def destroy(
        self, log: Logger, context: PyannoteDiarizationModelType
    ) -> None:
        log.info("Destroying pyannote diarization context")
        if hasattr(context, "_pipeline"):
            del context._pipeline
