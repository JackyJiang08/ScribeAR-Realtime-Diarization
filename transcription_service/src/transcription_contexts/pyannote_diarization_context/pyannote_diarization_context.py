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

    def __init__(self, pipeline: Any):
        self._pipeline = pipeline

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

        if self._config.device == "cuda":
            import torch

            pipeline.to(torch.device("cuda"))

        log.info("Pyannote diarization model loaded successfully")
        return PyannoteDiarizationService(pipeline)

    def destroy(
        self, log: Logger, context: PyannoteDiarizationModelType
    ) -> None:
        log.info("Destroying pyannote diarization context")
        if hasattr(context, "_pipeline"):
            del context._pipeline
