"""
Defines PyannoteDiarizationContext for caching a pyannote speaker
diarization pipeline in WorkerProcess
"""

# pylint: disable=import-outside-toplevel
# torch and pyannote are only imported when a diarization context is used,
# so deployments without the pyannote-diarization extra never load them

import math
import os
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, TypeAdapter

from src.shared.logger import Logger
from src.shared.utils.diarization_backend import (
    DevicePreference,
    DeviceSelection,
    DiarizationContextInterface,
    DiarizationPass,
    select_device,
    usable_embedding,
)
from src.shared.utils.diarization_backend.diarization_backend import (
    resolve_device_preference,
)
from src.shared.utils.speaker_reconciler import SpeakerSegment

#: The model this context ships configured for, and the page whose terms a
#: deployer accepts before the token can download it
DEFAULT_MODEL = "pyannote/speaker-diarization-community-1"
MODEL_TERMS_URL = (
    "https://huggingface.co/pyannote/speaker-diarization-community-1"
)
#: Environment variable the diarization Docker images set to the directory
#: the model was baked into at build time; when the configured `model` is a
#: HuggingFace id and this directory holds a pipeline (`config.yaml`), the
#: context loads from it and needs neither network nor token at runtime
MODEL_DIR_ENV_VAR = "SCRIBEAR_DIARIZATION_MODEL_DIR"

# Re-exported for callers that imported it from here before it moved to the
# backend-neutral module
__all__ = ["DiarizationPass"]

_usable = usable_embedding


def _annotation_segments(annotation: Any) -> list[SpeakerSegment]:
    return [
        SpeakerSegment(
            start=float(turn.start), end=float(turn.end), speaker=str(speaker)
        )
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


class PyannoteDiarizationService:
    """
    Service for speaker diarization backed by a pyannote Pipeline
    """

    def __init__(
        self,
        pipeline: Any,
        num_threads: int | None = None,
        overlap_aware: bool = False,
        local_speakers: bool = True,
        shared_embeddings: bool = True,
        device: str = "cpu",
    ):
        self._pipeline = pipeline
        self._num_threads = num_threads
        self._overlap_aware = overlap_aware
        self._local_speakers = local_speakers
        self._shared_embeddings = shared_embeddings
        self._device = device
        # What the context reports to the pool once created (device
        # selection, model source, load time); filled by the context
        self.runtime_info: dict[str, Any] = {}
        # Per-stage seconds of the last `diarize` call when the shared pass
        # ran it (the pipeline's own hooks time the other path). Read by the
        # benchmark harness; empty after a pipeline pass
        self.last_stage_times: dict[str, float] = {}

    @property
    def device(self) -> str:
        """
        The device the pipeline runs inference on ("cpu" or "cuda")
        """
        return self._device

    @property
    def num_threads(self) -> int | None:
        """
        Torch intra-op threads every `diarize` call runs with, or None to
        leave the process-wide setting alone
        """
        return self._num_threads

    @property
    def overlap_aware(self) -> bool:
        """
        Whether `segments` of every pass carry overlapping speech turns
        """
        return self._overlap_aware

    def diarize(
        self,
        samples: np.ndarray,
        sample_rate: int,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> DiarizationPass:
        """
        Run speaker diarization over mono float32 audio samples

        Args:
            samples         - Mono float32 audio samples
            sample_rate     - Sample rate of provided samples
            min_speakers    - Optional lower bound on speaker count
            max_speakers    - Optional upper bound on speaker count

        Returns:
            The pass: segments with timestamps relative to the provided
            audio, one embedding per raw label, and the overlap-aware turns
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

        self.last_stage_times = {}
        if self._local_speakers and self._shared_embeddings:
            shared = self._shared_local_pass(waveform, sample_rate)
            if shared is not None:
                return shared

        kwargs: dict[str, int] = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        captured: dict[str, Any] = {}

        def capture(step_name, step_artefact=None, file=None, **progress):
            # Progress calls of a step carry `completed`; the final call
            # of each step carries the whole artefact and no progress.
            del file
            if step_artefact is not None and progress.get("completed") is None:
                captured[step_name] = step_artefact

        output = self._pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            hook=capture,
            **kwargs,
        )

        if self._local_speakers:
            local = self._local_pass(captured)
            if local is not None:
                return local
        return self._clustered_pass(output)

    def _clustered_pass(self, output: Any) -> DiarizationPass:
        """
        The pipeline's own clustered output, both conventions and the
        per-speaker centroids
        """
        # Newer pyannote pipelines wrap the annotation in an output object
        # carrying both conventions and the per-speaker centroids; an older
        # pipeline returns the annotation alone.
        overlap = getattr(output, "speaker_diarization", output)
        exclusive = getattr(output, "exclusive_speaker_diarization", overlap)
        overlap_segments = _annotation_segments(overlap)
        segments = (
            overlap_segments
            if self._overlap_aware
            else _annotation_segments(exclusive)
        )
        return DiarizationPass(
            segments=segments,
            embeddings=self._embeddings(output, overlap),
            overlap_segments=overlap_segments,
        )

    def _local_pass(self, captured: dict[str, Any]) -> DiarizationPass | None:
        """
        The segmentation model's own speaker tracks for a one-chunk window,
        each with its embedding, instead of the pipeline's clustered output.

        On a window no longer than the segmentation chunk (10 s) the
        pipeline has a single chunk to cluster, and its VBx step collapses
        the chunk's two or three local speakers into one label almost
        every time (measured: 0 of 360 passes with more than one speaker on
        the AMI set, while the segmentation itself separated two voices in
        a fifth of them). The local tracks are what the session-level
        reconciler needs: one turn sequence per voice in the window and a
        centroid per track to match against the session's speakers.
        Multi-chunk windows keep the pipeline's clustering, which is needed
        to join a voice across chunks.

        Returns:
            The pass, or None when the window had no speech, more than one
            chunk, or the pipeline did not expose the artefacts
        """
        segmentations = captured.get("segmentation")
        embeddings = captured.get("embeddings")
        count = captured.get("speaker_counting")
        if segmentations is None or embeddings is None or count is None:
            return None
        if segmentations.data.shape[0] != 1:
            return None
        overlap = self._local_annotation(segmentations, count, exclusive=False)
        exclusive = self._local_annotation(segmentations, count, exclusive=True)
        mapping = {
            label: f"LOCAL_{label}"
            for label in set(overlap.labels()) | set(exclusive.labels())
        }
        overlap_segments = _annotation_segments(
            overlap.rename_labels(mapping=mapping)
        )
        return DiarizationPass(
            segments=(
                overlap_segments
                if self._overlap_aware
                else _annotation_segments(
                    exclusive.rename_labels(mapping=mapping)
                )
            ),
            embeddings=self._local_embeddings(embeddings, mapping),
            overlap_segments=overlap_segments,
        )

    def _shared_local_pass(  # pylint: disable=too-many-locals
        self, waveform: Any, sample_rate: int
    ) -> DiarizationPass | None:
        """
        The local pass of a one-chunk window with the embedding network run
        once: segmentation, speaker counting, then one forward pass of the
        embedding model's frame stage over the window and its pooling stage
        once per active speaker track.

        pyannote's own pipeline runs the whole embedding network once per
        (chunk, speaker slot): three times per 10 s window for community-1,
        inactive slots included, on the same waveform each time, because the
        speaker mask only enters the final statistics-pooling layer. Running
        the frame stage once and pooling per active track is the same
        computation (measured identical to 1.4e-6 relative on AMI audio) at
        about a third of the cost. The clustering stage is skipped because
        the local pass never used its output.

        Returns:
            The pass, or None when the window is longer than one
            segmentation chunk or the pipeline does not expose the stages,
            so the caller runs the full pipeline instead
        """
        pipeline = self._pipeline
        inference = getattr(pipeline, "_segmentation", None)
        embedding = getattr(pipeline, "_embedding", None)
        model = getattr(embedding, "model_", None)
        if (
            inference is None
            or model is None
            or not hasattr(model, "forward_frames")
            or not hasattr(model, "forward_embedding")
        ):
            return None
        chunk_samples = int(round(float(inference.duration) * sample_rate))
        if waveform.shape[-1] > chunk_samples:
            return None

        file = {"waveform": waveform, "sample_rate": sample_rate, "uri": "w"}
        started = time.perf_counter()
        segmentations = pipeline.get_segmentations(file)
        if segmentations.data.shape[0] != 1:
            return None
        if inference.model.specifications.powerset:
            binarized = segmentations
        else:
            from pyannote.audio.utils.signal import binarize

            binarized = binarize(
                segmentations,
                onset=pipeline.segmentation.threshold,
                initial_state=False,
            )
        count = pipeline.speaker_count(
            binarized, inference.model.receptive_field, warm_up=(0.0, 0.0)
        )
        segmentation_sec = time.perf_counter() - started

        started = time.perf_counter()
        raw = self._shared_embeddings_for(pipeline, embedding, file, binarized)
        embedding_sec = time.perf_counter() - started

        started = time.perf_counter()
        if np.nanmax(count.data) == 0.0:
            result = DiarizationPass(segments=[], embeddings={})
        else:
            overlap = self._local_annotation(
                segmentations, count, exclusive=False
            )
            exclusive = self._local_annotation(
                segmentations, count, exclusive=True
            )
            mapping = {
                label: f"LOCAL_{label}"
                for label in set(overlap.labels()) | set(exclusive.labels())
            }
            overlap_segments = _annotation_segments(
                overlap.rename_labels(mapping=mapping)
            )
            embeddings = {}
            for label, name in mapping.items():
                vector = raw.get(int(label))
                if vector is not None and _usable(vector):
                    embeddings[name] = vector.reshape(-1).copy()
            result = DiarizationPass(
                segments=(
                    overlap_segments
                    if self._overlap_aware
                    else _annotation_segments(
                        exclusive.rename_labels(mapping=mapping)
                    )
                ),
                embeddings=embeddings,
                overlap_segments=overlap_segments,
            )
        other_sec = time.perf_counter() - started
        self.last_stage_times = {
            "segmentation": segmentation_sec,
            "embeddings": embedding_sec,
            "clustering_and_other": other_sec,
            "total": segmentation_sec + embedding_sec + other_sec,
        }
        return result

    @staticmethod
    def _shared_embeddings_for(  # pylint: disable=too-many-locals
        pipeline: Any, embedding: Any, file: dict, binarized: Any
    ) -> dict[int, np.ndarray]:
        """
        One embedding per active speaker slot of the single chunk, with the
        pipeline's own mask rule (`embedding_exclude_overlap`: the slot's
        non-overlapping frames when there are enough of them, else all its
        frames) and one forward pass of the network's frame stage

        Returns:
            Slot index -> embedding, in the model's own scale
        """
        import torch

        duration = float(binarized.sliding_window.duration)
        _, num_frames, num_speakers = binarized.data.shape
        data = np.nan_to_num(binarized.data[0], nan=0.0).astype(np.float32)
        active = [s for s in range(num_speakers) if float(data[:, s].sum()) > 0]
        if not active:
            return {}
        if getattr(pipeline, "embedding_exclude_overlap", False):
            min_num_frames = math.ceil(
                num_frames
                * embedding.min_num_samples
                / (duration * embedding.sample_rate)
            )
            clean = data * (data.sum(axis=1, keepdims=True) < 2)
        else:
            min_num_frames = -1
            clean = data
        masks = np.stack(
            [
                (
                    clean[:, slot]
                    if float(clean[:, slot].sum()) > min_num_frames
                    else data[:, slot]
                )
                for slot in active
            ]
        )
        chunk = next(iter(binarized))[0]
        chunk_waveform, _ = (
            pipeline._audio.crop(  # pylint: disable=protected-access
                file, chunk, mode="pad"
            )
        )
        device = embedding.device
        model = embedding.model_
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frames = model.forward_frames(chunk_waveform[None].to(device))
            vectors = model.forward_embedding(
                frames, weights=torch.from_numpy(masks)[None].to(device)
            )
        vectors = vectors[0].cpu().numpy()
        return {slot: vectors[index] for index, slot in enumerate(active)}

    def _local_annotation(
        self, segmentations: Any, count: Any, exclusive: bool
    ):
        """
        Turns of the local speaker tracks, through the pipeline's own
        frame-to-time reconstruction; `exclusive` caps the speaker count
        at one per frame
        """
        from pyannote.core import SlidingWindowFeature

        pipeline = self._pipeline
        if exclusive:
            count = SlidingWindowFeature(
                np.minimum(count.data, 1), count.sliding_window
            )
        return pipeline.to_annotation(
            pipeline.to_diarization(segmentations, count),
            min_duration_on=0.0,
            min_duration_off=pipeline.segmentation.min_duration_off,
        )

    @staticmethod
    def _local_embeddings(
        embeddings: Any, mapping: dict
    ) -> dict[str, np.ndarray]:
        vectors = np.asarray(embeddings, dtype=np.float32)[0]
        raw: dict[str, np.ndarray] = {}
        for label, name in mapping.items():
            index = int(label)
            if index >= len(vectors):
                continue
            vector = vectors[index].reshape(-1)
            if _usable(vector):
                raw[name] = vector.copy()
        return raw

    @staticmethod
    def _embeddings(output: Any, diarization: Any) -> dict[str, np.ndarray]:
        """
        Pyannote's per-speaker centroids, keyed by raw label, in the
        embedding model's own scale. The array is sorted in `labels()` order
        and padded with zero rows for speakers without a centroid, which
        are skipped
        """
        centroids = getattr(output, "speaker_embeddings", None)
        if centroids is None:
            return {}
        centroids = np.asarray(centroids, dtype=np.float32)
        embeddings: dict[str, np.ndarray] = {}
        for index, label in enumerate(diarization.labels()):
            if index >= len(centroids):
                break
            vector = centroids[index].reshape(-1)
            if _usable(vector):
                embeddings[str(label)] = vector.copy()
        return embeddings

    @property
    def track_clusterer(self):
        """
        The session-level re-clustering the reconciler can call: the
        PLDA/VBx clustering that ships with the pipeline, applied to a
        session's track embeddings. None when the pipeline has no PLDA
        """
        clustering = getattr(self._pipeline, "clustering", None)
        plda = getattr(clustering, "plda", None)
        if plda is None:
            return None
        return build_track_clusterer(
            plda,
            float(getattr(clustering, "threshold", 0.6)),
            float(getattr(clustering, "Fa", 0.07)),
            float(getattr(clustering, "Fb", 0.8)),
        )


def build_track_clusterer(plda: Any, threshold: float, fa: float, fb: float):
    """
    The clustering step of the community-1 pipeline (agglomerative
    clustering on unit-normalised embeddings at `threshold`, refined by
    VBx in the PLDA space with `fa`, `fb`) as a function over a session's
    track embeddings. Each track is weighted by its speech seconds by
    repeating it (one copy per second, at most 20), so VBx's statistics
    follow speech time rather than the number of windows.

    Returns:
        clusterer(embeddings (n, d), seconds (n,)) -> cluster index (n,)
    """

    def clusterer(  # pylint: disable=too-many-locals
        embeddings: np.ndarray, seconds: np.ndarray
    ) -> np.ndarray:
        from pyannote.audio.utils.vbx import cluster_vbx
        from scipy.cluster.hierarchy import fcluster, linkage

        embeddings = np.asarray(embeddings, dtype=np.float64)
        count = len(embeddings)
        if count == 0:
            return np.zeros(0, dtype=int)
        if count == 1:
            return np.zeros(1, dtype=int)
        repeats = np.clip(np.rint(np.asarray(seconds)), 1, 20).astype(int)
        expanded = np.repeat(embeddings, repeats, axis=0)
        owner = np.repeat(np.arange(count), repeats)
        normed = expanded / np.maximum(
            np.linalg.norm(expanded, axis=1, keepdims=True), 1e-12
        )
        dendrogram = linkage(normed, method="centroid", metric="euclidean")
        ahc = fcluster(dendrogram, threshold, criterion="distance") - 1
        _, ahc = np.unique(ahc, return_inverse=True)
        if ahc.max() == 0:
            return np.zeros(count, dtype=int)
        features = plda(expanded)
        q, prior = cluster_vbx(
            ahc, features, plda.phi, Fa=fa, Fb=fb, maxIters=20
        )
        kept = q[:, prior > 1e-7]
        if kept.shape[1] == 0:
            return np.zeros(count, dtype=int)
        per_row = kept.argmax(axis=1)
        # Back to one cluster per track: the cluster most of its copies took
        out = np.zeros(count, dtype=int)
        for index in range(count):
            votes = np.bincount(per_row[owner == index])
            out[index] = int(votes.argmax())
        _, out = np.unique(out, return_inverse=True)
        return out.astype(int)

    return clusterer


PyannoteDiarizationModelType = PyannoteDiarizationService


class PyannoteDiarizationContextConfig(BaseModel):
    """
    Configuration schema for PyannoteDiarizationContext
    """

    # A HuggingFace model id (downloaded once with the token, then served
    # from the HuggingFace cache), or a local directory holding a pipeline
    # `config.yaml` with its model folders (no token, no network). The
    # diarization Docker images bake the default model into a directory and
    # name it in SCRIBEAR_DIARIZATION_MODEL_DIR, which takes over for the
    # default id without a config change (see `PyannoteDiarizationContext`).
    model: str = DEFAULT_MODEL
    # Inference device. "cpu" (the default; production is CPU-only), "cuda",
    # or "auto": CUDA when torch sees a working device, otherwise CPU. A CUDA
    # device that is missing or fails its probe falls back to the CPU with a
    # warning in the log and on /metrics/status (`deviceFallbacks`); it never
    # fails the worker or a session.
    device: DevicePreference = "cpu"
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
    # Report pyannote's overlap-aware turns instead of its exclusive ones
    # (one speaker per instant). Off by default: words are attributed to
    # the speaker overlapping them the most either way, and the Phase 2b
    # measurement in docs/speaker_diarization.md records what each
    # convention costs.
    overlap_aware: bool = False
    # On windows of one segmentation chunk (10 s or less) report the
    # segmentation model's own speaker tracks with their embeddings instead
    # of the pipeline's clustered output, which collapses such a window to
    # one speaker (see PyannoteDiarizationService._local_pass). The
    # session-level reconciler does the clustering. Longer windows always
    # use the pipeline's clustering.
    local_speakers: bool = True
    # Override of the pipeline's clustering threshold (VBx agglomerative
    # distance, 0.6 in the shipped community-1 config; lower splits more
    # readily inside a window). None keeps the model's value.
    clustering_threshold: float | None = None
    # With `local_speakers`, run the embedding network's frame stage once
    # per window and pool it once per active speaker track, instead of the
    # pipeline's one full pass per speaker slot (three per window, inactive
    # slots included). Same embeddings (the mask only enters the pooling
    # layer), about a third of the pass cost; the Phase 2 wrap-up in
    # docs/speaker_diarization.md records the measurement. Off runs the
    # pipeline unchanged.
    shared_embeddings: bool = True


pyannote_diarization_context_config_adapter = TypeAdapter(
    PyannoteDiarizationContextConfig
)


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class PyannoteDiarizationContext(DiarizationContextInterface):
    """
    Job context definition for managing pyannote diarization pipeline lifecycle

    Start-up is fail-fast with a message that names the fix: a missing
    `pyannote.audio` install, a model that is neither baked in nor cached nor
    downloadable (no token, terms not accepted, no network), or a pipeline
    that does not load, each fail the worker's initialization and so the
    service's start, instead of the first session. The one thing that never
    fails start-up is the device: CUDA that is missing or broken falls back
    to the CPU with a warning (`select_device`).
    """

    def __init__(self, context_config: Any, tags: list[str]):
        super().__init__(tags)
        self._config = (
            pyannote_diarization_context_config_adapter.validate_python(
                context_config
            )
        )
        # Resolved in the main process for reporting (no GPU is touched
        # here); the worker makes the real selection in `create` and reports
        # what it ended up on through `runtime_info`.
        self._resolved_device = resolve_device_preference(self._config.device)

    @property
    def device(self) -> str:
        # Reported on /metrics/status through the provider registry's
        # tag-to-device map, the same way the whisper context reports its own;
        # the registry prefers the worker's reported device once loaded.
        return self._resolved_device

    @property
    def configured_device(self) -> DevicePreference:
        return self._config.device

    @property
    def model_terms_url(self) -> str:
        """
        Where a deployer accepts the model's terms (the gated form)
        """
        return MODEL_TERMS_URL

    def resolve_model_source(self) -> tuple[str, str, str | None]:
        """
        Decides where the model is loaded from, without loading it

        Returns:
            (checkpoint to pass to pyannote, source kind, token env var that
            is needed or None): source kind is "local_dir" for a directory
            holding `config.yaml` (the configured path, or the baked model
            directory named by SCRIBEAR_DIARIZATION_MODEL_DIR when the model
            is the default id), "hf_cache_offline" when HF_HUB_OFFLINE is set
            (the HuggingFace cache must already hold the model; no token
            needed), else "huggingface" (download or cache check over the
            network, token required)
        """
        model = self._config.model
        if os.path.isdir(model):
            return model, "local_dir", None
        baked = os.environ.get(MODEL_DIR_ENV_VAR, "").strip()
        if baked and model == DEFAULT_MODEL:
            if os.path.isfile(os.path.join(baked, "config.yaml")):
                return baked, "local_dir", None
            raise RuntimeError(
                f"{MODEL_DIR_ENV_VAR}={baked!r} is set but holds no "
                "config.yaml: the diarization image's model bake failed or "
                "the directory was replaced; rebuild the image or unset the "
                "variable to load from HuggingFace"
            )
        if _truthy_env("HF_HUB_OFFLINE"):
            return model, "hf_cache_offline", None
        return model, "huggingface", self._config.token_env_var

    def create(self, log: Logger) -> PyannoteDiarizationModelType:
        # pylint: disable=import-outside-toplevel,too-many-locals
        started = time.perf_counter()
        try:
            # Only import pyannote when a diarization context is configured
            from pyannote.audio import Pipeline
        except ImportError as error:
            raise RuntimeError(
                "pyannote.audio is not installed but a pyannote-diarization "
                "context is configured: install the optional dependency "
                "group (uv sync --extra pyannote-diarization), use the "
                "transcription-service-*-diarization image, or remove the "
                f"context and turn diarization_detector off ({error})"
            ) from error

        checkpoint, source, token_var = self.resolve_model_source()
        token = None
        if token_var is not None:
            token = os.environ.get(token_var)
            if not token:
                raise RuntimeError(
                    f"Environment variable '{token_var}' must be set to load "
                    f"{self._config.model} from HuggingFace (accept the model "
                    f"terms at {MODEL_TERMS_URL} first), or bake the model "
                    f"into the image and point {MODEL_DIR_ENV_VAR} at it, or "
                    "set HF_HUB_OFFLINE=1 with the model already in the "
                    "HuggingFace cache"
                )

        selection = select_device(self._config.device, log)
        log.info(
            f"Loading {self._config.model} diarization model from "
            f"{source} ({checkpoint}) on device: {selection.device}"
        )
        try:
            pipeline = Pipeline.from_pretrained(checkpoint, token=token)
        except Exception as error:  # pylint: disable=broad-exception-caught
            raise RuntimeError(
                self._describe_load_failure(source, checkpoint, error)
            ) from error
        if pipeline is None:
            raise RuntimeError(
                f"Pipeline.from_pretrained returned nothing for "
                f"{checkpoint}; accept the model terms at {MODEL_TERMS_URL} "
                f"and check {self._config.token_env_var}"
            )

        selection = self._place_pipeline(pipeline, selection, log)

        if self._config.segmentation_step is not None:
            applied = self._apply_segmentation_step(pipeline)
            log.info(
                f"Pyannote segmentation step set to "
                f"{self._config.segmentation_step} of the segmentation "
                f"window ({applied:.2f}s)"
            )
        if self._config.clustering_threshold is not None:
            self._apply_clustering_threshold(pipeline)
            log.info(
                "Pyannote clustering threshold set to "
                f"{self._config.clustering_threshold}"
            )

        self._lower_scheduling_priority(log)

        load_sec = time.perf_counter() - started
        log.info(
            "Pyannote diarization model loaded successfully in "
            f"{load_sec:.1f}s on {selection.device} "
            f"(torch threads per pass: {self._config.num_threads})"
        )
        service = PyannoteDiarizationService(
            pipeline,
            self._config.num_threads,
            self._config.overlap_aware,
            self._config.local_speakers,
            self._config.shared_embeddings,
            device=selection.device,
        )
        service.runtime_info = selection.as_runtime_info() | {
            "model": self._config.model,
            "model_source": source,
            "model_checkpoint": checkpoint,
            "model_load_sec": round(load_sec, 2),
        }
        return service

    def runtime_info(self, context: PyannoteDiarizationModelType) -> dict:
        return dict(getattr(context, "runtime_info", {}) or {})

    @staticmethod
    def _place_pipeline(
        pipeline: Any, selection: DeviceSelection, log: Logger
    ) -> DeviceSelection:
        """
        Moves the pipeline to the selected device. A failure moving to CUDA
        (driver, memory) is the second place a GPU can break after the
        probe; it falls back to the CPU like the probe does
        """
        if selection.device != "cuda":
            return selection
        # pylint: disable=import-outside-toplevel
        import torch

        try:
            pipeline.to(torch.device("cuda"))
            return selection
        except Exception as error:  # pylint: disable=broad-exception-caught
            reason = (
                "moving the pipeline to CUDA failed: "
                f"{type(error).__name__}: {error}"
            )
            log.warning(
                "Diarization pipeline could not be placed on CUDA; falling "
                "back to CPU",
                context={"reason": reason},
            )
            pipeline.to(torch.device("cpu"))
            return DeviceSelection(
                configured=selection.configured,
                device="cpu",
                fallback=True,
                reason=reason,
            )

    def _describe_load_failure(
        self, source: str, checkpoint: str, error: Exception
    ) -> str:
        detail = f"{type(error).__name__}: {error}"
        if source == "local_dir":
            missing = [
                name
                for name in ("config.yaml", "segmentation", "embedding")
                if not (Path(checkpoint) / name).exists()
            ]
            hint = (
                f"missing {', '.join(missing)} under {checkpoint}; "
                if missing
                else ""
            )
            return (
                f"Could not load the diarization pipeline from the local "
                f"directory {checkpoint}: {hint}rebuild the diarization image "
                f"(the model is downloaded at build time) or point `model` / "
                f"{MODEL_DIR_ENV_VAR} at a complete pipeline directory "
                f"({detail})"
            )
        if source == "hf_cache_offline":
            return (
                f"Could not load {checkpoint} from the HuggingFace cache with "
                "HF_HUB_OFFLINE set: the model is not cached on this host. "
                "Download it once with a token, bake it into the image, or "
                f"unset HF_HUB_OFFLINE ({detail})"
            )
        return (
            f"Could not load {checkpoint} from HuggingFace: accept the model "
            f"terms at {MODEL_TERMS_URL} with the account that owns the "
            f"token in {self._config.token_env_var}, check the token and "
            "that the host can reach huggingface.co, or bake the model into "
            f"the image ({detail})"
        )

    def _apply_clustering_threshold(self, pipeline: Any) -> None:
        """
        Re-points the loaded pipeline's clustering at the configured
        threshold. The pipeline instantiates its hyper-parameters as plain
        attributes of the clustering object, which it reads on every call.
        """
        clustering = pipeline.clustering
        if not hasattr(clustering, "threshold"):
            raise RuntimeError(
                "clustering_threshold is set but this pyannote pipeline's "
                f"clustering ({type(clustering).__name__}) has no threshold"
            )
        clustering.threshold = float(self._config.clustering_threshold)

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
