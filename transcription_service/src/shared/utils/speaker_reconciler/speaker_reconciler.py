"""
Defines SpeakerReconciler for keeping speaker labels stable across
repeated diarization runs, with a per-session memory of speaker embeddings
"""

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

#: Session-level re-clustering: (raw embeddings (n, d), seconds per row (n,))
#: -> integer cluster per row. Provided by the diarization context (the
#: PLDA/VBx clustering that ships with the pyannote model); the reconciler
#: only knows cosine similarity on its own.
TrackClusterer = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class SpeakerSegment:
    """
    Speaker label active over a time range

    Properties:
        start   - Segment start in seconds relative to transcription session
        end     - Segment end in seconds relative to transcription session
        speaker - Speaker label for this time range
    """

    start: float
    end: float
    speaker: str


@dataclass
class SpeakerReconcilerConfig:
    """
    Thresholds of the reconciler. Every similarity is a cosine similarity
    between unit-normalised speaker embeddings (1 is the same vector) plus
    the overlap bonus described on `overlap_bonus`. The defaults were tuned
    on the AMI benchmark set (see docs/speaker_diarization.md, "Phase 2b")
    and are exposed through the provider config.

    Properties:
        label_prefix            - Prefix of session labels (`spk_0`, ...)
        match_threshold         - Score at or above which a raw cluster is
                                    attached to the best-matching known
                                    speaker
        new_speaker_threshold   - A raw cluster mints a new speaker only
                                    when its best score against every known
                                    speaker is below this ("clearly far")
                                    and it is long enough
        attach_threshold        - A cluster too short to mint attaches to
                                    the best-matching speaker when the score
                                    reaches this; below it the cluster stays
                                    unlabelled and is pooled as a candidate
        min_mint_duration_sec   - Speech a voice must have (in one run, or
                                    accumulated as a candidate over the
                                    runs that heard it) before it can mint
                                    a label
        min_mint_passes         - Runs an unknown voice must be heard in
                                    before it can mint; 1 lets a single long
                                    cluster mint at once, 2 requires the
                                    next window (which overlaps the
                                    previous one) to find the same voice
                                    again
        max_speakers            - Hard bound on labels per session; past it
                                    unknown voices attach to the nearest
                                    speaker or stay unlabelled
        max_candidates          - Bound on pooled unknown short voices
        merge_threshold         - Two known speakers whose centroids become
                                    at least this similar are merged (the
                                    junior label maps onto the senior one);
                                    1.0 disables merging
        centroid_memory_sec     - Cap on the weight of a centroid, so a
                                    speaker's centroid can still follow the
                                    voice after this much speech
        overlap_bonus           - Added to the score of (raw cluster, known
                                    speaker) in proportion to how much of
                                    the cluster's audio inside the previous
                                    run's window that speaker labelled; it
                                    carries identity across consecutive
                                    windows when embeddings are weak or
                                    absent
        min_update_sec          - Clusters shorter than this never move a
                                    centroid (backchannels are too noisy)
        sustained_split_sec     - A voice that keeps scoring between the
                                    new-speaker and the match threshold
                                    against its best speaker (near, but
                                    never a clear match) mints its own
                                    label once it has this many seconds of
                                    evidence, in one pass or accumulated
                                    over the passes that heard it. Lets a
                                    sustained second voice split off a
                                    speaker it resembles without lowering
                                    the threshold for everyone. 0 disables
        recluster_period_sec    - Every this many seconds of session time
                                    the session's track history is
                                    re-clustered with the model's own
                                    PLDA/VBx scoring (`clusterer`), and
                                    speakers whose tracks the clustering
                                    joins are merged while a speaker whose
                                    tracks fall into two clusters is split.
                                    Labels already sent never change; the
                                    new partition rules from the next pass
                                    on. 0 disables
        recluster_min_split_sec - A split needs at least this much speech
                                    in the part that leaves the speaker
        recluster_merge_fraction - Two speakers merge when at least this
                                    fraction of each one's speech lands in
                                    the same cluster
        max_tracks              - Bound on the track history kept for
                                    re-clustering (oldest dropped first)
    """

    # Tuned 2026-10-03 on the three AMI meetings (full 10 min, 10 s window,
    # 5 s period): benchmarks/diarization/tune_reconciler.py, grids in
    # benchmarks/diarization/configs/tune_grid*.json.
    label_prefix: str = "spk_"
    match_threshold: float = 0.4
    new_speaker_threshold: float = 0.3
    attach_threshold: float = 0.2
    min_mint_duration_sec: float = 2.5
    min_mint_passes: int = 1
    max_speakers: int = 32
    max_candidates: int = 8
    merge_threshold: float = 1.0
    centroid_memory_sec: float = 120.0
    overlap_bonus: float = 0.15
    min_update_sec: float = 0.5
    sustained_split_sec: float = 0.0
    recluster_period_sec: float = 0.0
    recluster_min_split_sec: float = 10.0
    recluster_merge_fraction: float = 0.7
    max_tracks: int = 600


@dataclass(eq=False)
class SpeakerMemory:
    """
    What the session remembers about one speaker. Lives in memory only.

    Properties:
        label       - Session label
        centroid    - Unit-normalised mean embedding, or None when the
                        speaker was minted without an embedding
        weight      - Seconds of speech behind the centroid, capped
        total_sec   - Seconds of speech attributed to the speaker so far
        last_seen   - Session time the speaker was last heard
    """

    label: str
    centroid: np.ndarray | None
    weight: float
    total_sec: float
    last_seen: float


@dataclass(eq=False)
class _Candidate:
    """
    An unknown voice not yet trusted enough to mint, pooled until it has
    been heard enough. `duration` is evidence seconds: every run that hears
    the voice adds its cluster, overlap between consecutive windows
    included, so a second look at the same voice counts as confirmation.
    """

    centroid: np.ndarray
    duration: float
    last_seen: float
    passes: int = 0


@dataclass(eq=False)
class _Track:
    """
    One labelled cluster of one run, kept for session-level re-clustering:
    the raw embedding (the model's scale, as the PLDA expects), its speech
    seconds, when it ended and the session label it was given
    """

    embedding: np.ndarray
    duration: float
    end: float
    label: str


@dataclass
class SpeakerReconcilerState:
    """
    Everything a reconciler needs to continue a session: the speaker
    memory, the candidate pools, the track history, the previous run and
    the label counter. Picklable, held in memory only, never written to
    disk.
    """

    speakers: list[SpeakerMemory] = field(default_factory=list)
    candidates: list[_Candidate] = field(default_factory=list)
    # Voices heard near, but never clearly matching, a known speaker
    # (`sustained_split_sec`)
    shadows: list[_Candidate] = field(default_factory=list)
    tracks: list[_Track] = field(default_factory=list)
    previous: list[SpeakerSegment] = field(default_factory=list)
    next_label_id: int = 0
    version: int = 0
    last_recluster_at: float = 0.0


@dataclass
class _RawCluster:
    """
    One raw label of a run with what the run tells about it
    """

    label: str
    segments: list[SpeakerSegment]
    duration: float
    embedding: np.ndarray | None
    first_start: float
    raw_embedding: np.ndarray | None = None

    @property
    def last_end(self) -> float:
        """
        Session time the cluster ends
        """
        return max(s.end for s in self.segments)


def _unit(vector) -> np.ndarray | None:
    """
    The vector normalised to unit length, or None when it is zero (pyannote
    pads missing centroids with zeros)
    """
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= 0.0:
        return None
    return array / norm


def _weighted_centroid(tracks: list["_Track"]) -> np.ndarray | None:
    """
    Unit-normalised, duration-weighted mean of the tracks' embeddings
    """
    if not tracks:
        return None
    total = sum(t.embedding.astype(np.float64) * t.duration for t in tracks)
    return _unit(total)


class SpeakerReconciler:
    """
    Maps per-run diarization labels to stable session-wide labels.

    Diarization pipelines assign arbitrary labels per run (SPEAKER_00, ...),
    so the same voice can receive a different label every time the newest
    window is re-diarized. The reconciler keeps a running centroid
    embedding per session speaker and matches every run's clusters against
    the whole session, not only against the previous window: a voice keeps
    its label when the window slides past it, after silence, and after a
    long gap. Time overlap with the previous run still counts, as a bonus,
    because consecutive windows share audio; runs without embeddings fall
    back to that overlap vote alone.

    A new label is minted only for a cluster that is clearly far from every
    known speaker and long enough to trust; short clusters (backchannels)
    attach to the best match or stay unlabelled and are pooled until that
    voice has spoken enough to deserve a label. Memory per session is
    bounded by `max_speakers` and `max_candidates` centroids.

    Two guards against under-counting (merging different people into one
    label) are available on top: `sustained_split_sec` lets a voice that
    keeps landing just below the match threshold mint its own label once it
    has spoken long enough, and `recluster_period_sec` re-clusters the
    session's track history periodically with the clustering the model
    ships (`clusterer`, PLDA/VBx) and merges or splits speakers to follow
    it. Neither changes a label already given to earlier audio: a split
    only rules from the next pass on, and a merge is reported through
    `last_merges` for the consumer to apply where it still can.

    After every run `last_mapping` holds raw label -> session label (or
    None for a cluster left unlabelled), `last_confidence` the score each
    session label was attached with, `last_merges` the (junior, senior)
    label pairs merged during the run so a consumer can relabel what it
    had, and `last_splits` the (parent, child) pairs split off.
    """

    def __init__(
        self,
        label_prefix: str = "spk_",
        config: SpeakerReconcilerConfig | None = None,
        state: SpeakerReconcilerState | None = None,
        clusterer: TrackClusterer | None = None,
    ):
        self._config = config or SpeakerReconcilerConfig()
        if config is None:
            self._config = replace(self._config, label_prefix=label_prefix)
        self._state = state if state is not None else SpeakerReconcilerState()
        self.clusterer = clusterer
        self.last_mapping: dict[str, str | None] = {}
        self.last_confidence: dict[str, float] = {}
        self.last_merges: list[tuple[str, str]] = []
        self.last_splits: list[tuple[str, str]] = []
        self.reclusterings = 0
        # Raw labels attached in the grey zone during the current run; they
        # must not pull the speaker's centroid towards a voice that may be
        # someone else's (see `sustained_split_sec`)
        self._hold_centroid: set[str] = set()

    @property
    def labels_minted(self) -> int:
        """
        Number of session-wide labels minted so far
        """
        return self._state.next_label_id

    @property
    def speakers(self) -> list[SpeakerMemory]:
        """
        The session's speaker memory, senior first
        """
        return list(self._state.speakers)

    @property
    def state_version(self) -> int:
        """
        Increments whenever the exported state changes
        """
        return self._state.version

    def export_state(self) -> SpeakerReconcilerState:
        """
        A snapshot of the session's speaker memory, for continuing the
        session in a rebuilt reconciler (reconnect). Centroids are copied
        """
        return SpeakerReconcilerState(
            speakers=[
                replace(
                    s,
                    centroid=(
                        None if s.centroid is None else s.centroid.copy()
                    ),
                )
                for s in self._state.speakers
            ],
            candidates=[
                replace(c, centroid=c.centroid.copy())
                for c in self._state.candidates
            ],
            shadows=[
                replace(c, centroid=c.centroid.copy())
                for c in self._state.shadows
            ],
            tracks=[
                replace(t, embedding=t.embedding.copy())
                for t in self._state.tracks
            ],
            previous=list(self._state.previous),
            next_label_id=self._state.next_label_id,
            version=self._state.version,
            last_recluster_at=self._state.last_recluster_at,
        )

    def reconcile(
        self,
        segments: list[SpeakerSegment],
        embeddings: dict[str, np.ndarray] | None = None,
    ) -> list[SpeakerSegment]:
        """
        Convert one diarization run's raw labels to session-wide labels

        Args:
            segments    - Segments from one diarization run, with timestamps
                            relative to the transcription session
            embeddings  - Optional raw label -> speaker embedding for the
                            run (pyannote's per-speaker centroids). Without
                            them identity is carried by time overlap only

        Returns:
            Segments with raw labels replaced by stable session-wide labels.
            Segments of a cluster left unlabelled (too short to mint and
            unlike every known speaker) are omitted
        """
        self.last_mapping = {}
        self.last_confidence = {}
        self.last_merges = []
        self.last_splits = []
        self._hold_centroid = set()
        # Keep the last non-empty run so continuity survives silent ticks
        if len(segments) == 0:
            return []

        clusters = self._clusters(segments, embeddings or {})
        overlap = self._overlap_fractions(clusters)
        mapping = self._assign_strong_matches(clusters, overlap)

        # Everything else: mint, attach, or pool, in order of appearance
        for cluster in sorted(clusters, key=lambda c: c.first_start):
            if cluster.label in mapping:
                continue
            best_label, best_score = self._best(cluster, overlap)
            decision = self._decide(cluster, best_label, best_score)
            if decision is None:
                self.last_mapping[cluster.label] = None
                continue
            label, confidence = decision
            mapping[cluster.label] = label
            self.last_confidence[label] = max(
                self.last_confidence.get(label, 0.0), confidence
            )

        for cluster in clusters:
            label = mapping.get(cluster.label)
            self.last_mapping[cluster.label] = label
            if label is not None:
                self._update_speaker(
                    label, cluster, cluster.label not in self._hold_centroid
                )
                self._remember_track(label, cluster)
        self._merge_similar_speakers(mapping)
        self._maybe_recluster(mapping, max(c.last_end for c in clusters))

        reconciled = [
            replace(segment, speaker=mapping[segment.speaker])
            for segment in segments
            if segment.speaker in mapping
        ]
        self._state.previous = reconciled
        self._state.version += 1
        return reconciled

    def _assign_strong_matches(
        self, clusters: list[_RawCluster], overlap: dict
    ) -> dict[str, str]:
        """
        Strong matches first, one-to-one, best score first, so a raw
        cluster cannot take a speaker another cluster resembles more
        """
        mapping: dict[str, str] = {}
        claimed: set[str] = set()
        scored = []
        for cluster in clusters:
            for speaker in self._state.speakers:
                score = self._score(cluster, speaker, overlap)
                if score is not None and score >= self._config.match_threshold:
                    scored.append((score, cluster.label, speaker.label))
        for score, raw, label in sorted(scored, key=lambda s: -s[0]):
            if raw in mapping or label in claimed:
                continue
            mapping[raw] = label
            claimed.add(label)
            self.last_confidence[label] = max(
                self.last_confidence.get(label, 0.0), score
            )
        return mapping

    # ------------------------------------------------------------------
    # Scoring

    def _clusters(
        self, segments: list[SpeakerSegment], embeddings: dict
    ) -> list[_RawCluster]:
        by_label: dict[str, list[SpeakerSegment]] = {}
        for segment in segments:
            by_label.setdefault(segment.speaker, []).append(segment)
        clusters = []
        for label, group in by_label.items():
            embedding = embeddings.get(label)
            unit = _unit(embedding) if embedding is not None else None
            clusters.append(
                _RawCluster(
                    label=label,
                    segments=group,
                    duration=sum(max(0.0, s.end - s.start) for s in group),
                    embedding=unit,
                    first_start=min(s.start for s in group),
                    raw_embedding=(
                        np.asarray(embedding, dtype=np.float32).reshape(-1)
                        if unit is not None
                        else None
                    ),
                )
            )
        return clusters

    def _overlap_fractions(
        self, clusters: list[_RawCluster]
    ) -> dict[tuple[str, str], float]:
        """
        For every (raw label, session label): the fraction of the raw
        cluster's audio inside the previous run's span that the session
        label covered in that run
        """
        previous = self._state.previous
        if not previous:
            return {}
        prev_start = min(s.start for s in previous)
        prev_end = max(s.end for s in previous)
        fractions: dict[tuple[str, str], float] = {}
        for cluster in clusters:
            inside = sum(
                max(0.0, min(s.end, prev_end) - max(s.start, prev_start))
                for s in cluster.segments
            )
            if inside <= 0.0:
                continue
            votes: dict[str, float] = {}
            for segment in cluster.segments:
                for old in previous:
                    overlap = min(segment.end, old.end) - max(
                        segment.start, old.start
                    )
                    if overlap > 0:
                        votes[old.speaker] = (
                            votes.get(old.speaker, 0.0) + overlap
                        )
            for label, seconds in votes.items():
                fractions[(cluster.label, label)] = min(1.0, seconds / inside)
        return fractions

    def _score(
        self,
        cluster: _RawCluster,
        speaker: SpeakerMemory,
        overlap: dict[tuple[str, str], float],
    ) -> float | None:
        """
        Cosine similarity plus the overlap bonus, or None when neither an
        embedding comparison nor an overlap vote is possible
        """
        fraction = overlap.get((cluster.label, speaker.label), 0.0)
        if cluster.embedding is not None and speaker.centroid is not None:
            similarity = float(np.dot(cluster.embedding, speaker.centroid))
            return similarity + self._config.overlap_bonus * fraction
        if fraction > 0.0:
            # No embedding on one side: the vote alone decides, scaled so
            # a cluster mostly covered by one speaker clears the match bar
            return fraction * max(self._config.match_threshold, 1e-6) / 0.5
        return None

    def _best(
        self, cluster: _RawCluster, overlap: dict
    ) -> tuple[str | None, float]:
        best_label = None
        best_score = -1.0
        for speaker in self._state.speakers:
            score = self._score(cluster, speaker, overlap)
            if score is not None and score > best_score:
                best_label, best_score = speaker.label, score
        return best_label, best_score

    def _decide(
        self, cluster: _RawCluster, best_label: str | None, best_score: float
    ) -> tuple[str, float] | None:
        """
        What to do with a cluster no strong match claimed: mint, attach to
        the best speaker, or leave unlabelled (and pool it)
        """
        cfg = self._config
        long_enough = cluster.duration >= cfg.min_mint_duration_sec
        can_mint = len(self._state.speakers) < cfg.max_speakers
        clearly_far = (
            best_label is None or best_score < cfg.new_speaker_threshold
        )
        mint_at_once = long_enough and cfg.min_mint_passes <= 1

        if mint_at_once and clearly_far and can_mint:
            return self._mint(cluster.embedding), cfg.match_threshold
        if cluster.embedding is not None and clearly_far:
            pooled = self._pool_candidate(cluster)
            if pooled is not None and can_mint:
                # The voice has now spoken enough across runs to deserve a
                # label; this run's audio is the first it labels
                self._state.candidates.remove(pooled)
                return self._mint(pooled.centroid), cfg.match_threshold
        if (
            cfg.sustained_split_sec > 0
            and can_mint
            and cluster.embedding is not None
            and best_label is not None
            and not clearly_far
        ):
            # Near a known speaker but never a clear match: pool the voice
            # on its own and let it split off once it has sustained that
            # for long enough (a second person who sounds like the first,
            # or a far-field voice the centroid drifted away from)
            shadow = self._pool_candidate(
                cluster, self._state.shadows, cfg.sustained_split_sec
            )
            if shadow is not None:
                self._state.shadows.remove(shadow)
                return self._mint(shadow.centroid), cfg.match_threshold
            # Attached below, but without moving the centroid: a centroid
            # that follows every near voice would soon match it clearly
            # and the split could never happen
            self._hold_centroid.add(cluster.label)
        if best_label is not None and (
            best_score >= cfg.attach_threshold or (long_enough and not can_mint)
        ):
            return best_label, best_score
        return None

    def _pool_candidate(
        self,
        cluster: _RawCluster,
        pool: list[_Candidate] | None = None,
        min_duration: float | None = None,
    ) -> _Candidate | None:
        """
        Adds the cluster to the candidate it resembles (or a new one) in
        `pool` (default: the unknown-voice pool) and returns the candidate
        when it has been heard in enough runs and has accumulated at least
        `min_duration` seconds (default: the minting minimum) of speech
        """
        cfg = self._config
        assert cluster.embedding is not None
        if pool is None:
            pool = self._state.candidates
        if min_duration is None:
            min_duration = cfg.min_mint_duration_sec
        best = None
        best_similarity = cfg.match_threshold
        for candidate in pool:
            similarity = float(np.dot(cluster.embedding, candidate.centroid))
            if similarity >= best_similarity:
                best, best_similarity = candidate, similarity
        if best is None:
            best = _Candidate(cluster.embedding.copy(), 0.0, 0.0)
            pool.append(best)
            if len(pool) > cfg.max_candidates:
                pool.sort(key=lambda c: c.last_seen)
                del pool[0]
                if best not in pool:
                    return None
        # Evidence seconds: every run that hears the voice counts its whole
        # cluster, so a voice the next (overlapping) window finds again
        # earns its label on that second look. Tuned against counting each
        # stretch of audio once, which minted later and cost 0.02 DER.
        heard = cluster.duration
        merged = best.centroid * best.duration + cluster.embedding * heard
        unit = _unit(merged)
        best.centroid = unit if unit is not None else best.centroid
        best.duration += heard
        best.passes += 1
        best.last_seen = max(
            best.last_seen, max(s.end for s in cluster.segments)
        )
        if best.duration >= min_duration and best.passes >= cfg.min_mint_passes:
            return best
        return None

    def _mint(self, embedding: np.ndarray | None) -> str:
        label = f"{self._config.label_prefix}{self._state.next_label_id}"
        self._state.next_label_id += 1
        self._state.speakers.append(
            SpeakerMemory(
                label=label,
                centroid=None if embedding is None else embedding.copy(),
                weight=0.0,
                total_sec=0.0,
                last_seen=0.0,
            )
        )
        return label

    # ------------------------------------------------------------------
    # Memory

    def _speaker(self, label: str) -> SpeakerMemory:
        for speaker in self._state.speakers:
            if speaker.label == label:
                return speaker
        raise KeyError(label)

    def _update_speaker(
        self, label: str, cluster: _RawCluster, move_centroid: bool = True
    ) -> None:
        cfg = self._config
        speaker = self._speaker(label)
        speaker.total_sec += cluster.duration
        speaker.last_seen = max(
            speaker.last_seen, max(s.end for s in cluster.segments)
        )
        if (
            not move_centroid
            or cluster.embedding is None
            or cluster.duration < cfg.min_update_sec
        ):
            return
        if speaker.centroid is None:
            speaker.centroid = cluster.embedding.copy()
            speaker.weight = min(cluster.duration, cfg.centroid_memory_sec)
            return
        merged = (
            speaker.centroid * speaker.weight
            + cluster.embedding * cluster.duration
        )
        unit = _unit(merged)
        if unit is not None:
            speaker.centroid = unit
        speaker.weight = min(
            speaker.weight + cluster.duration, cfg.centroid_memory_sec
        )

    def _remember_track(self, label: str, cluster: _RawCluster) -> None:
        """
        Keeps the cluster for session-level re-clustering, when enabled
        """
        cfg = self._config
        if (
            cfg.recluster_period_sec <= 0
            or cluster.raw_embedding is None
            or cluster.duration < cfg.min_update_sec
        ):
            return
        tracks = self._state.tracks
        tracks.append(
            _Track(
                cluster.raw_embedding.copy(),
                cluster.duration,
                cluster.last_end,
                label,
            )
        )
        if len(tracks) > cfg.max_tracks:
            del tracks[: len(tracks) - cfg.max_tracks]

    # ------------------------------------------------------------------
    # Session-level re-clustering

    def _maybe_recluster(  # pylint: disable=too-many-locals,too-many-branches
        self, mapping: dict[str, str], now: float
    ) -> None:
        """
        Every `recluster_period_sec` of session time: re-clusters the track
        history with the model's clustering and brings the speaker memory
        in line with it (merges and splits). The current run's mapping is
        updated for merges; a split only affects later runs.
        """
        cfg = self._config
        if cfg.recluster_period_sec <= 0 or self.clusterer is None:
            return
        if now - self._state.last_recluster_at < cfg.recluster_period_sec:
            return
        self._state.last_recluster_at = now
        known = {s.label for s in self._state.speakers}
        tracks = [t for t in self._state.tracks if t.label in known]
        if len(tracks) < 2:
            return
        embeddings = np.stack([t.embedding for t in tracks]).astype(np.float32)
        weights = np.asarray([t.duration for t in tracks], dtype=np.float32)
        partition = np.asarray(self.clusterer(embeddings, weights)).reshape(-1)
        if len(partition) != len(tracks):
            return
        self.reclusterings += 1

        # Seconds of every speaker in every cluster
        seconds: dict[str, dict[int, float]] = {}
        for track, cluster in zip(tracks, partition):
            per = seconds.setdefault(track.label, {})
            per[int(cluster)] = per.get(int(cluster), 0.0) + track.duration
        primary = {
            label: max(per.items(), key=lambda kv: kv[1])[0]
            for label, per in seconds.items()
        }

        # Merges: speakers whose speech mostly shares one cluster
        by_cluster: dict[int, list[str]] = {}
        for label, cluster in primary.items():
            per = seconds[label]
            if per[cluster] / sum(per.values()) >= cfg.recluster_merge_fraction:
                by_cluster.setdefault(cluster, []).append(label)
        for labels in by_cluster.values():
            if len(labels) < 2:
                continue
            ordered = [s for s in self._state.speakers if s.label in labels]
            senior = ordered[0]
            for junior in ordered[1:]:
                self._merge(junior, senior, mapping)
                for track in tracks:
                    if track.label == junior.label:
                        track.label = senior.label
                seconds[senior.label] = {
                    k: seconds[senior.label].get(k, 0.0)
                    + seconds[junior.label].get(k, 0.0)
                    for k in set(seconds[senior.label])
                    | set(seconds[junior.label])
                }
                del seconds[junior.label]
        primary = {
            label: max(per.items(), key=lambda kv: kv[1])[0]
            for label, per in seconds.items()
        }

        # Splits: a speaker with a second cluster of its own, large enough
        # and not the home of another speaker
        for label, per in seconds.items():
            if len(self._state.speakers) >= cfg.max_speakers:
                break
            homes = {c for lbl, c in primary.items() if lbl != label}
            for cluster, duration in sorted(per.items(), key=lambda kv: -kv[1]):
                if cluster == primary[label] or cluster in homes:
                    continue
                if duration < cfg.recluster_min_split_sec:
                    continue
                leaving = [
                    t
                    for t, c in zip(tracks, partition)
                    if t.label == label and int(c) == cluster
                ]
                staying = [
                    t
                    for t, c in zip(tracks, partition)
                    if t.label == label and int(c) != cluster
                ]
                child = self._split(label, leaving, staying)
                if child is None:
                    break
                for track in leaving:
                    track.label = child
                homes.add(cluster)

    def _split(
        self, label: str, leaving: list[_Track], staying: list[_Track]
    ) -> str | None:
        """
        Mints a speaker for the `leaving` tracks and rebuilds the parent's
        centroid from the `staying` ones. Returns the child label
        """
        child_centroid = _weighted_centroid(leaving)
        parent_centroid = _weighted_centroid(staying)
        if child_centroid is None or parent_centroid is None:
            return None
        parent = self._speaker(label)
        child = self._mint(child_centroid)
        child_memory = self._speaker(child)
        child_memory.weight = min(
            sum(t.duration for t in leaving), self._config.centroid_memory_sec
        )
        child_memory.total_sec = sum(t.duration for t in leaving)
        child_memory.last_seen = max(t.end for t in leaving)
        parent.centroid = parent_centroid
        parent.weight = min(
            sum(t.duration for t in staying), self._config.centroid_memory_sec
        )
        parent.total_sec = max(0.0, parent.total_sec - child_memory.total_sec)
        self.last_splits.append((label, child))
        return child

    def _merge_similar_speakers(self, mapping: dict[str, str]) -> None:
        """
        Merges speakers whose centroids have converged: the junior label is
        mapped onto the senior one for this run and from now on
        """
        cfg = self._config
        if cfg.merge_threshold >= 1.0:
            return
        changed = True
        while changed:
            changed = False
            speakers = self._state.speakers
            for i, senior in enumerate(speakers):
                if senior.centroid is None:
                    continue
                for junior in speakers[i + 1 :]:
                    if junior.centroid is None:
                        continue
                    similarity = float(np.dot(senior.centroid, junior.centroid))
                    if similarity < cfg.merge_threshold:
                        continue
                    self._merge(junior, senior, mapping)
                    changed = True
                    break
                if changed:
                    break

    def _merge(
        self, junior: SpeakerMemory, senior: SpeakerMemory, mapping: dict
    ) -> None:
        cfg = self._config
        assert junior.centroid is not None and senior.centroid is not None
        merged = senior.centroid * max(
            senior.weight, 1e-3
        ) + junior.centroid * max(junior.weight, 1e-3)
        unit = _unit(merged)
        if unit is not None:
            senior.centroid = unit
        senior.weight = min(
            senior.weight + junior.weight, cfg.centroid_memory_sec
        )
        senior.total_sec += junior.total_sec
        senior.last_seen = max(senior.last_seen, junior.last_seen)
        self._state.speakers.remove(junior)
        for raw, label in list(mapping.items()):
            if label == junior.label:
                mapping[raw] = senior.label
        self._state.previous = [
            replace(s, speaker=senior.label) if s.speaker == junior.label else s
            for s in self._state.previous
        ]
        if junior.label in self.last_confidence:
            self.last_confidence[senior.label] = max(
                self.last_confidence.get(senior.label, 0.0),
                self.last_confidence.pop(junior.label),
            )
        self.last_merges.append((junior.label, senior.label))
