from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from live_camera_pipeline.human_detector import RawHumanDetection

from tracking.cost_functions import match_score
from tracking.track_state import TrackState
from tracking.tracked_human import TrackedHuman
from tracking.tracker import SingleCameraTracker


# Single-Camera Tracking Framework milestone, Phase 6 -- named
# defaults, never inline magic numbers. A caller overrides any of these
# per SimpleSingleCameraTracker instance.
DEFAULT_MAX_MISSING_FRAMES = 5
DEFAULT_MINIMUM_CONFIDENCE = 0.0
DEFAULT_IOU_THRESHOLD = 0.3
DEFAULT_MAX_CENTROID_DISTANCE = 50.0


@dataclass
class _Track:

    # Internal, mutable per-track bookkeeping -- never returned to a
    # caller directly (TrackedHuman, an immutable snapshot, is built
    # from this each update() call). Lives only inside
    # SimpleSingleCameraTracker's own per-camera store.

    track_id: str
    camera_id: str
    bounding_box: Optional[Tuple[float, float, float, float]]
    confidence: float
    age: int
    frames_seen: int
    frames_missing: int
    last_timestamp: float


class SimpleSingleCameraTracker(SingleCameraTracker):

    # A clean, deterministic engineering baseline -- IoU matching,
    # falling back to centroid distance (tracking/cost_functions.py's
    # own "IoU, or centroid distance, or both" strategy), greedy
    # one-to-one assignment, no learned model, no motion prediction.
    # Deliberately NOT DeepSORT/ByteTrack/StrongSORT/OCSORT (Phase 5's
    # explicit exclusion list) -- a future tracker can replace this
    # class entirely; nothing importing SingleCameraTracker (Phase 3's
    # seam) needs to change when that happens.
    #
    # One instance may be called with many different camera_id values
    # -- track state is namespaced per camera_id internally (self.
    # _tracks: camera_id -> track_id -> _Track, self._next_id:
    # camera_id -> int), so detections from one camera can never match
    # against another camera's tracks (Phase 12's "camera isolation"
    # requirement).
    #
    # Output contract (stronger than the base SingleCameraTracker ABC
    # requires, but always true for THIS implementation): update()
    # always returns exactly len(detections) + (however many prior
    # tracks were not matched this cycle) entries -- the first
    # len(detections) correspond positionally, one-to-one, to
    # `detections` itself, regardless of confidence. This is what
    # keeps live_camera_pipeline.pipeline.LiveCameraPipeline's own
    # tracker integration a plain zip(), no confidence-filtering
    # duplicated on the caller's side.
    #
    # minimum_confidence does not drop a detection's output entry --
    # it only decides whether that detection is allowed to touch
    # persistent track state at all. A detection below
    # minimum_confidence can never match an existing track (a weak,
    # noisy detection must not silently refresh a real track's
    # continuity) and never creates a persistent track either -- it is
    # reported as a one-off, disposable NEW track (never stored, never
    # matchable again next cycle), so a low-confidence blip cannot
    # fabricate stable identity or steal frames_missing headroom from
    # a real track.

    def __init__(
        self,
        max_missing_frames: int = DEFAULT_MAX_MISSING_FRAMES,
        minimum_confidence: float = DEFAULT_MINIMUM_CONFIDENCE,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        max_centroid_distance: float = DEFAULT_MAX_CENTROID_DISTANCE,
    ):

        self.max_missing_frames = max_missing_frames
        self.minimum_confidence = minimum_confidence
        self.iou_threshold = iou_threshold
        self.max_centroid_distance = max_centroid_distance

        self._tracks: Dict[str, Dict[str, _Track]] = {}
        self._next_id: Dict[str, int] = {}

    # =====================================================

    def meets_minimum_confidence(self, detection: RawHumanDetection) -> bool:

        return detection.confidence >= self.minimum_confidence

    # =====================================================

    def update(
        self,
        camera_id: str,
        timestamp: float,
        detections: Sequence[RawHumanDetection],
    ) -> Tuple[TrackedHuman, ...]:

        tracks = self._tracks.setdefault(camera_id, {})

        qualifying_indices = [
            index for index, detection in enumerate(detections)
            if self.meets_minimum_confidence(detection)
        ]
        qualifying = [detections[index] for index in qualifying_indices]

        assignment = self._assign(tracks, qualifying)

        matched_track_ids = set()
        output: List[Optional[TrackedHuman]] = [None] * len(detections)

        for local_index, (detection, track_id) in enumerate(zip(qualifying, assignment)):

            original_index = qualifying_indices[local_index]

            if track_id is None:
                track_id = self._create_track(camera_id, detection, timestamp)
            else:
                self._update_matched_track(tracks[track_id], detection, timestamp)

            matched_track_ids.add(track_id)

            state = TrackState.TRACKED if tracks[track_id].frames_seen > 1 else TrackState.NEW
            output[original_index] = self._snapshot(tracks[track_id], state)

        for index, detection in enumerate(detections):

            if output[index] is None:
                # Disqualified by minimum_confidence -- a disposable,
                # never-persisted NEW track, never eligible to be
                # matched again (see class docstring).
                output[index] = self._ephemeral_snapshot(camera_id, detection, timestamp)

        remainder = self._advance_unmatched(tracks, matched_track_ids, timestamp)

        return tuple(output) + tuple(remainder)

    # =====================================================
    # Matching (Phase 5)
    # =====================================================

    def _assign(
        self,
        tracks: Dict[str, _Track],
        qualifying: Sequence[RawHumanDetection],
    ) -> List[Optional[str]]:

        # Greedy, deterministic one-to-one assignment: every candidate
        # (detection_index, track_id) pair that qualifies as a match
        # (tracking/cost_functions.match_score) is considered, best
        # score first; ties broken by detection index then track_id so
        # the result never depends on dict/iteration order. A track or
        # a detection already claimed by a better-scoring pair is
        # skipped -- each track matches at most one detection, and vice
        # versa, this cycle.

        candidates = []

        for detection_index, detection in enumerate(qualifying):

            for track_id, track in tracks.items():

                score = match_score(
                    detection.bounding_box, track.bounding_box,
                    iou_threshold=self.iou_threshold,
                    max_centroid_distance=self.max_centroid_distance,
                )

                if score is not None:
                    candidates.append((score, detection_index, track_id))

        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

        assignment: List[Optional[str]] = [None] * len(qualifying)
        claimed_tracks = set()

        for _, detection_index, track_id in candidates:

            if assignment[detection_index] is not None:
                continue

            if track_id in claimed_tracks:
                continue

            assignment[detection_index] = track_id
            claimed_tracks.add(track_id)

        return assignment

    # =====================================================
    # Track lifecycle (Phase 6)
    # =====================================================

    def _next_track_id(self, camera_id: str) -> str:

        next_index = self._next_id.get(camera_id, 0) + 1
        self._next_id[camera_id] = next_index

        return f"{camera_id}-T{next_index}"

    # =====================================================

    def _create_track(self, camera_id: str, detection: RawHumanDetection, timestamp: float) -> str:

        track_id = self._next_track_id(camera_id)

        self._tracks[camera_id][track_id] = _Track(
            track_id=track_id,
            camera_id=camera_id,
            bounding_box=detection.bounding_box,
            confidence=detection.confidence,
            age=1,
            frames_seen=1,
            frames_missing=0,
            last_timestamp=timestamp,
        )

        return track_id

    # =====================================================

    def _update_matched_track(self, track: _Track, detection: RawHumanDetection, timestamp: float) -> None:

        track.bounding_box = detection.bounding_box
        track.confidence = detection.confidence
        track.age += 1
        track.frames_seen += 1
        track.frames_missing = 0
        track.last_timestamp = timestamp

    # =====================================================

    def _advance_unmatched(
        self,
        tracks: Dict[str, _Track],
        matched_track_ids: set,
        timestamp: float,
    ) -> List[TrackedHuman]:

        remainder = []

        for track_id in list(tracks.keys()):

            if track_id in matched_track_ids:
                continue

            track = tracks[track_id]
            track.age += 1
            track.frames_missing += 1

            if track.frames_missing > self.max_missing_frames:

                remainder.append(self._snapshot(track, TrackState.EXPIRED))
                del tracks[track_id]

            else:

                remainder.append(self._snapshot(track, TrackState.MISSING))

        return remainder

    # =====================================================

    def _ephemeral_snapshot(self, camera_id: str, detection: RawHumanDetection, timestamp: float) -> TrackedHuman:

        return TrackedHuman(
            track_id=self._next_track_id(camera_id),
            camera_id=camera_id,
            bounding_box=detection.bounding_box,
            confidence=detection.confidence,
            state=TrackState.NEW,
            age=1,
            frames_seen=1,
            frames_missing=0,
            last_timestamp=timestamp,
        )

    # =====================================================

    def _snapshot(self, track: _Track, state: TrackState) -> TrackedHuman:

        return TrackedHuman(
            track_id=track.track_id,
            camera_id=track.camera_id,
            bounding_box=track.bounding_box,
            confidence=track.confidence,
            state=state,
            age=track.age,
            frames_seen=track.frames_seen,
            frames_missing=track.frames_missing,
            last_timestamp=track.last_timestamp,
        )
