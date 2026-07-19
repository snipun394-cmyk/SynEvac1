from typing import Dict, Iterable, Optional

from multi_camera_fusion.confidence import ConfidenceFusionStrategy, HighestConfidenceStrategy
from multi_camera_fusion.track import FusedTrack, FusionResult, TrackHistory, TrackTransition


DEFAULT_MAX_HISTORY_LENGTH = 20


class MultiCameraFusionEngine:

    # Merges raw, per-camera Detections (virtual_camera/detection.py's
    # Detection today; anything duck-type-compatible with it
    # tomorrow -- see below) into one unified FusedTrack per real
    # person, across every camera that currently sees them.
    #
    # -----------------------------------------------------------
    # Association -- the one honest limitation this milestone accepts
    # -----------------------------------------------------------
    # This engine associates detections into a track by Detection.
    # occupant_id equality, and by nothing else. It performs no
    # appearance matching, no re-identification, no geometric
    # trajectory prediction -- every one of those is a computer-vision
    # problem, explicitly out of scope for this milestone ("Do NOT
    # implement computer vision"). In Simulation mode this is not a
    # simplification that loses anything: occupant_id already IS the
    # single, honest, ground-truth identity every camera's Virtual
    # Camera resolves independently (see virtual_camera/camera.py),
    # so two Detections sharing an occupant_id are, with certainty,
    # the same physical person, and different occupant_ids are, with
    # certainty, different people. A future Live CCTV deployment's own
    # DetectionProvider is the one place a real re-identification model
    # would need to live -- its job would be producing Detections whose
    # occupant_id already reflects a resolved cross-camera identity;
    # this engine would then fuse them exactly as it fuses Simulation's
    # today, completely unchanged (Phase 7).
    #
    # -----------------------------------------------------------
    # Detector-agnostic by construction
    # -----------------------------------------------------------
    # Nothing in this module imports virtual_camera, camera_manager,
    # simulator, or any Ground Truth package (enforced by
    # tests.test_multi_camera_fusion.MultiCameraFusionPackageDependency
    # DirectionTests) -- fuse() accepts any iterable of plain objects
    # exposing occupant_id/camera_id/floor_id/zone_id/confidence/
    # classification/human_state/is_false_positive, the exact shape
    # Detection already has. This is what "works with the existing
    # DetectionProvider interface" and "the only thing that changes is
    # the DetectionProvider" both mean in practice: this class never
    # calls a DetectionProvider itself, so it cannot know or care which
    # one produced what it was handed.
    #
    # -----------------------------------------------------------
    # State
    # -----------------------------------------------------------
    # The one piece of memory this class keeps between fuse() calls is
    # each track's TrackHistory (previous/current camera, last
    # observation time, zone/floor transitions) -- everything else in
    # a FusedTrack is recomputed fresh every call from whatever
    # Detections were just handed in, same "derived, never stale"
    # discipline the rest of this codebase's Provider-shaped classes
    # already follow.

    def __init__(
        self,
        confidence_strategy: Optional[ConfidenceFusionStrategy] = None,
        max_history_length: int = DEFAULT_MAX_HISTORY_LENGTH,
    ):

        self.confidence_strategy = confidence_strategy or HighestConfidenceStrategy()
        self.max_history_length = max_history_length

        self._histories: Dict[str, TrackHistory] = {}

    # =====================================================

    def fuse(self, detections: Iterable, time: float) -> FusionResult:

        grouped = self._group_by_occupant(detections)

        tracks = tuple(
            self._fuse_group(track_id, group, time)
            for track_id, group in sorted(grouped.items())
        )

        return FusionResult(timestamp=time, tracks=tracks)

    # =====================================================

    def track_history(self, track_id: str) -> Optional[TrackHistory]:

        # Exposed so a caller can inspect a track's history even for an
        # instant where the person was not currently detected by any
        # camera (see FusionResult's own docstring on why an absent
        # track's history still survives internally).

        return self._histories.get(track_id)

    # =====================================================
    # Internals
    # =====================================================

    def _group_by_occupant(self, detections):

        grouped: Dict[str, list] = {}

        for detection in detections:

            if detection.is_false_positive:
                # Phase 2's "prevent duplicate occupants" starts here:
                # a false positive has no real person behind it at all,
                # so fusing it would fabricate a track for someone who
                # does not exist, in every camera that happened to
                # generate one this tick.
                continue

            grouped.setdefault(detection.occupant_id, []).append(detection)

        return grouped

    # =====================================================

    def _fuse_group(self, track_id, group, time) -> FusedTrack:

        # The single highest-individual-confidence Detection in this
        # group is treated as the most trustworthy source for every
        # non-confidence field (classification/human_state/floor/zone/
        # camera) -- confidence itself is the one field actually
        # combined across the whole group, via whichever
        # ConfidenceFusionStrategy this engine was built with.

        best = max(group, key=lambda detection: detection.confidence)

        confidences = tuple(detection.confidence for detection in group)
        fused_confidence = self.confidence_strategy.fuse(confidences)

        source_camera_ids = tuple(sorted({detection.camera_id for detection in group}))

        history = self._update_history(track_id, best, time)

        return FusedTrack(
            track_id=track_id,
            source_camera_ids=source_camera_ids,
            floor_id=best.floor_id,
            zone_id=best.zone_id,
            classification=best.classification,
            human_state=best.human_state,
            confidence=fused_confidence,
            timestamp=time,
            history=history,
        )

    # =====================================================

    def _update_history(self, track_id, best_detection, time) -> TrackHistory:

        previous = self._histories.get(track_id)

        current_camera_id = best_detection.camera_id
        current_zone_id = best_detection.zone_id
        current_floor_id = best_detection.floor_id

        if previous is None:

            previous_camera_id = None
            zone_transitions = ()
            floor_transitions = ()

        else:

            # previous_camera_id is exactly "whichever camera was
            # current as of the last fuse() call" -- equal to
            # current_camera_id on every call except the one instant a
            # handover actually happens (Phase 3), after which both
            # settle back to the new camera until the next handover.
            previous_camera_id = previous.current_camera_id

            zone_transitions = self._record_transition(
                previous.zone_transitions, previous.current_zone_id, current_zone_id, time,
            )

            floor_transitions = self._record_transition(
                previous.floor_transitions, previous.current_floor_id, current_floor_id, time,
            )

        history = TrackHistory(
            track_id=track_id,
            previous_camera_id=previous_camera_id,
            current_camera_id=current_camera_id,
            last_observation_time=time,
            current_zone_id=current_zone_id,
            current_floor_id=current_floor_id,
            zone_transitions=zone_transitions,
            floor_transitions=floor_transitions,
        )

        self._histories[track_id] = history

        return history

    # =====================================================

    def _record_transition(self, existing_transitions, previous_value, current_value, time):

        if current_value == previous_value:
            return existing_transitions

        updated = existing_transitions + (
            TrackTransition(time=time, from_value=previous_value, to_value=current_value),
        )

        if len(updated) > self.max_history_length:
            updated = updated[-self.max_history_length:]

        return updated
