from dataclasses import dataclass, field
from typing import Optional, Tuple

from perception.models.human_observation import HumanClassification, HumanState


@dataclass(frozen=True)
class TrackTransition:

    # One recorded change of zone or floor -- Phase 5's "prepare the
    # structure for future trajectory estimation" requirement, kept as
    # the plainest possible (time, from, to) record rather than
    # anything trajectory-shaped itself (no interpolated path, no
    # speed/heading estimate -- that is explicitly a future module's
    # job, not this one's).

    time: float
    from_value: Optional[str]
    to_value: Optional[str]


@dataclass(frozen=True)
class TrackHistory:

    # A short, bounded memory of one track across fuse() calls --
    # the one piece of state MultiCameraFusionEngine keeps between
    # calls (everything else it produces is recomputed fresh each
    # time, same "derived, not stored" discipline as CameraVisibility/
    # FloorCoverage). previous_camera_id/current_camera_id together are
    # exactly Phase 3's Camera Handover signal: they differ for exactly
    # one fuse() call at the instant a handover happens, and match again
    # on every call afterward until the next one.

    track_id: str

    previous_camera_id: Optional[str]
    current_camera_id: Optional[str]

    last_observation_time: float

    current_zone_id: Optional[str]
    current_floor_id: Optional[str]

    # Bounded to MultiCameraFusionEngine.max_history_length entries,
    # oldest dropped first -- "short track history" is Phase 5's own
    # phrase, not an unbounded log.
    zone_transitions: Tuple[TrackTransition, ...] = field(default_factory=tuple)
    floor_transitions: Tuple[TrackTransition, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FusedTrack:

    # One unified person, as of one fuse() call -- Phase 2's required
    # shape exactly: Track ID, Source camera(s), Current floor,
    # Current zone, Current classification, Current human state,
    # Confidence, plus the TrackHistory Phase 5 asks to maintain
    # alongside it.
    #
    # track_id is the occupant_id every source Detection agreed on --
    # this milestone never invents a separate track identity, because
    # doing so honestly (reconciling two DIFFERENT camera-local ids as
    # the same physical person) is a re-identification problem that
    # requires computer vision, explicitly out of scope here. A future
    # Live CCTV DetectionProvider is the one place that re-
    # identification would need to happen -- see multi_camera_fusion/
    # engine.py's own module docstring.

    track_id: str

    source_camera_ids: Tuple[str, ...]

    floor_id: Optional[str]
    zone_id: Optional[str]

    classification: HumanClassification
    human_state: Optional[HumanState]

    confidence: float

    timestamp: float

    history: TrackHistory


@dataclass(frozen=True)
class FusionResult:

    # One fuse() call's complete output -- every currently-tracked
    # person, at this instant, from every camera combined. A track
    # absent from `tracks` this call was not detected by ANY camera
    # this instant (evacuated, or genuinely between two cameras'
    # coverage) -- its TrackHistory is still retained internally by
    # MultiCameraFusionEngine (see that class's own docstring) so a
    # later reappearance continues the same track rather than starting
    # a new one.

    timestamp: float
    tracks: Tuple[FusedTrack, ...] = field(default_factory=tuple)

    def track(self, track_id: str) -> Optional[FusedTrack]:

        for fused_track in self.tracks:

            if fused_track.track_id == track_id:
                return fused_track

        return None
