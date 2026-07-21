from dataclasses import dataclass
from typing import Optional, Tuple

from tracking.track_state import TrackState


@dataclass(frozen=True)
class TrackedHuman:

    # One track's snapshot, as of one SingleCameraTracker.update() call
    # -- Single-Camera Tracking Framework milestone, Phase 4. Contains
    # ONLY tracking information -- geometry, confidence, and lifecycle
    # bookkeeping local to one camera. Deliberately contains NONE of:
    #
    #   occupant_id        -- a global identity is IdentityResolver's
    #                          job (live_camera_pipeline/identity_
    #                          resolver.py), never this package's.
    #   building zone/floor -- zone/floor localization is
    #                          RawHumanDetection's own honest-uncertain
    #                          field, unrelated to *tracking* geometry.
    #   behavior/human_state -- behavior recognition is an explicitly
    #                          out-of-scope future milestone.
    #   AI state / BuildingState references -- this package has no
    #                          dependency on either (see Phase 12
    #                          architecture guards,
    #                          tests/test_tracking_architecture_guards.py).
    #
    # track_id is LOCAL and TEMPORARY -- stable only within this one
    # camera, only for as long as this track stays alive (never
    # survives past EXPIRED). It is NOT a global occupant identity and
    # NOT a cross-camera re-identification result (Phase 8's identity
    # boundary). Downstream, a track_id is used purely as a more
    # stable substitute for a single frame's own local_track_id when
    # feeding IdentityResolver (see live_camera_pipeline/pipeline.py) --
    # IdentityResolver.resolve() still performs the one and only global
    # identity assignment.

    track_id: str
    camera_id: str

    bounding_box: Optional[Tuple[float, float, float, float]]
    confidence: float

    state: TrackState

    # Cycles since this track was created (matched or not) -- Phase 4's
    # "age" field, incremented once every update() call this track
    # survives.
    age: int

    # Cycles in which this track was actually matched to a detection
    # (including its creation cycle) -- never incremented while
    # MISSING/EXPIRED.
    frames_seen: int

    # CONSECUTIVE cycles since the last successful match -- reset to 0
    # the instant a match happens again; this is what
    # max_missing_frames is compared against.
    frames_missing: int

    # Timestamp of the last cycle this track was actually MATCHED to a
    # detection (not merely "still alive but coasting") -- distinct
    # from frames_missing (a count) by giving a caller the actual last-
    # seen time, useful once real wall-clock cycles are not evenly
    # spaced (a real RTSP-driven pipeline, unlike a fixed-rate replay).
    last_timestamp: float
