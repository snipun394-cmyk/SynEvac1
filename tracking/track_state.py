from enum import Enum, auto


class TrackState(Enum):

    # A track's lifecycle, entirely local to one SingleCameraTracker
    # instance's bookkeeping for one camera_id -- never exposed to, or
    # meaningful outside of, this package (Single-Camera Tracking
    # Framework milestone, Phase 6).

    # First cycle this physical-looking detection was matched to a
    # brand new track (no prior track it could continue).
    NEW = auto()

    # Matched to an existing track this cycle -- local_track_id is
    # stable and continuing from a previous cycle.
    TRACKED = auto()

    # Not matched to any detection this cycle, but still within
    # max_missing_frames -- the track is coasting on its last known
    # position/box, not yet deleted, so a reappearing detection can
    # still resume the same track_id (Phase 6's "temporarily missing"
    # requirement).
    MISSING = auto()

    # frames_missing has just exceeded max_missing_frames -- reported
    # exactly once, on the cycle it happens, then the track is removed
    # from internal state and never returned again.
    EXPIRED = auto()
