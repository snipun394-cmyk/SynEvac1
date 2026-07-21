from abc import ABC, abstractmethod
from typing import Optional, Sequence

from behavior_recognition.observation import RecognizedBehavior

from cross_camera_identity.identity_registry import GlobalIdentityRecord
from cross_camera_identity.observation import CrossCameraObservation
from cross_camera_identity.topology import CameraTopology


DEFAULT_MIN_TRANSITION_TIME = 0.0
DEFAULT_MAX_TRANSITION_TIME = 30.0
DEFAULT_MIN_TRACK_AGE_FOR_MATCHING = 1


class CrossCameraMatcher(ABC):

    # Cross-Camera Identity Resolution (ReID Framework) milestone,
    # Phase 5 -- the seam a future learned matcher plugs into. Uses
    # ONLY camera topology, time continuity, track age/history, and
    # (when available) behavior evidence -- explicitly NOT image
    # embeddings, appearance similarity, facial recognition, or any
    # deep-learning model (Phase 5's own exclusion list; also enforced
    # mechanically, see tests/
    # test_cross_camera_identity_architecture_guards.py).

    @abstractmethod
    def find_match(
        self,
        observation: CrossCameraObservation,
        candidates: Sequence[GlobalIdentityRecord],
        topology: CameraTopology,
    ) -> Optional[str]:

        # `candidates` is expected to already be filtered to identities
        # currently UNBOUND and not yet expired (cross_camera_identity.
        # identity_registry.IdentityRegistry.unbound_records() /
        # cross_camera_identity.transition_model.TransitionModel.
        # is_expired()) -- this method's only job is deciding WHICH of
        # those (if any) is the same physical person as `observation`.
        # Returns the matched global_id, or None -- a caller (see
        # cross_camera_identity.resolver.RuleBasedCrossCameraIdentityResolver)
        # mints a brand new global identity when None is returned,
        # never forcing a match that isn't genuinely justified.
        ...


class RuleBasedCrossCameraMatcher(CrossCameraMatcher):

    # A clean, deterministic engineering baseline -- topology
    # plausibility (is this camera-to-camera transition even possible,
    # and within a believable time window?) first, then the candidate
    # whose actual elapsed time is closest to that transition's own
    # expected_transition_time wins. Ties broken by global_id string,
    # so results never depend on dict/candidate iteration order (Phase
    # 9's "no randomness").
    #
    # min_track_age_for_matching guards against matching a brand-new,
    # possibly-flickery 1-frame-old local track (tracking.tracked_human.
    # TrackedHuman.age) against a real prior identity -- a track this
    # young is treated as too unreliable a candidate to commit a cross-
    # camera match to; it becomes its own new identity instead (which a
    # LATER cycle's cross-camera resolution can still correct, since
    # candidates simply stay available until they expire).
    #
    # require_departure_motion (disabled by default, since it needs
    # behavior data this package cannot assume is always configured
    # upstream) optionally rejects a candidate whose most recent
    # behavior_recognition.observation.RecognizedBehavior was STATIONARY
    # right before it disappeared -- a person who was standing still,
    # not moving toward any exit, is weaker evidence of an imminent
    # camera transition than one who was WALKING/RUNNING.

    def __init__(
        self,
        default_min_transition_time: float = DEFAULT_MIN_TRANSITION_TIME,
        default_max_transition_time: float = DEFAULT_MAX_TRANSITION_TIME,
        min_track_age_for_matching: int = DEFAULT_MIN_TRACK_AGE_FOR_MATCHING,
        require_departure_motion: bool = False,
    ):

        self.default_min_transition_time = default_min_transition_time
        self.default_max_transition_time = default_max_transition_time
        self.min_track_age_for_matching = min_track_age_for_matching
        self.require_departure_motion = require_departure_motion

    # =====================================================

    def find_match(
        self,
        observation: CrossCameraObservation,
        candidates: Sequence[GlobalIdentityRecord],
        topology: CameraTopology,
    ) -> Optional[str]:

        if observation.track_age < self.min_track_age_for_matching:
            return None

        if self.require_departure_motion and observation.behavior is not None:

            if observation.behavior.recognized_behavior == RecognizedBehavior.STATIONARY:
                return None

        best_global_id: Optional[str] = None
        best_score: Optional[float] = None

        for candidate in candidates:

            elapsed = observation.timestamp - candidate.last_timestamp

            if elapsed < 0:
                # A candidate that (per its own last_timestamp) has not
                # even departed yet as of this observation's time --
                # never an honest match.
                continue

            plausible, expected = topology.is_plausible_transition(
                candidate.last_camera_id, observation.camera_id, elapsed,
                self.default_min_transition_time, self.default_max_transition_time,
            )

            if not plausible:
                continue

            score = abs(elapsed - expected) if expected is not None else elapsed

            if (
                best_score is None
                or score < best_score
                or (score == best_score and candidate.global_id < best_global_id)
            ):
                best_score = score
                best_global_id = candidate.global_id

        return best_global_id
