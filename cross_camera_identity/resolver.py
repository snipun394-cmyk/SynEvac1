from abc import ABC, abstractmethod
from typing import Mapping, Optional, Sequence, Tuple

from tracking.track_state import TrackState
from tracking.tracked_human import TrackedHuman

from behavior_recognition.observation import BehaviorObservation

from cross_camera_identity.identity_registry import IdentityRegistry
from cross_camera_identity.matching import CrossCameraMatcher, RuleBasedCrossCameraMatcher
from cross_camera_identity.observation import CrossCameraObservation, ResolvedIdentity
from cross_camera_identity.topology import CameraTopology
from cross_camera_identity.transition_model import TransitionModel


class CrossCameraIdentityResolver(ABC):

    # Cross-Camera Identity Resolution (ReID Framework) milestone,
    # Phase 6/7 -- the seam a future learned cross-camera resolution
    # strategy plugs into. Maintains a single, persistent GLOBAL
    # occupant identity as the SAME physical person moves between
    # DIFFERENT cameras -- explicitly NOT deep-learning ReID, NOT
    # appearance embeddings, NOT facial recognition (Phase 5's own
    # exclusion list). Local tracking IDs (tracking.tracked_human.
    # TrackedHuman.track_id) are never altered or exposed as global IDs
    # by this seam -- see cross_camera_identity.identity_registry.
    # IdentityRegistry's own "OCC-N" format, structurally distinct from
    # any tracker-local id.

    @abstractmethod
    def resolve(
        self,
        camera_id: str,
        timestamp: float,
        tracked_humans: Sequence[TrackedHuman],
        behaviors_by_track_id: Mapping[str, BehaviorObservation],
    ) -> Tuple[ResolvedIdentity, ...]:

        # `tracked_humans` is the FULL per-cycle output of one
        # SingleCameraTracker.update() call for one camera (every
        # NEW/TRACKED/MISSING/EXPIRED entry) -- the same convention
        # behavior_recognition.recognizer.BehaviorRecognizer.recognize()
        # already establishes, for the same reason: EXPIRED entries are
        # needed here too, to release a departed track's binding
        # (cross_camera_identity.identity_registry.IdentityRegistry.
        # release()) the moment it happens.
        #
        # `behaviors_by_track_id` is keyed by LOCAL track_id, may be
        # empty (no BehaviorRecognizer configured upstream -- an honest
        # "no evidence", not an error).
        #
        # Output: exactly one ResolvedIdentity for each entry in state
        # NEW or TRACKED, in the same relative order -- MISSING/EXPIRED
        # entries produce no output (nothing currently observed to
        # honestly assign an identity to).
        ...


class RuleBasedCrossCameraIdentityResolver(CrossCameraIdentityResolver):

    # A clean, deterministic engineering baseline, composing exactly
    # the three collaborators Phase 2-6 name: IdentityRegistry (pure
    # storage), TransitionModel (expiry/possible-destinations policy),
    # and CrossCameraMatcher (scoring/decision). One instance is meant
    # to be SHARED across every camera in a deployment (constructed
    # once, handed to live_camera_pipeline.pipeline.LiveCameraPipeline)
    # -- cross-camera resolution is meaningless with a separate
    # registry per camera.

    def __init__(
        self,
        topology: CameraTopology,
        registry: Optional[IdentityRegistry] = None,
        transition_model: Optional[TransitionModel] = None,
        matcher: Optional[CrossCameraMatcher] = None,
    ):

        self.topology = topology
        self.registry = registry if registry is not None else IdentityRegistry()
        self.transition_model = transition_model if transition_model is not None else TransitionModel(topology)
        self.matcher = matcher if matcher is not None else RuleBasedCrossCameraMatcher()

    # =====================================================

    def resolve(
        self,
        camera_id: str,
        timestamp: float,
        tracked_humans: Sequence[TrackedHuman],
        behaviors_by_track_id: Mapping[str, BehaviorObservation],
    ) -> Tuple[ResolvedIdentity, ...]:

        results = []
        claimed_this_cycle = set()

        for tracked in tracked_humans:

            if tracked.state == TrackState.EXPIRED:
                self.registry.release(camera_id, tracked.track_id)
                continue

            if tracked.state == TrackState.MISSING:
                continue

            global_id = self._resolve_one(camera_id, timestamp, tracked, behaviors_by_track_id, claimed_this_cycle)
            claimed_this_cycle.add(global_id)

            results.append(
                ResolvedIdentity(camera_id=camera_id, track_id=tracked.track_id, global_id=global_id, timestamp=timestamp)
            )

        self._expire_stale_identities(timestamp)

        return tuple(results)

    # =====================================================

    def _resolve_one(self, camera_id, timestamp, tracked, behaviors_by_track_id, claimed_this_cycle) -> str:

        existing_global_id = self.registry.lookup_binding(camera_id, tracked.track_id)

        if existing_global_id is not None:
            # Same local track continuing on the same camera -- no
            # cross-camera matching needed at all, just a refresh.
            self.registry.touch(existing_global_id, camera_id, tracked.track_id, timestamp)
            return existing_global_id

        candidates = [
            record for record in self.registry.unbound_records()
            if record.last_camera_id != camera_id
            and record.global_id not in claimed_this_cycle
            and not self.transition_model.is_expired(record, timestamp)
        ]

        observation = CrossCameraObservation(
            camera_id=camera_id, track_id=tracked.track_id, timestamp=timestamp,
            track_confidence=tracked.confidence, track_age=tracked.age,
            behavior=behaviors_by_track_id.get(tracked.track_id),
        )

        matched_global_id = self.matcher.find_match(observation, candidates, self.topology)

        if matched_global_id is not None:
            self.registry.touch(matched_global_id, camera_id, tracked.track_id, timestamp)
            return matched_global_id

        return self.registry.create(camera_id, tracked.track_id, timestamp)

    # =====================================================

    def _expire_stale_identities(self, now) -> None:

        for record in self.registry.unbound_records():

            if self.transition_model.is_expired(record, now):
                self.registry.delete(record.global_id)
