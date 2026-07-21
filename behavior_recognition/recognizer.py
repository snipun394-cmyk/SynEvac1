from abc import ABC, abstractmethod
from typing import Sequence, Tuple

from tracking.tracked_human import TrackedHuman

from behavior_recognition.observation import BehaviorObservation


class BehaviorRecognizer(ABC):

    # The seam a future ML-based behavior model plugs into -- Human
    # Behavior Recognition Framework milestone, Phase 4. Recognizes
    # behavior using ONLY temporal tracking information (a track's own
    # bounding-box/confidence history over time, already available from
    # tracking.tracker.SingleCameraTracker) -- explicitly NOT pose
    # estimation, NOT a neural network, NOT appearance-based.
    #
    # Completely independent of AI, BuildingState, Command Center,
    # Advisory, RTSP, and any YOLO backend (Phase 12, enforced by
    # tests/test_behavior_recognition_architecture_guards.py) --
    # depends only on TrackedHuman, geometry, and time.

    @abstractmethod
    def recognize(
        self,
        camera_id: str,
        timestamp: float,
        tracked_humans: Sequence[TrackedHuman],
    ) -> Tuple[BehaviorObservation, ...]:

        # `tracked_humans` is the FULL per-cycle output of one
        # SingleCameraTracker.update() call for one camera (every
        # NEW/TRACKED/MISSING/EXPIRED entry) -- not just the
        # currently-observed subset. A conforming implementation:
        #
        #   - emits exactly one BehaviorObservation for each entry in
        #     state NEW or TRACKED, in the same relative order those
        #     entries appeared in `tracked_humans` (there IS a current
        #     observation to honestly report for these);
        #   - emits NOTHING for an entry in state MISSING or EXPIRED
        #     (there is no current observation -- fabricating one for
        #     someone not actually seen this cycle would violate this
        #     milestone's own "no fabrication" requirement), though it
        #     may still use a MISSING/EXPIRED entry's presence to
        #     advance or evict its own internal history bookkeeping.
        #
        # Because tracking.tracker.SingleCameraTracker.update() itself
        # guarantees its own first len(detections) entries are always
        # exactly the NEW/TRACKED ones, in the same order those
        # detections were supplied, a caller can always zip() its
        # original per-cycle detections against this method's return
        # value with no additional bookkeeping (see
        # live_camera_pipeline.pipeline.LiveCameraPipeline's own
        # integration of both seams together).
        ...
