from abc import ABC, abstractmethod
from typing import Sequence, Tuple

from live_camera_pipeline.human_detector import RawHumanDetection

from tracking.tracked_human import TrackedHuman


class SingleCameraTracker(ABC):

    # The seam a real single-camera tracking strategy plugs into --
    # Single-Camera Tracking Framework milestone, Phase 3. Maintains
    # STABLE LOCAL identities across consecutive frames from the SAME
    # camera only. Explicitly NOT cross-camera re-identification (that
    # remains live_camera_pipeline.identity_resolver.IdentityResolver's
    # job, entirely untouched by this package), NOT behavior
    # recognition, NOT pose estimation.
    #
    # Completely independent of YOLO, BuildingState, AI, Command
    # Center, Advisory, and RTSP -- depends only on RawHumanDetection,
    # geometry, and time (Phase 12, enforced by
    # tests/test_tracking_architecture_guards.py). A caller is free to
    # invoke update() for as many different camera_id values as it
    # likes against one SingleCameraTracker instance; a conforming
    # implementation must keep each camera_id's tracks fully isolated
    # from every other's (never match a CAM-A detection against a
    # CAM-B track).

    @abstractmethod
    def update(
        self,
        camera_id: str,
        timestamp: float,
        detections: Sequence[RawHumanDetection],
    ) -> Tuple[TrackedHuman, ...]:

        # Output contract: the returned tuple's first len(detections)
        # entries correspond POSITIONALLY, one-to-one, to `detections`
        # itself -- output[i] is always the TrackedHuman produced for
        # detections[i] this cycle (state NEW or TRACKED), regardless
        # of that detection's own confidence or geometry. A caller can
        # therefore always zip(detections, result) with no additional
        # bookkeeping (see live_camera_pipeline.pipeline.
        # LiveCameraPipeline's tracker integration). Any further
        # entries beyond that prefix are tracks that existed before
        # this call but were NOT matched to any detection this cycle
        # (state MISSING, still coasting within max_missing_frames --
        # or EXPIRED, reported exactly once on the cycle it is
        # deleted, then never returned again).
        #
        # A concrete implementation is free to decide internally which
        # detections influence PERSISTENT track state (e.g.
        # SimpleSingleCameraTracker's minimum_confidence filter,
        # tracking/simple_tracker.py, gives a low-confidence detection
        # a one-off, disposable track rather than matching/creating a
        # persistent one) -- but every detection still gets exactly
        # one positional output entry.
        ...
