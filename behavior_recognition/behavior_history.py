from collections import deque
from typing import Deque, Dict, Optional, Tuple

from behavior_recognition.metrics import BoundingBox, Point, Sample, WorldSample


DEFAULT_MAX_HISTORY_LENGTH = 30


class BehaviorHistory:

    # Human Behavior Recognition Framework milestone, Phase 7 -- a
    # bounded, per-(camera_id, track_id) memory of recent
    # (timestamp, bounding_box) samples, entirely local to this package
    # (never imports tracking.simple_tracker's own internal _Track --
    # this is a wholly independent, smaller-scoped store keyed only by
    # the two plain strings/identifiers a caller already has).
    #
    # "No memory leaks" (Phase 7's own requirement) is satisfied two
    # ways: (1) each individual track's sample deque is itself bounded
    # by max_length (collections.deque(maxlen=...) evicts its own
    # oldest entry automatically on append -- no unbounded growth per
    # track), and (2) clear(camera_id, track_id) lets a caller (see
    # rule_based_recognizer.py, called when a TrackedHuman reports
    # state=EXPIRED) drop a track's entire entry the moment tracking.
    # tracker.SingleCameraTracker itself forgets that track -- otherwise
    # this dict would grow by one entry for every track_id that ever
    # existed, for the lifetime of the process.

    def __init__(self, max_length: int = DEFAULT_MAX_HISTORY_LENGTH):

        self.max_length = max_length
        self._samples: Dict[Tuple[str, str], Deque[Sample]] = {}

        # Camera Calibration & World Coordinate Projection milestone,
        # Phase 6 -- a SEPARATE, parallel store for world-space
        # (meters) positions, kept alongside (never merged into) the
        # pixel-space `_samples` above: a track may have pixel samples
        # with no world samples at all (no calibration configured for
        # that camera), and the two are measured in different units, so
        # conflating them into one deque would be dishonest. clear()
        # below drops both together -- there is exactly one lifecycle
        # per track, regardless of how many kinds of samples it has.
        self._world_samples: Dict[Tuple[str, str], Deque[WorldSample]] = {}

    # =====================================================

    def append(
        self,
        camera_id: str,
        track_id: str,
        timestamp: float,
        bounding_box: Optional[BoundingBox],
        world_position: Optional[Point] = None,
    ) -> None:

        key = (camera_id, track_id)

        if bounding_box is not None:

            if key not in self._samples:
                self._samples[key] = deque(maxlen=self.max_length)

            self._samples[key].append((timestamp, bounding_box))

        if world_position is not None:

            if key not in self._world_samples:
                self._world_samples[key] = deque(maxlen=self.max_length)

            self._world_samples[key].append((timestamp, world_position))

    # =====================================================

    def trim(self, camera_id: str, track_id: str) -> None:

        # Explicit trim to the CURRENT max_length -- append() already
        # enforces this automatically via deque(maxlen=...), so this
        # method exists for a caller that changed max_length after
        # samples already accumulated (deque.maxlen itself is
        # immutable once constructed, so this rebuilds the deque at the
        # new length) or simply wants to assert the invariant
        # explicitly (Phase 7's own named requirement).

        key = (camera_id, track_id)
        existing = self._samples.get(key)

        if existing is not None and existing.maxlen != self.max_length:
            self._samples[key] = deque(existing, maxlen=self.max_length)

        existing_world = self._world_samples.get(key)

        if existing_world is not None and existing_world.maxlen != self.max_length:
            self._world_samples[key] = deque(existing_world, maxlen=self.max_length)

    # =====================================================

    def clear(self, camera_id: Optional[str] = None, track_id: Optional[str] = None) -> None:

        # clear() with no arguments drops everything (every camera,
        # every track). clear(camera_id) drops every track for one
        # camera. clear(camera_id, track_id) drops exactly one track --
        # the case rule_based_recognizer.py calls on EXPIRED.

        if camera_id is None:
            self._samples.clear()
            self._world_samples.clear()
            return

        if track_id is None:
            for key in [key for key in self._samples if key[0] == camera_id]:
                del self._samples[key]
            for key in [key for key in self._world_samples if key[0] == camera_id]:
                del self._world_samples[key]
            return

        self._samples.pop((camera_id, track_id), None)
        self._world_samples.pop((camera_id, track_id), None)

    # =====================================================

    def recent(self, camera_id: str, track_id: str) -> Tuple[Sample, ...]:

        return tuple(self._samples.get((camera_id, track_id), ()))

    # =====================================================

    def recent_world(self, camera_id: str, track_id: str) -> Tuple[WorldSample, ...]:

        return tuple(self._world_samples.get((camera_id, track_id), ()))

    # =====================================================

    def __len__(self) -> int:

        # Number of distinct (camera_id, track_id) tracks currently
        # remembered -- a simple, direct way to assert "no leak" in
        # tests (e.g. this count returning to 0 once every track has
        # expired).
        return len(self._samples)
