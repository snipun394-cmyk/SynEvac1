import dataclasses
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class GlobalIdentityRecord:

    # Cross-Camera Identity Resolution (ReID Framework) milestone,
    # Phase 3 -- one global occupant identity's current state, entirely
    # owned by IdentityRegistry. last_track_id is None exactly when this
    # identity is currently UNBOUND (its local track just disappeared
    # from last_camera_id -- a candidate departure, still "active" and
    # eligible for cross-camera matching until
    # transition_model.TransitionModel decides it has timed out) -- the
    # same "Optional means genuinely absent, never fabricated" pattern
    # every other model in this codebase already follows.

    global_id: str
    last_camera_id: str
    last_track_id: Optional[str]
    last_timestamp: float
    created_at: float


class IdentityRegistry:

    # Phase 3's own required responsibilities, and nothing else --
    # creating global occupant IDs, maintaining active identities,
    # mapping local track IDs to global IDs, and tracking last
    # observation/camera/timestamp. Deliberately does NOT decide
    # expiry policy or matching strategy itself (see
    # cross_camera_identity.transition_model.TransitionModel and
    # cross_camera_identity.matching.CrossCameraMatcher) -- this class
    # is pure storage/bookkeeping, not policy.
    #
    # Global IDs are minted sequentially ("OCC-1", "OCC-2", ...) --
    # deterministic, no randomness (Phase 9's own requirement), and
    # NEVER reused once a record is deleted (Phase 8's "identity reuse
    # prevention" -- the counter only ever increments).

    def __init__(self):

        self._records: Dict[str, GlobalIdentityRecord] = {}
        self._binding: Dict[Tuple[str, str], str] = {}
        self._next_id = 0

    # =====================================================
    # Local -> global lookup
    # =====================================================

    def lookup_binding(self, camera_id: str, local_track_id: str) -> Optional[str]:

        return self._binding.get((camera_id, local_track_id))

    # =====================================================
    # Creation
    # =====================================================

    def create(self, camera_id: str, local_track_id: str, timestamp: float) -> str:

        self._next_id += 1
        global_id = f"OCC-{self._next_id}"

        self._records[global_id] = GlobalIdentityRecord(
            global_id=global_id, last_camera_id=camera_id, last_track_id=local_track_id,
            last_timestamp=timestamp, created_at=timestamp,
        )
        self._binding[(camera_id, local_track_id)] = global_id

        return global_id

    # =====================================================
    # Continuation / transition
    # =====================================================

    def touch(self, global_id: str, camera_id: str, local_track_id: str, timestamp: float) -> None:

        # Refreshes an EXISTING global identity's current position --
        # used both for "same local track, still there" (camera_id/
        # local_track_id unchanged) and for "a cross-camera match just
        # happened" (camera_id/local_track_id now point at the NEW
        # camera/track) -- the bookkeeping is identical either way.

        record = self._records[global_id]

        self._records[global_id] = dataclasses.replace(
            record, last_camera_id=camera_id, last_track_id=local_track_id, last_timestamp=timestamp,
        )
        self._binding[(camera_id, local_track_id)] = global_id

    # =====================================================
    # Departure
    # =====================================================

    def release(self, camera_id: str, local_track_id: str) -> None:

        # The local track behind this binding is gone (tracking.
        # tracker.SingleCameraTracker reported it EXPIRED) -- drops the
        # (camera_id, local_track_id) -> global_id binding, but the
        # global identity record itself is KEPT alive (last_track_id
        # set to None) so it remains a candidate for a cross-camera
        # match on a later cycle, until transition_model.TransitionModel
        # decides it has timed out (Phase 8's "temporary
        # disappearance"/"camera exit" requirements).

        global_id = self._binding.pop((camera_id, local_track_id), None)

        if global_id is None:
            return

        record = self._records.get(global_id)

        if record is not None:
            self._records[global_id] = dataclasses.replace(record, last_track_id=None)

    # =====================================================
    # Queries
    # =====================================================

    def unbound_records(self) -> Tuple[GlobalIdentityRecord, ...]:

        # Every identity currently NOT bound to any live local track --
        # exactly the candidate pool for cross-camera matching
        # (cross_camera_identity.matching.CrossCameraMatcher). Sorted by
        # global_id for fully deterministic downstream iteration/tie-
        # breaking (Phase 9's "no randomness").

        return tuple(sorted(
            (record for record in self._records.values() if record.last_track_id is None),
            key=lambda record: record.global_id,
        ))

    def get(self, global_id: str) -> Optional[GlobalIdentityRecord]:

        return self._records.get(global_id)

    # =====================================================
    # Cleanup
    # =====================================================

    def delete(self, global_id: str) -> None:

        # Permanently forgets this identity -- global_id itself is
        # never reused (Phase 8), and any binding still pointing at it
        # (only possible for an already-unbound record, since a bound
        # one is still actively touch()'d every cycle) is dropped too.

        record = self._records.pop(global_id, None)

        if record is not None and record.last_track_id is not None:
            self._binding.pop((record.last_camera_id, record.last_track_id), None)

    # =====================================================

    def __len__(self) -> int:

        return len(self._records)
