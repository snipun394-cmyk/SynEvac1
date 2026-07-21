from dataclasses import dataclass
from typing import Tuple

from cross_camera_identity.identity_registry import GlobalIdentityRecord
from cross_camera_identity.topology import CameraTopology


DEFAULT_IDENTITY_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class PendingTransition:

    # Cross-Camera Identity Resolution (ReID Framework) milestone,
    # Phase 8 -- the explicit "this identity just left a camera and is
    # currently awaiting a possible re-entry" concept. Purely a read-
    # only, derived view over a GlobalIdentityRecord + CameraTopology --
    # never stored anywhere itself (a fresh one is computed on demand,
    # same "derived, not stale" discipline every other snapshot type in
    # this codebase already follows).

    global_id: str
    departed_camera_id: str
    departed_at: float
    possible_destinations: Tuple[str, ...]


class TransitionModel:

    # Owns exactly the POLICY questions cross_camera_identity.
    # identity_registry.IdentityRegistry deliberately does not: how long
    # an unbound identity remains a valid candidate (identity timeout,
    # Phase 8), and where it could plausibly be headed
    # (possible_destinations, reusing CameraTopology -- Phase 4). Kept
    # separate from IdentityRegistry (pure storage) and
    # cross_camera_identity.matching (scoring/decision) so each of the
    # three has exactly one job.

    def __init__(self, topology: CameraTopology, timeout_seconds: float = DEFAULT_IDENTITY_TIMEOUT_SECONDS):

        self.topology = topology
        self.timeout_seconds = timeout_seconds

    # =====================================================

    def pending_transition_for(self, record: GlobalIdentityRecord) -> PendingTransition:

        return PendingTransition(
            global_id=record.global_id,
            departed_camera_id=record.last_camera_id,
            departed_at=record.last_timestamp,
            possible_destinations=self.topology.possible_destinations(record.last_camera_id),
        )

    # =====================================================

    def is_expired(self, record: GlobalIdentityRecord, now: float) -> bool:

        # Identity timeout (Phase 8) -- once an unbound identity has
        # gone longer than timeout_seconds without reappearing anywhere,
        # it is no longer an honest candidate for a future match (too
        # much uncertainty has accumulated) and is eligible for deletion
        # (cross_camera_identity.identity_registry.IdentityRegistry.
        # delete()). Only ever consulted for identities already unbound
        # (last_track_id is None) -- a currently-bound identity is kept
        # fresh by touch() every cycle it continues to be seen, so it
        # never ages toward this timeout at all.

        return (now - record.last_timestamp) > self.timeout_seconds
