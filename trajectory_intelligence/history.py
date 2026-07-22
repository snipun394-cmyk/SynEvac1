import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


# =====================================================
# Live Occupant Trajectory, Movement Anomaly & Route-Deviation
# Intelligence milestone, Phase 4 -- the ONE piece of state this
# package owns beyond live_occupants.occupant.LiveOccupant.history
# (position_samples/velocity_samples/zone_transitions are all already
# bounded and reused as-is, see trajectory.py/route_progress.py/
# anomaly.py). route_distance_m has no home on LiveOccupant/OccupantHistory
# (it is derived from the Navigation Graph, not from perception), so a
# bounded, per-occupant history of it is kept here -- same "immutable
# value + mutable owning registry" convention live_occupants.manager.
# LiveOccupantManager/cross_camera_identity.identity_registry.
# IdentityRegistry already establish, applied to one small new value
# instead of a whole new occupant registry (TrajectoryHistoryStore is
# NOT a second occupant registry -- it is pruned to exactly the
# canonical occupant_id set LiveOccupantManager reports active each
# cycle, see engine.py's own _prune()).
# =====================================================


@dataclass(frozen=True)
class RouteDistanceSample:

    timestamp: float
    distance: float


@dataclass(frozen=True)
class OccupantTrajectoryState:

    route_distance_samples: Tuple[RouteDistanceSample, ...] = field(default_factory=tuple)

    # Phase 11 -- whether this occupant's PREVIOUS cycle was already
    # inside a hazardous zone, so ENTERED_HAZARDOUS_ZONE/REMAINS_IN_
    # HAZARDOUS_ZONE can be told apart (a genuine cross-cycle fact, not
    # derivable from a single cycle's BuildingState alone).
    was_in_hazardous_zone: bool = False

    max_length: int = 30

    def with_route_distance_sample(self, timestamp: float, distance: float) -> "OccupantTrajectoryState":

        sample = RouteDistanceSample(timestamp, distance)
        samples = (*self.route_distance_samples, sample)[-self.max_length:]

        return dataclasses.replace(self, route_distance_samples=samples)

    def with_hazard_zone_flag(self, was_in_hazardous_zone: bool) -> "OccupantTrajectoryState":

        return dataclasses.replace(self, was_in_hazardous_zone=was_in_hazardous_zone)


class TrajectoryHistoryStore:

    # The mutable owning registry -- exactly mirrors LiveOccupantManager's
    # own "immutable value objects, a mutable owning registry by
    # replacing its own stored reference" convention, scoped to one
    # small value instead of the whole occupant.

    def __init__(self, max_length: int = 30):

        self.max_length = max_length
        self._states: Dict[str, OccupantTrajectoryState] = {}

    # =====================================================

    def get(self, occupant_id: str) -> OccupantTrajectoryState:

        return self._states.get(occupant_id, OccupantTrajectoryState(max_length=self.max_length))

    # =====================================================

    def set(self, occupant_id: str, state: OccupantTrajectoryState) -> None:

        self._states[occupant_id] = state

    # =====================================================

    def prune(self, active_occupant_ids) -> None:

        # Phase 16/17's own boundary: this store must never grow
        # unbounded and must never silently keep stale per-occupant
        # state for an occupant LiveOccupantManager no longer reports
        # as canonical (EXITED/EXPIRED) -- pruned to exactly the active
        # set every cycle, the same "no unlimited growth" requirement
        # Phase 4 applies to position history itself.

        stale_ids = set(self._states.keys()) - set(active_occupant_ids)

        for occupant_id in stale_ids:
            del self._states[occupant_id]
