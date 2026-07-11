from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping
from uuid import uuid4

from occupancy.observation import OccupancyObservation


@dataclass(frozen=True)
class OccupancySnapshot:

    # An immutable, point-in-time capture of where people are --
    # entirely independent of HazardSnapshot (no shared base class, no
    # import of it, no import of Navigation/Pathfinding/Simulation/
    # Behavior either). Keyed by the same node_id strings
    # HazardSnapshot already uses (Navigation Graph Node.id) purely as
    # a shared coordinate system -- that is the *only* thing that lets
    # a future AI Decision Engine line up a HazardSnapshot and an
    # OccupancySnapshot for the same place; neither snapshot type
    # needs to know the other, or Navigation, exists to make that
    # possible.
    #
    # snapshot_id/timestamp follow the exact same conventions as
    # HazardSnapshot and for the same reasons: a stable identity
    # independent of timestamp/content (for future replay/logging),
    # and a simulation-time this snapshot represents (consumed by an
    # OccupancyProvider.snapshot_at(time) lookup).

    snapshot_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = 0.0

    observations: Mapping[str, OccupancyObservation] = field(default_factory=dict)

    # =====================================================

    def __post_init__(self):

        # MappingProxyType-wrapped so the snapshot stays immutable
        # even though dict/dataclass equality still works -- same
        # technique HazardSnapshot/HazardContribution/BehaviorDecision
        # all already use.
        object.__setattr__(self, "observations", MappingProxyType(dict(self.observations)))

    # =====================================================
    # Total accessor -- always returns a usable observation, never
    # raises and never returns None, mirroring HazardSnapshot.
    # node_state()'s "absent means no reading" convention.
    # =====================================================

    def observation_at(self, node_id) -> OccupancyObservation:

        return self.observations.get(node_id, OccupancyObservation())
