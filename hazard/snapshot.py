from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping
from uuid import uuid4

from hazard.edge_state import HazardEdgeState
from hazard.node_state import HazardNodeState


@dataclass(frozen=True)
class HazardSnapshot:

    # An immutable, point-in-time capture of dynamic environmental
    # conditions across the graph -- entirely independent of the
    # Building Model (it holds no reference to Zone/Door/Exit/
    # Staircase, only the same node_id/edge_id strings Node.id/Edge.id
    # already expose). Conditions changing over the course of an
    # incident is expressed as a *new* HazardSnapshot at a later
    # timestamp, never as mutation of one already handed to a running
    # query -- same immutable-value-object pattern as Route and
    # BehaviorDecision.
    #
    # snapshot_id is a stable identity independent of timestamp/content
    # -- what lets future replay, visualization, live-sensor, and RL
    # workflows refer to "this exact snapshot" unambiguously (e.g. in a
    # logged dataset, two snapshots at the same timestamp from
    # different scenario branches must still be distinguishable).
    # timestamp is the simulation-time this snapshot represents,
    # consumed by a HazardProvider's snapshot_at(time) lookup.

    snapshot_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = 0.0

    node_states: Mapping[str, HazardNodeState] = field(default_factory=dict)
    edge_states: Mapping[str, HazardEdgeState] = field(default_factory=dict)

    # =====================================================

    def __post_init__(self):

        # Wrapped in MappingProxyType so the snapshot stays immutable
        # even though dict/dataclass equality still works -- same
        # technique BehaviorDecision uses for its `metadata` field.
        object.__setattr__(self, "node_states", MappingProxyType(dict(self.node_states)))
        object.__setattr__(self, "edge_states", MappingProxyType(dict(self.edge_states)))

    # =====================================================
    # Total accessors -- always return a usable state, never raise and
    # never return None, mirroring DefaultCostModel/DefaultCapacityModel's
    # "every edge always has *some* usable value" convention. A node/
    # edge absent from this snapshot simply means "no hazard reported
    # there" (HazardNodeState()/HazardEdgeState() defaults: clear, fully
    # traversable, unmodified capacity/cost).
    # =====================================================

    def node_state(self, node_id) -> HazardNodeState:

        return self.node_states.get(node_id, HazardNodeState())

    # =====================================================

    def edge_state(self, edge_id) -> HazardEdgeState:

        return self.edge_states.get(edge_id, HazardEdgeState())
