from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional

from hazard.edge_state import HazardEdgeState
from hazard.node_state import HazardNodeState


@dataclass(frozen=True)
class HazardContribution:

    # One HazardSource's proposed opinion for a single evolution step --
    # deliberately not a HazardSnapshot. A snapshot is a definitive,
    # point-in-time capture (see hazard/snapshot.py); a contribution is
    # only a sparse input toward producing one, and carries no
    # timestamp/snapshot_id of its own -- those belong to the merged
    # result HazardEvolutionEngine.evolve() returns, not to any single
    # source's proposal.
    #
    # Frozen and MappingProxyType-wrapped for the same reason
    # HazardSnapshot is: a source's proposal, once returned, must not
    # be mutable out from under the engine that's about to merge it.

    node_states: Mapping[str, HazardNodeState] = field(default_factory=dict)
    edge_states: Mapping[str, HazardEdgeState] = field(default_factory=dict)

    # =====================================================

    def __post_init__(self):

        object.__setattr__(self, "node_states", MappingProxyType(dict(self.node_states)))
        object.__setattr__(self, "edge_states", MappingProxyType(dict(self.edge_states)))

    # =====================================================
    # Unlike HazardSnapshot.node_state()/edge_state() (total functions
    # that default absence to "clear"), these return None for a node/
    # edge this source has no opinion about. That distinction is the
    # whole point of this type: "no opinion" (carry the previous
    # snapshot's value forward unchanged) is not the same thing as
    # "this source asserts clear" (an explicit HazardNodeState()/
    # HazardEdgeState() with hazard_score=0.0). Only HazardEvolution
    # Engine collapses that distinction, when it builds the next
    # HazardSnapshot -- callers of a HazardSnapshot itself never see
    # "no opinion" as a state.
    # =====================================================

    def node_state(self, node_id) -> Optional[HazardNodeState]:

        return self.node_states.get(node_id)

    # =====================================================

    def edge_state(self, edge_id) -> Optional[HazardEdgeState]:

        return self.edge_states.get(edge_id)
