from typing import List

from hazard.edge_state import HazardEdgeState
from hazard.node_state import HazardNodeState


class HazardMergeStrategy:

    # The one interface HazardEvolutionEngine depends on for resolving
    # *concurrent* sources that both have an opinion about the same
    # node/edge in the same step -- mirrors CostModel/CapacityModel's
    # role as the single seam a policy plugs into. The engine embeds no
    # merge policy of its own; it only ever calls through this
    # interface, so swapping strategies changes conflict-resolution
    # behavior with zero changes to HazardEvolutionEngine itself.
    #
    # Each list passed in is already guaranteed non-empty and contains
    # only sources that actually proposed something for this node/edge
    # (HazardContribution's None-means-no-opinion entries are filtered
    # out by the engine before calling this). Implementations must be
    # deterministic and side-effect free -- called once per touched id
    # per step, order of the input list is not guaranteed meaningful
    # (concurrent sources are conceptually unordered), and the same
    # inputs must always produce the same output.

    def merge_node_states(self, states: List[HazardNodeState]) -> HazardNodeState:

        raise NotImplementedError

    # =====================================================

    def merge_edge_states(self, states: List[HazardEdgeState]) -> HazardEdgeState:

        raise NotImplementedError


class DefaultHazardMergeStrategy(HazardMergeStrategy):

    # V1's whole merge policy: worst-case-wins. A conservative,
    # life-safety-appropriate default -- not an architectural
    # requirement -- chosen because it is commutative and order-
    # independent (max/min/OR/sum all satisfy merge(a, b) == merge(b,
    # a) and generalize to any number of inputs the same way), so no
    # HazardSource ever needs to know about another source or about
    # the order proposals happen to be collected in. Future strategies
    # (confidence-weighted fusion, sensor-priority fusion, probabilistic
    # fusion, ML-based fusion) plug in as separate HazardMergeStrategy
    # implementations without touching this one or the engine.

    def merge_node_states(self, states: List[HazardNodeState]) -> HazardNodeState:

        visibilities = [s.visibility for s in states if s.visibility is not None]
        temperatures = [s.temperature for s in states if s.temperature is not None]
        smoke_levels = [s.smoke_level for s in states if s.smoke_level is not None]

        return HazardNodeState(
            smoke_level=max(smoke_levels) if smoke_levels else None,
            visibility=min(visibilities) if visibilities else None,
            temperature=max(temperatures) if temperatures else None,
            hazard_score=max(s.hazard_score for s in states),
        )

    # =====================================================

    def merge_edge_states(self, states: List[HazardEdgeState]) -> HazardEdgeState:

        traversable_opinions = [s.traversable for s in states if s.traversable is not None]

        if False in traversable_opinions:
            traversable = False
        elif True in traversable_opinions:
            traversable = True
        else:
            traversable = None

        return HazardEdgeState(
            traversable=traversable,
            capacity_modifier=min(s.capacity_modifier for s in states),
            cost_penalty=sum(s.cost_penalty for s in states),
        )
