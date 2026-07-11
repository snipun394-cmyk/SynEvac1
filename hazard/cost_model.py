from navigation.cost import CostModel

from hazard.edge_state import HazardEdgeState
from hazard.snapshot import HazardSnapshot


class HazardAwareCostModel(CostModel):

    # Wraps any existing CostModel (typically DefaultCostModel) with a
    # HazardSnapshot -- the "future Simulation/Dynamic Hazard layers"
    # extension point navigation/cost.py's own comments already
    # illustrate. PathfindingEngine only ever calls cost_model.cost
    # (edge); it never needs to know this wrapping exists.
    #
    # An edge the snapshot marks impassable (traversable is False)
    # gets HazardEdgeState.BLOCKED_COST (infinite), not a negative or
    # "removed" cost -- this keeps the >= 0 contract Dijkstra/A* depend
    # on intact. PathfindingEngine's relaxation only ever accepts a
    # strictly *smaller* tentative cost than the current best (see
    # PathfindingEngine._relax's `tentative_cost < best_cost.get(...,
    # math.inf)`), so an infinite-cost edge is never relaxed onto --
    # in effect it is treated exactly like a structurally blocked edge
    # (a locked door, a blocked exit) already is. If hazard-blocking
    # removes every path to a goal, dijkstra()/a_star() correctly
    # return None, the same "no Route" contract Route's own docstring
    # already documents for unreachable nodes -- no special case
    # needed anywhere for the hazard-only-path-is-blocked scenario.
    # A snapshot never *loosens* Edge.traversable -- it can only add a
    # restriction on top of the base model, never remove one.

    def __init__(self, base_model: CostModel, hazard_snapshot: HazardSnapshot):

        self.base_model = base_model
        self.hazard_snapshot = hazard_snapshot

    # =====================================================

    def cost(self, edge):

        state = self.hazard_snapshot.edge_state(edge.id)

        if state.traversable is False:
            return HazardEdgeState.BLOCKED_COST

        return self.base_model.cost(edge) + state.cost_penalty
