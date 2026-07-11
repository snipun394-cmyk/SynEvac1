class CostModel:

    # The one interface a future Pathfinding Engine is allowed to
    # depend on for edge cost -- never Door/Stair/Exit directly. New
    # routing behavior (smoke-aware, congestion-aware, ...) arrives as
    # a new CostModel implementation, not as changes to the graph,
    # Edge, or the pathfinder itself.

    def cost(self, edge):

        raise NotImplementedError


class DefaultCostModel(CostModel):

    # V1 cost model: Edge.traversal_cost as-is (straight-line walking
    # distance where derivable -- see NavigationGraphGenerator's
    # _door_walking_distance/_exit_walking_distance and
    # Staircase.travel_distance -- otherwise Edge.DEFAULT_TRAVERSAL_
    # COST). No smoke, fire, congestion, or obstacle penalties yet.

    def cost(self, edge):

        return edge.traversal_cost


# =====================================================
# Extension point for future Simulation / Dynamic Hazard layers
#
# Node and Edge deliberately carry no blocked/smoke/fire/congestion
# fields (see the comments on both classes) -- the graph is rebuilt
# from the Building on every change, so nothing stored on a Node/Edge
# instance would survive that. Dynamic state instead belongs to
# whichever layer produces it, kept in its own store keyed by
# Node.id/Edge.id, independent of the graph's lifecycle.
#
# That layer plugs into routing by wrapping a CostModel rather than by
# changing Edge -- e.g. (illustrative only, not implemented here):
#
#   class HazardAwareCostModel(CostModel):
#       def __init__(self, base_model, hazard_state):
#           self.base_model = base_model      # e.g. DefaultCostModel()
#           self.hazard_state = hazard_state  # owned by the Hazard layer
#
#       def cost(self, edge):
#           penalty = self.hazard_state.penalty_for(edge.id)
#           return self.base_model.cost(edge) + penalty
#
# The Pathfinding Engine only ever calls cost_model.cost(edge); it
# never needs to know whether that model is the default one or a
# hazard-aware wrapper.
# =====================================================
