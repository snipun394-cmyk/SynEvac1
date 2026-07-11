class CostModel:

    # The one interface a future Pathfinding Engine is allowed to
    # depend on for edge cost -- never Door/Stair/Exit directly. New
    # routing behavior (smoke-aware, congestion-aware, ...) arrives as
    # a new CostModel implementation, not as changes to the graph,
    # Edge, or the pathfinder itself.

    def cost(self, edge):

        raise NotImplementedError


class DefaultCostModel(CostModel):

    # V1 cost model: Edge.traversal_cost as-is (walking distance where
    # known, otherwise Edge.DEFAULT_TRAVERSAL_COST). No smoke, fire,
    # congestion, or obstacle penalties yet -- those are future
    # CostModel implementations layered on top of this one.

    def cost(self, edge):

        return edge.traversal_cost
