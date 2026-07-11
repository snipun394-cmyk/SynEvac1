from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HazardEdgeState:

    # One edge's environmental conditions at the moment a HazardSnapshot
    # was taken -- keyed externally by Edge.id (see HazardSnapshot),
    # never stored on Edge itself, same reasoning as HazardNodeState/
    # Node.
    #
    # traversable is an override, not a restatement: None means "no
    # hazard opinion, defer entirely to Edge.traversable" (the door's
    # own lock/active state); True/False means the hazard layer is
    # actively asserting the edge is passable/impassable regardless of
    # its structural state. A hazard can only ever make a structurally
    # open edge impassable, never make a locked door or blocked exit
    # passable -- HazardAwareCostModel/HazardAwareCapacityModel apply
    # this override on top of, not instead of, the base model.
    #
    # cost_penalty is additive and must stay >= 0 -- CostModel's own
    # contract ("cost(edge) must return a value >= 0", see
    # navigation/cost.py) is non-negotiable for Dijkstra/A* correctness,
    # and adding a non-negative penalty to an already non-negative base
    # cost trivially preserves it regardless of what the base cost is.
    # A *blocked* edge (traversable is False) is represented as
    # infinite effective cost -- see BLOCKED_COST and
    # HazardAwareCostModel -- never as a negative or synthetic
    # "removed" cost.
    #
    # capacity_modifier is a multiplier in [0.0, 1.0] applied to the
    # base CapacityModel's output. It only ever narrows capacity, never
    # widens it -- a hazard reducing how many people can safely be on
    # an edge at once is the only direction this model supports.

    traversable: Optional[bool] = None
    capacity_modifier: float = 1.0
    cost_penalty: float = 0.0

    # The effective cost HazardAwareCostModel uses for an edge this
    # state marks impassable -- infinite, not negative, so a blocked
    # edge is simply never chosen rather than corrupting Dijkstra/A*'s
    # non-negative-weight assumption. Centralized here so there is one
    # place that decides how "blocked" is represented numerically.
    BLOCKED_COST = float("inf")

    # =====================================================

    def __post_init__(self):

        if self.cost_penalty < 0:
            raise ValueError(
                f"HazardEdgeState.cost_penalty must be >= 0, "
                f"got {self.cost_penalty!r} -- a negative penalty would "
                f"violate CostModel's non-negative cost contract."
            )

        if not (0.0 <= self.capacity_modifier <= 1.0):
            raise ValueError(
                f"HazardEdgeState.capacity_modifier must be in [0.0, 1.0], "
                f"got {self.capacity_modifier!r} -- hazards may only "
                f"narrow capacity, never widen it."
            )
