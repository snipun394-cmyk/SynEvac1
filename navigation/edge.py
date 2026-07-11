from dataclasses import dataclass
from typing import Any


@dataclass
class Edge:

    # An Edge is a thin, read-only view over the engineering object
    # that creates it (a Door, an Exit, or a Staircase) -- the
    # Navigation Graph never owns or duplicates Designer data. `id`
    # is always that object's own id, for the same reason Node.id
    # always reuses the Zone's id.
    #
    # Every V1 edge is inherently bidirectional (a Door, Exit, or
    # Stair can be walked both ways), so there is no separate
    # direction flag -- from_node/to_node just name the two ends.

    id: str
    edge_type: str
    from_node: str
    to_node: str
    reference: Any = None

    # The physical distance (meters) a person walks along this edge.
    # Unlike width/capacity below, this can't be read straight off
    # `reference` with a single getattr: a Door's distance depends on
    # both endpoints' positions plus the door's own position, and a
    # Stair's depends on the Building's floor heights. Only
    # NavigationGraphGenerator has all of that in hand, so it computes
    # this once at build time and hands it in -- still fully derived
    # (never hand-entered, always recomputed on every rebuild), just
    # not computable from `reference` alone the way a property could.
    # None means "not derivable" (e.g. an endpoint has no geometry).
    walking_distance: float = None

    # Deliberately no dynamic-state field here (blocked, smoke, fire,
    # congestion, ...) -- same reasoning as Node: rebuilding the graph
    # replaces every Edge instance, so nothing stored on one would
    # survive the next rebuild. Smoke/Fire/Congestion/Obstacle
    # penalties belong to a future Simulation/Dynamic Hazard layer
    # that keeps its own store keyed by Edge.id and supplies penalised
    # costs through a CostModel (see navigation/cost.py) that wraps
    # DefaultCostModel -- not through fields added to Edge.

    DOOR = "Door"
    EXIT = "Exit"
    STAIR = "Stair"

    EDGE_TYPES = (
        DOOR,
        EXIT,
        STAIR,
    )

    # Used by traversal_cost whenever walking_distance isn't
    # derivable, so every edge always has *some* usable cost.
    DEFAULT_TRAVERSAL_COST = 1.0

    # A rough, static evacuation walking speed used only to turn a
    # known distance into an estimated time -- not a simulation input,
    # just a fixed assumption until a real speed model exists.
    ASSUMED_WALK_SPEED_M_PER_S = 1.2

    # =====================================================
    # Engineering properties -- all derived from `reference` (or, for
    # traversal_cost/traversal_time, from the fields above), never
    # stored redundantly. None means "not applicable/not derivable".
    # =====================================================

    @property
    def width(self):

        # Door, Exit, and Staircase all already expose their own
        # `width` -- reused as-is.
        return getattr(self.reference, "width", None)

    # =====================================================

    @property
    def capacity(self):

        # Only Exit models a capacity today; Door/Staircase simply
        # have no such attribute, so this naturally returns None for
        # them without needing an edge_type check.
        return getattr(self.reference, "capacity", None)

    # =====================================================

    @property
    def traversable(self):

        # Whether this edge can currently be walked at all. Stair has
        # no blocking flag in V1, so it is always traversable.
        if self.edge_type == self.DOOR:

            return (
                bool(getattr(self.reference, "active", True))
                and not getattr(self.reference, "locked", False)
            )

        if self.edge_type == self.EXIT:
            return not getattr(self.reference, "is_blocked", False)

        return True

    # =====================================================

    @property
    def traversal_cost(self):

        # V1's whole cost model: known walking distance, or a flat
        # default when distance can't be derived. Smoke/fire/
        # congestion/obstacle penalties are NOT applied here -- they
        # belong in a future CostModel (see navigation/cost.py) built
        # on top of this, not a change to this property.
        if self.walking_distance is not None:
            return self.walking_distance

        return self.DEFAULT_TRAVERSAL_COST

    # =====================================================

    @property
    def traversal_time(self):

        if self.walking_distance is None:
            return None

        return self.walking_distance / self.ASSUMED_WALK_SPEED_M_PER_S
