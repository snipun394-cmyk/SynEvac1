from dataclasses import dataclass, field
from typing import Any, Tuple

from navigation.obstacle_geometry import segment_blocked_by_obstacles


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

    # Obstacle -> Navigation & Evacuation Connectivity milestone --
    # LIVE references to the same Obstacle objects Floor.obstacles
    # already owns (populated once by NavigationGraphGenerator for
    # Door/Exit edges, the two edge types with an actual line-segment
    # geometry co-located with one floor's obstacle list -- Stair's
    # cross-floor point-to-point connection is deliberately out of
    # scope, see docs/architecture/obstacle_navigation_integration.md).
    # NOT a snapshot/copy of obstacle state at build time -- exactly
    # like `reference` itself, these are the SAME live Obstacle
    # instances, so traversable (below) always reflects an obstacle's
    # CURRENT position/active/traversability with no graph rebuild
    # required, the same "no dynamic state baked into Edge, always
    # read live from the real engineering object" discipline
    # Door.locked/Exit.is_blocked already established one property
    # down. An empty tuple (the default) means "no obstacles were on
    # this edge's floor at build time" -- Stair edges, and any Door/
    # Exit edge built before this milestone's own tests supply one,
    # simply never have anything to check, never a fabricated block.
    blocking_obstacles: Tuple[Any, ...] = field(default_factory=tuple)

    # Deliberately no other dynamic-state field here (smoke, fire,
    # congestion, ...) -- same reasoning as Node: rebuilding the graph
    # replaces every Edge instance, so nothing stored on one would
    # survive the next rebuild. Smoke/Fire/Congestion penalties belong
    # to a future Simulation/Dynamic Hazard layer that keeps its own
    # store keyed by Edge.id and supplies penalised costs through a
    # CostModel (see navigation/cost.py) that wraps DefaultCostModel --
    # not through fields added to Edge.

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
                and not self._blocked_by_obstacle()
            )

        if self.edge_type == self.EXIT:

            return (
                not getattr(self.reference, "is_blocked", False)
                and not self._blocked_by_obstacle()
            )

        return True

    # =====================================================

    def _blocked_by_obstacle(self) -> bool:

        # Obstacle -> Navigation & Evacuation Connectivity milestone --
        # Door/Exit both expose the identical start_point/end_point
        # shape (models/door.py, models/exit.py), read here via
        # getattr so this stays duck-typed rather than importing
        # either model. Live geometry check against LIVE obstacle
        # references (see blocking_obstacles' own field comment above)
        # -- never cached, never requiring a graph rebuild.

        if not self.blocking_obstacles:
            return False

        seg_start = getattr(self.reference, "start_point", None)
        seg_end = getattr(self.reference, "end_point", None)

        return segment_blocked_by_obstacles(seg_start, seg_end, self.blocking_obstacles)

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
