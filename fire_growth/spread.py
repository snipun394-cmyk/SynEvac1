from typing import Optional

from navigation.edge import Edge

from hazard.node_state import HazardNodeState

from hazard_evolution.contribution import HazardContribution
from hazard_evolution.source import HazardSource

from smoke_propagation.graph_distance import shortest_graph_distances

from fire_growth.growth_curve import FireGrowthCurve, TSquaredFireGrowthCurve


class FireSpreadModel(HazardSource):

    # Companion to FireGrowthModel: that class evolves hazard_score at
    # exactly the ignition node forever (by design -- see its own
    # docstring); this class is the second HazardSource needed for
    # fire to ever leave that node, spreading outward through Door and
    # Stair connectivity exactly the way SmokePropagationModel already
    # spreads smoke -- same shortest_graph_distances() front-propagation
    # algorithm (reused, not reimplemented), same "static topology, read
    # once at construction" discipline, same HazardSource interface, so
    # HazardEvolutionEngine needed zero changes to host a second fire-
    # related source alongside FireGrowthModel.
    #
    # Differs from smoke's own propagation in exactly the one place
    # real fire behavior actually differs from smoke behavior: how much
    # each connection resists it. SmokePropagationModel's own docstring
    # is explicit that it deliberately ignores Edge.traversable/door
    # state ("smoke... seeps through/under a closed opening regardless
    # of its lock/active state") -- fire is the opposite case: a closed
    # door measurably delays fire spread into the next space, and a
    # door authored as Building.DOOR_TYPES's own "Fire Door" is
    # specifically rated to resist it far longer. _fire_edge_weight()
    # below reads exactly that already-existing data (Edge.reference is
    # always the originating Door/Staircase object -- confirmed in
    # navigation/graph_builder.py, no new Building/Door field needed)
    # to scale each edge's distance into an effective fire-spread
    # distance, all still fed through the same Dijkstra
    # shortest_graph_distances() smoke already uses, just with a
    # different edge_weight_fn.
    #
    # Only ever proposes for nodes OTHER than its own ignition_node_id
    # -- that node's own growth is FireGrowthModel's job (avoids two
    # sources both authoring the same node's hazard_score; harmless
    # either way under DefaultHazardMergeStrategy's max(), just
    # cleaner not to).

    # Real fire spread through structure/contents is far slower than
    # smoke/hot-gas migration (SmokePropagationModel's own
    # DEFAULT_FRONT_SPEED_M_PER_S=0.5) -- an arbitrary, documented
    # placeholder (not a validated design-fire value, same honesty as
    # every other constant in this module), not a measured value.
    # Always overridable via the front_speed argument.
    DEFAULT_FRONT_SPEED_M_PER_S = 0.05

    DEFAULT_GROWTH_TIME = 300.0  # seconds -- mirrors FireGrowthModel's
                                  # own default; a newly-ignited zone
                                  # grows by the same t-squared shape,
                                  # not a copy of the ignition zone's
                                  # already-elapsed growth.

    # Resistance multipliers -- each scales an edge's own
    # walking_distance into an effective fire-spread distance (a larger
    # multiplier means fire takes proportionally longer to cross that
    # edge, since arrival_time = ignition_time + weighted_distance /
    # front_speed). All four are arbitrary, disclosed engineering
    # placeholders, not validated fire-resistance-rating data -- same
    # "documented, not validated" honesty as every threshold elsewhere
    # in this codebase's hazard layer. Always overridable via the
    # constructor.
    OPEN_DOOR_RESISTANCE = 1.0
    CLOSED_DOOR_RESISTANCE = 4.0
    LOCKED_DOOR_RESISTANCE = 4.0  # "locked" is an access-control state,
                                   # not a fire rating -- treated the
                                   # same as an ordinary closed door,
                                   # never as more fire-resistant than
                                   # that just because it is locked.
    FIRE_DOOR_RESISTANCE = 25.0  # Building.DOOR_TYPES's own "Fire
                                   # Door" -- a door specifically rated
                                   # to resist fire far longer than an
                                   # ordinary one; still eventually
                                   # crossed in a long-enough incident
                                   # (real fire doors do eventually
                                   # fail), never an unconditional
                                   # block, which would be less
                                   # realistic than a graduated delay.
    STAIR_RESISTANCE = 6.0  # Vertical spread through a stair shaft --
                              # real stair enclosures are typically
                              # among a building's more fire-protected
                              # vertical paths (by code, specifically to
                              # remain viable for evacuation), so this
                              # sits well above an open door but below
                              # a dedicated Fire Door.

    # Smoke propagates through Door and Stair edges only, never Exit
    # (fire reaching the exterior "Outside" node has no meaning here)
    # -- identical restriction, identical reasoning, to
    # SmokePropagationModel.PROPAGATED_EDGE_TYPES.
    PROPAGATED_EDGE_TYPES = (Edge.DOOR, Edge.STAIR)

    def __init__(
        self,
        graph,
        ignition_node_id: str,
        ignition_time: float,
        growth_curve: Optional[FireGrowthCurve] = None,
        front_speed: float = DEFAULT_FRONT_SPEED_M_PER_S,
        open_door_resistance: float = OPEN_DOOR_RESISTANCE,
        closed_door_resistance: float = CLOSED_DOOR_RESISTANCE,
        locked_door_resistance: float = LOCKED_DOOR_RESISTANCE,
        fire_door_resistance: float = FIRE_DOOR_RESISTANCE,
        stair_resistance: float = STAIR_RESISTANCE,
    ):

        if front_speed <= 0:
            raise ValueError(
                f"FireSpreadModel.front_speed must be > 0, got {front_speed!r} -- "
                f"a non-positive speed would never let the fire front arrive anywhere."
            )

        self.ignition_node_id = ignition_node_id
        self.ignition_time = ignition_time
        self.growth_curve = growth_curve or TSquaredFireGrowthCurve(self.DEFAULT_GROWTH_TIME)
        self.front_speed = front_speed

        self._open_door_resistance = open_door_resistance
        self._closed_door_resistance = closed_door_resistance
        self._locked_door_resistance = locked_door_resistance
        self._fire_door_resistance = fire_door_resistance
        self._stair_resistance = stair_resistance

        # Static graph topology (including each edge's Door/Staircase
        # reference), read once and cached -- same discipline
        # SmokePropagationModel's own constructor already established.
        # Never re-reads `graph` again after this line.
        self._distances = shortest_graph_distances(
            graph, ignition_node_id, self.PROPAGATED_EDGE_TYPES,
            edge_weight_fn=self._fire_edge_weight,
        )

    # =====================================================

    def _fire_edge_weight(self, edge) -> float:

        base_distance = edge.walking_distance if edge.walking_distance is not None else 1.0

        if edge.edge_type == Edge.STAIR:
            return base_distance * self._stair_resistance

        # Edge.DOOR -- edge.reference is always the originating Door
        # (navigation/graph_builder.py::_add_door_edges), read
        # defensively via getattr the same way Edge's own width/
        # capacity/traversable properties already do.
        door_type = getattr(edge.reference, "door_type", "Standard")

        if door_type == "Fire Door":
            return base_distance * self._fire_door_resistance

        if getattr(edge.reference, "locked", False):
            return base_distance * self._locked_door_resistance

        if getattr(edge.reference, "normally_open", False):
            return base_distance * self._open_door_resistance

        return base_distance * self._closed_door_resistance

    # =====================================================

    def propose(self, previous_snapshot, time, dt) -> HazardContribution:

        step_end_time = time + dt

        node_states = {}

        for node_id, distance in self._distances.items():

            if node_id == self.ignition_node_id:
                continue

            arrival_time = self.ignition_time + distance / self.front_speed

            if step_end_time < arrival_time:
                continue

            elapsed_time = step_end_time - arrival_time
            hazard_score = self.growth_curve.intensity_at(elapsed_time)

            node_states[node_id] = HazardNodeState(hazard_score=hazard_score)

        return HazardContribution(node_states=node_states)
