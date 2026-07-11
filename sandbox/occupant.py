from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
from uuid import uuid4

from navigation.edge import Edge
from pathfinding.route import Route


class SandboxOccupantState(Enum):

    # A deliberately small state machine for the Manual Simulation
    # Sandbox -- not simulator.occupant.OccupantState (that one is
    # MultiAgentSimulation's own runtime record, with capacity/
    # queueing states this tool explicitly must not implement yet).
    # This is a separate, independent type for a separate, lightweight
    # tool -- the two are never interchanged.

    IDLE = auto()          # placed, no destination chosen yet
    ROUTED = auto()        # route computed, not yet moving
    MOVING = auto()        # currently advancing along the route
    ARRIVED = auto()        # reached the last node of the route
    UNREACHABLE = auto()    # PathfindingEngine found no route


class SandboxDestinationType:

    # Exposed as a tuple for the Property Panel combo box, same
    # convention Zone.ZONE_TYPES/Detector.DETECTOR_TYPES already use.

    EXIT = "Exit"
    ASSEMBLY_POINT = "Assembly Point"

    OPTIONS = (EXIT, ASSEMBLY_POINT)


class SandboxDistribution:

    # Exposed as a tuple for the Occupant Generation dialog's combo
    # box, same convention SandboxDestinationType/Zone.ZONE_TYPES
    # already use.

    UNIFORM = "Uniform"
    RANDOM = "Random"

    OPTIONS = (UNIFORM, RANDOM)


@dataclass
class SandboxOccupant:

    # A single occupant's runtime record inside the Manual Simulation
    # Sandbox -- deliberately not saved with the project (see
    # SandboxManager), and deliberately not models/occupant.py: this
    # is a temporary debugging object, never part of the Building
    # Model, never touched by Serializer.
    #
    # position is always in meters, in whichever floor's coordinate
    # space floor_id currently names -- the same convention every
    # engineering model (Zone.x/y, Camera.position, ...) already uses.

    occupant_id: str
    name: str

    floor_id: str
    position: tuple

    # The exact (x, y) meters this occupant was actually generated/
    # placed at -- captured once, at placement, and never mutated
    # again. Used only as the interpolation source for the very first
    # edge of a route (see SandboxManager._sync_position): the Zone a
    # route starts in has a single Navigation Graph node with one
    # center point, but many occupants can be placed at many distinct
    # points within that same Zone. Interpolating the first edge from
    # the Zone's shared center rather than from here would collapse
    # every occupant in the same Zone onto one identical starting
    # position the instant a route is computed -- visually
    # indistinguishable, and any early movement imperceptible since
    # they would all move in lockstep from the same point.
    origin_position: Optional[tuple] = None

    # The floor_id/zone_id this occupant was actually generated onto --
    # captured once, at placement, and never mutated again, same
    # convention as origin_position. floor_id/zone_id below are
    # deliberately mutable (Occupant.floor_id/zone_id are overwritten
    # by SandboxManager._advance_one_node() every time a route crosses
    # onto a new node), so restoring "the initial state" -- what Reset
    # means -- requires remembering what they were before any of that
    # happened, not just resetting node_index/edge_progress and
    # recomputing a route from wherever the occupant currently is.
    origin_floor_id: Optional[str] = None
    origin_zone_id: Optional[str] = None

    zone_id: Optional[str] = None

    destination_type: Optional[str] = None
    route: Optional[Route] = None

    # Index into route.nodes of the node this occupant currently
    # occupies (has fully arrived at) -- route.edges[node_index] is
    # the edge currently being crossed towards route.nodes[node_index
    # + 1]. edge_progress is how far across that edge (0.0 just
    # departed, 1.0 fully arrived at the next node) -- 0.0 whenever
    # node_index just changed.
    node_index: int = 0
    edge_progress: float = 0.0

    speed: float = Edge.ASSUMED_WALK_SPEED_M_PER_S

    state: SandboxOccupantState = SandboxOccupantState.IDLE

    # =====================================================
    # Occupant Information (Property Panel) -- every field below is
    # derived from route/node_index/edge_progress above, never stored
    # redundantly.
    # =====================================================

    @property
    def current_node_id(self) -> Optional[str]:

        if self.route is None or not self.route.nodes:
            return self.zone_id

        return self.route.node_ids[self.node_index]

    # =====================================================

    @property
    def next_node_id(self) -> Optional[str]:

        if self.route is None:
            return None

        if self.node_index + 1 >= len(self.route.nodes):
            return None

        return self.route.node_ids[self.node_index + 1]

    # =====================================================

    @property
    def destination_node_id(self) -> Optional[str]:

        if self.route is None or not self.route.nodes:
            return None

        return self.route.node_ids[-1]

    # =====================================================

    @property
    def remaining_distance(self) -> Optional[float]:

        if self.route is None:
            return None

        remaining_edges = self.route.edges[self.node_index:]

        if not remaining_edges:
            return 0.0

        total = 0.0

        for index, edge in enumerate(remaining_edges):

            distance = (
                edge.walking_distance
                if edge.walking_distance is not None
                else Edge.DEFAULT_TRAVERSAL_COST
            )

            if index == 0:
                distance *= (1.0 - self.edge_progress)

            total += distance

        return total

    # =====================================================

    @property
    def current_speed(self) -> float:

        return self.speed if self.state == SandboxOccupantState.MOVING else 0.0
