import math
import random

from uuid import uuid4

from navigation.edge import Edge
from navigation.node import Node
from navigation.graph_builder import NavigationGraphGenerator

from pathfinding.engine import PathfindingEngine

from sandbox.occupant import (
    SandboxDestinationType,
    SandboxDistribution,
    SandboxOccupant,
    SandboxOccupantState,
)


class SandboxManager:

    # The Manual Simulation Sandbox's whole runtime state -- owns the
    # list of SandboxOccupant instances for the *current* project only.
    # Deliberately holds no reference into Building/Floor's own model
    # lists (there is no Floor.occupants) and is never touched by
    # Serializer: occupants are temporary debugging objects, not part
    # of the Building Model, exactly as the milestone requires.
    #
    # Verifies Designer -> Navigation Graph -> Pathfinding -> Movement
    # end to end using only already-frozen public APIs
    # (NavigationGraphGenerator.build(), PathfindingEngine.
    # nearest_exit()/nearest_assembly_point()) -- this class implements
    # no routing or graph-construction logic of its own.

    def __init__(self):

        self.occupants = []

        self._next_number = 1

        # See ensure_routes_current() -- a snapshot of every field
        # that feeds Navigation Graph connectivity, taken the last
        # time routes were (re)computed for the whole batch. None
        # until the first ensure_routes_current() call.
        self._routes_signature = None

    # =====================================================
    # Placement
    # =====================================================

    def place_occupant(self, floor, x_m, y_m) -> "SandboxOccupant | None":

        # Mirrors every drawing tool's own "click inside a Zone"
        # contract -- a click that lands outside every Zone on this
        # floor creates nothing, silently, the same way the Zone/
        # Door/Exit tools already no-op on a locked floor rather than
        # raising.

        zone = self._find_zone(floor, x_m, y_m)

        if zone is None:
            return None

        occupant = SandboxOccupant(
            occupant_id=str(uuid4()),
            name=f"Occupant {self._next_number}",
            floor_id=floor.id,
            position=(x_m, y_m),
            origin_position=(x_m, y_m),
            origin_floor_id=floor.id,
            origin_zone_id=zone.id,
            zone_id=zone.id,
        )

        self._next_number += 1

        self.occupants.append(occupant)

        return occupant

    # =====================================================

    def _find_zone(self, floor, x_m, y_m):

        for zone in floor.zones:

            if zone.contains(x_m, y_m):
                return zone

        return None

    # =====================================================
    # Batch generation (Occupant Tool's drag-rectangle workflow) --
    # the rectangle itself is only a placement aid, exactly like the
    # single-click contract above: never stored, never part of the
    # Building Model. Every generated occupant still goes through
    # place_occupant() one at a time, so zone/floor resolution and
    # naming are never duplicated -- a point that lands outside every
    # Zone on this floor simply produces no occupant for that point,
    # the same silent no-op a single click outside a Zone already is.
    # =====================================================

    # A documented placeholder, not a validated retry policy -- caps
    # how hard Random sampling tries to fill a rectangle that only
    # partially overlaps a Zone before giving up with fewer than
    # `count` occupants, so a rectangle drawn entirely outside every
    # Zone can never hang generation.
    MAX_RANDOM_ATTEMPTS_PER_OCCUPANT = 25

    def generate_occupants(self, floor, x1_m, y1_m, x2_m, y2_m, count, distribution):

        if count <= 0:
            return []

        left, right = sorted((x1_m, x2_m))
        top, bottom = sorted((y1_m, y2_m))

        if distribution == SandboxDistribution.RANDOM:
            positions = self._random_positions(left, top, right, bottom, count)
        else:
            positions = self._uniform_positions(left, top, right, bottom, count)

        occupants = []

        for x, y in positions:

            occupant = self.place_occupant(floor, x, y)

            if occupant is not None:
                occupants.append(occupant)

            if len(occupants) >= count:
                break

        return occupants

    # =====================================================
    # Uniform -- an evenly spaced grid of cell centers over the
    # rectangle, sized to fit at least `count` cells (as square a grid
    # as possible), read off in row-major order and truncated to
    # exactly `count` candidate points. Deterministic: the same
    # rectangle/count always produces the same candidate positions.
    # =====================================================

    def _uniform_positions(self, left, top, right, bottom, count):

        columns = math.ceil(math.sqrt(count))
        rows = math.ceil(count / columns)

        width = right - left
        height = bottom - top

        cell_width = width / columns
        cell_height = height / rows

        positions = []

        for row in range(rows):

            for column in range(columns):

                if len(positions) >= count:
                    break

                positions.append(
                    (
                        left + (column + 0.5) * cell_width,
                        top + (row + 0.5) * cell_height,
                    )
                )

        return positions

    # =====================================================
    # Random -- uniform-random sampling within the rectangle, one
    # point at a time, up to MAX_RANDOM_ATTEMPTS_PER_OCCUPANT * count
    # candidates total. generate_occupants() stops as soon as it has
    # `count` actual occupants (points outside every Zone are simply
    # skipped there), so this only ever needs to hand back candidates
    # lazily rather than pre-deciding how many will succeed.
    # =====================================================

    def _random_positions(self, left, top, right, bottom, count):

        max_attempts = count * self.MAX_RANDOM_ATTEMPTS_PER_OCCUPANT

        for _ in range(max_attempts):

            yield (
                random.uniform(left, right),
                random.uniform(top, bottom),
            )

    # =====================================================

    def remove_occupant(self, occupant):

        if occupant in self.occupants:
            self.occupants.remove(occupant)

    # =====================================================

    def occupants_on_floor(self, floor_id):

        return [
            occupant
            for occupant in self.occupants
            if occupant.floor_id == floor_id
        ]

    # =====================================================

    def clear(self):

        # Called whenever the underlying project is replaced (New/
        # Open) -- occupants belong to whichever project placed them
        # and are meaningless (dangling floor_id/zone_id/route
        # references) once that project is gone.

        self.occupants = []
        self._next_number = 1
        self._routes_signature = None

    # =====================================================
    # Routing
    # =====================================================

    # Lazy route invalidation -- rather than recomputing every
    # occupant's route on every tick (wasteful) or only ever repairing
    # them as a side effect of Reset (surprising -- Reset restoring
    # origin state is a separate concern from a route having gone
    # stale), a cheap signature of every field that actually feeds
    # Navigation Graph connectivity is compared once per
    # start_simulation()/step_simulation() call. Building edits made
    # between generating occupants and pressing Start/Step (e.g.
    # finishing a Stair's From/To Zone in the Property Panel) are
    # picked up in exactly one recompute pass, before the first
    # simulation step runs -- never mid-playback.
    def _topology_signature(self, building):

        signature = []

        for floor in building.ordered_floors():

            for zone in floor.zones:
                signature.append(("zone", zone.id))

            for door in floor.doors:
                signature.append(
                    ("door", door.id, door.zone_a_id, door.zone_b_id)
                )

            for exit_obj in floor.exits:
                signature.append(("exit", exit_obj.id, exit_obj.zone_id))

            for stair in floor.stairs:
                signature.append((
                    "stair", stair.id,
                    stair.from_floor_id, stair.to_floor_id,
                    stair.from_zone_id, stair.to_zone_id,
                ))

        return tuple(signature)

    # =====================================================

    def ensure_routes_current(self, building):

        signature = self._topology_signature(building)

        if signature == self._routes_signature:
            return

        for occupant in self.occupants:

            if occupant.destination_type is not None:
                self.compute_route(occupant, building, occupant.destination_type)

        self._routes_signature = signature

    # =====================================================

    def compute_route(self, occupant, building, destination_type):

        # Rebuilds the Navigation Graph fresh from the current
        # Building every time -- the same "always derived, never
        # cached" convention NavigationGraphGenerator itself documents.
        # Cheap at Designer/debugging scale, and guarantees a route
        # always reflects the building exactly as it stands right now.

        graph = NavigationGraphGenerator().build(building)
        engine = PathfindingEngine(graph)

        if destination_type == SandboxDestinationType.EXIT:
            route = engine.nearest_exit(occupant.zone_id)

        elif destination_type == SandboxDestinationType.ASSEMBLY_POINT:
            route = engine.nearest_assembly_point(occupant.zone_id)

        else:
            route = None

        occupant.destination_type = destination_type
        occupant.route = route
        occupant.node_index = 0
        occupant.edge_progress = 0.0

        if route is None:
            occupant.state = SandboxOccupantState.UNREACHABLE

        elif not route.edges:
            occupant.state = SandboxOccupantState.ARRIVED

        else:
            occupant.state = SandboxOccupantState.ROUTED

        self._sync_position(occupant)

        return route

    # =====================================================

    def reset_occupant(self, occupant, building):

        # Restores an occupant to exactly the state it was generated
        # in -- deliberately not just "recompute the current route",
        # which would silently root the reset at wherever the
        # occupant currently stands (its live floor_id/zone_id, which
        # _advance_one_node() overwrites every time a route crosses
        # onto a new node) rather than where it actually started, for
        # any occupant Reset is pressed on after it has already
        # crossed one or more Doors/Stairs. origin_position/
        # origin_floor_id/origin_zone_id are never mutated after
        # placement, so they are always the correct values to restore.

        occupant.floor_id = occupant.origin_floor_id
        occupant.zone_id = occupant.origin_zone_id
        occupant.position = occupant.origin_position

        occupant.node_index = 0
        occupant.edge_progress = 0.0

        if occupant.destination_type is not None:

            self.compute_route(occupant, building, occupant.destination_type)

        else:

            occupant.route = None
            occupant.state = SandboxOccupantState.IDLE

    # =====================================================
    # Movement
    # =====================================================

    def step(self, occupant):

        # Step Mode -- exactly one node forward per call, regardless
        # of speed/elapsed time. Extremely useful for debugging: each
        # press is one deterministic Navigation Graph hop.

        if not self._can_advance(occupant):
            return

        self._advance_one_node(occupant)
        self._sync_position(occupant)

    # =====================================================

    def tick(self, occupant, dt):

        # Continuous playback -- advances edge_progress by
        # speed * dt / distance, crossing as many nodes as dt allows
        # in one call (never overshoots the destination).

        if not self._can_advance(occupant):
            return

        occupant.state = SandboxOccupantState.MOVING

        remaining_time = dt

        while remaining_time > 0 and occupant.state == SandboxOccupantState.MOVING:

            edge = occupant.route.edges[occupant.node_index]

            distance = (
                edge.walking_distance
                if edge.walking_distance is not None
                else Edge.DEFAULT_TRAVERSAL_COST
            )

            remaining_on_edge = distance * (1.0 - occupant.edge_progress)

            time_to_finish_edge = (
                remaining_on_edge / occupant.speed
                if occupant.speed > 0
                else float("inf")
            )

            if remaining_time >= time_to_finish_edge:

                remaining_time -= time_to_finish_edge

                self._advance_one_node(occupant)

            else:

                if distance > 0:

                    occupant.edge_progress += (
                        (occupant.speed * remaining_time) / distance
                    )

                remaining_time = 0.0

        self._sync_position(occupant)

    # =====================================================

    def _can_advance(self, occupant):

        return (
            occupant.route is not None
            and occupant.state
            not in (SandboxOccupantState.ARRIVED, SandboxOccupantState.UNREACHABLE)
        )

    # =====================================================

    def _advance_one_node(self, occupant):

        occupant.node_index += 1
        occupant.edge_progress = 0.0

        node = occupant.route.nodes[occupant.node_index]

        # Outside has no floor_id of its own (see Node.OUTSIDE_NODE_ID)
        # -- keep whichever real floor the occupant last stood on
        # rather than overwriting it with "".
        if node.floor_id:
            occupant.floor_id = node.floor_id

        occupant.zone_id = node.id if node.node_type == Node.ZONE else None

        if occupant.node_index >= len(occupant.route.edges):
            occupant.state = SandboxOccupantState.ARRIVED
        else:
            occupant.state = SandboxOccupantState.MOVING

    # =====================================================
    # Position resolution (meters) -- a Node carries no geometry of
    # its own (see navigation/node.py), so position is always derived
    # from whatever engineering object the node's `reference` points
    # at, exactly the same "derived, never duplicated" convention
    # Node's own engineering properties already follow.
    # =====================================================

    def _node_position(self, node, arriving_edge=None):

        if node.node_type == Node.ZONE:
            return node.reference.center

        if node.node_type == Node.ASSEMBLY_POINT:
            return node.reference.position

        if node.node_type == Node.OUTSIDE:

            # Outside itself is placeless -- the closest meaningful
            # position is the specific Exit doorway just used to
            # reach it.
            if arriving_edge is not None and arriving_edge.reference is not None:
                return arriving_edge.reference.center

            return None

        return None

    # =====================================================

    def _sync_position(self, occupant):

        if occupant.route is None or not occupant.route.nodes:
            return

        # The first edge starts from wherever this occupant was
        # actually placed, not from the Zone's single shared center --
        # see SandboxOccupant.origin_position for why. Every later
        # edge (node_index >= 1) starts from a real Navigation Graph
        # waypoint (a door, the next Zone, ...) this occupant has
        # actually arrived at, so _node_position() is correct there.
        if occupant.node_index == 0 and occupant.origin_position is not None:
            current_position = occupant.origin_position
        else:
            current_node = occupant.route.nodes[occupant.node_index]
            current_position = self._node_position(current_node)

        if occupant.node_index < len(occupant.route.edges):

            edge = occupant.route.edges[occupant.node_index]

            # Stair edges connect two different floors' coordinate
            # spaces -- there is no single meaningful straight-line
            # interpolation between them (the same reasoning
            # StairItem's own two-marker rendering already documents),
            # so position holds at the entrance until the edge
            # completes rather than interpolating across floors.
            if edge.edge_type != Edge.STAIR:

                next_node = occupant.route.nodes[occupant.node_index + 1]
                next_position = self._node_position(next_node, arriving_edge=edge)

                if current_position is not None and next_position is not None:

                    t = occupant.edge_progress

                    occupant.position = (
                        current_position[0] + (next_position[0] - current_position[0]) * t,
                        current_position[1] + (next_position[1] - current_position[1]) * t,
                    )

                    return

        if current_position is not None:
            occupant.position = current_position
