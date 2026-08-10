import math

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from models.building import Building
from navigation.edge import Edge
from navigation.node import Node

from simulator.occupant import OccupantState

from simulation_recording.occupant_routes import OccupantRouteRecord


# =====================================================
# Occupant position interpolation for replay -- turns an
# OccupantRouteRecord (hop-by-hop node/edge ids and timing, already
# recorded, never re-simulated) plus a point in time into a single
# world (x, y) position and a live OccupantState-named state, for
# drawing a moving occupant marker. simulator.occupant.OccupantState is
# imported only for its own already-existing name vocabulary (PENDING/
# AT_NODE/QUEUED/TRAVERSING/ARRIVED/UNREACHABLE/STATIONARY) -- this
# module never constructs an Occupant or touches simulator.coordinator.
#
# Node/Edge (navigation/) carry no position of their own (see each
# module's own docstring) -- position is always resolved through
# whatever engineering object a node/edge represents. This module reads
# that position directly off Building model objects (Zone.center,
# AssemblyPoint.position, Door/Exit.center), mirroring the exact
# resolution rule sandbox/manager.py's own
# _node_position()/_sync_position() already establish (including
# holding position across a Stair hop, since a Staircase's own
# from_position/to_position live in two different, non-comparable
# floors' coordinate spaces) -- without importing sandbox.manager
# itself, which is Designer's own tightly-coupled live sandbox state,
# never intended to be reused by another package.
#
# This module never runs a simulation, builds a NavigationGraph, or
# calls PathfindingEngine -- it only reads Building geometry, exactly
# the same read-only, presentation-layer role command_center.
# incident_data's own _object_name_lookup() already plays for names.
# =====================================================


@dataclass(frozen=True)
class OccupantPosition:

    occupant_id: str

    x: Optional[float]
    y: Optional[float]
    floor_id: Optional[str]

    # The Zone id this occupant is currently in/departing from, or None
    # while at a non-Zone node (Outside, an AssemblyPoint) or when no
    # position could be resolved at all.
    zone_id: Optional[str]

    # The Staircase id this occupant is currently traversing, or None.
    current_stair_id: Optional[str]

    current_node_id: Optional[str]

    # A live, per-instant OccupantState name -- deliberately NOT
    # OccupantRouteRecord.state (that field is the run's own FINAL,
    # terminal state; a mid-hop occupant reports TRAVERSING here, never
    # the eventual ARRIVED they have not yet reached -- see
    # _live_state_for() below for exactly which OccupantState name
    # applies at each point along a record's own hops).
    state: str
    speed: Optional[float]


class BuildingPositionIndex:

    # Precomputed once per Building (not per occupant, not per frame) --
    # every Zone/AssemblyPoint id this Building's Navigation Graph could
    # ever use as a node id, mapped to its own (x, y)/floor_id, plus
    # every Door/Exit id (an edge id) mapped to its own (x, y) -- the
    # latter is what resolves the Navigation Graph's single, shared
    # "outside" node to a concrete point: the specific Exit doorway an
    # occupant actually used, exactly as sandbox.manager._node_position()
    # already does for the OUTSIDE case.

    def __init__(self, building: Building):

        self._zone_positions: Dict[str, Tuple[Tuple[float, float], str]] = {}
        self._assembly_point_positions: Dict[str, Tuple[Tuple[float, float], str]] = {}
        self._edge_positions: Dict[str, Tuple[float, float]] = {}

        for floor in building.ordered_floors():

            for zone in floor.zones:
                self._zone_positions[zone.id] = (zone.center, floor.id)

            for assembly_point in floor.assembly_points:
                self._assembly_point_positions[assembly_point.id] = (assembly_point.position, floor.id)

            for door in floor.doors:
                self._edge_positions[door.id] = door.center

            for exit_obj in floor.exits:
                self._edge_positions[exit_obj.id] = exit_obj.center

    # =====================================================

    def is_zone(self, node_id: str) -> bool:

        return node_id in self._zone_positions

    # =====================================================

    def node_position(self, node_id: str, arriving_edge_id: Optional[str] = None):

        # Returns ((x, y), floor_id) or (None, None) when this node id
        # is not resolvable at all -- never a fabricated point.

        if node_id == Node.OUTSIDE_NODE_ID:

            if arriving_edge_id is not None and arriving_edge_id in self._edge_positions:
                return self._edge_positions[arriving_edge_id], None

            return None, None

        if node_id in self._zone_positions:
            return self._zone_positions[node_id]

        if node_id in self._assembly_point_positions:
            return self._assembly_point_positions[node_id]

        return None, None

    # =====================================================

    def edge_position(self, edge_id: str) -> Optional[Tuple[float, float]]:

        # The Door/Exit's own physical center -- already populated in
        # __init__ from every Door's/Exit's own .center. Returns None
        # for any other edge id (a Stair, or an id this Building has no
        # Door/Exit for) -- never fabricated.

        return self._edge_positions.get(edge_id)


# =====================================================


def _lerp(start, end, t):

    return start + (end - start) * t


def _position_via_point(from_position, via_position, to_position, t):

    # Occupant Movement Path Fix -- interpolates through an edge's own
    # physical connection point (a Door/Exit's .center) instead of
    # directly between the two zone centroids it connects. Mirrors the
    # SAME two-segment distance split navigation/graph_builder.py's own
    # _door_walking_distance()/_exit_walking_distance() already use to
    # compute the edge's walking_distance/duration -- this only makes
    # the VISUAL position consistent with a calculation the duration
    # math already performs, never introducing new geometry or changing
    # any timing.
    #
    # via_position is None for every edge that isn't a Door/Exit (e.g.
    # BuildingPositionIndex.edge_position() returns None for a Stair
    # hop, though the caller already routes stairs through their own
    # separate, unmodified branch before ever reaching here) -- falls
    # back to the original direct from->to lerp, byte-identical to
    # this function's own pre-fix behavior.

    if via_position is None:
        return _lerp(from_position[0], to_position[0], t), _lerp(from_position[1], to_position[1], t)

    leg1 = math.dist(from_position, via_position)
    leg2 = math.dist(via_position, to_position)
    total = leg1 + leg2

    if total <= 0:
        # from == via == to (a degenerate, zero-length hop) -- every
        # candidate point coincides, so there is nothing to divide.
        return via_position

    split = leg1 / total

    if split > 0 and t < split:
        leg_t = t / split
        return _lerp(from_position[0], via_position[0], leg_t), _lerp(from_position[1], via_position[1], leg_t)

    leg_t = (t - split) / (1.0 - split) if split < 1.0 else 1.0
    return _lerp(via_position[0], to_position[0], leg_t), _lerp(via_position[1], to_position[1], leg_t)


def _at_node_position(record, node_id, position_index, state) -> OccupantPosition:

    position, floor_id = position_index.node_position(node_id) if node_id else (None, None)

    return OccupantPosition(
        occupant_id=record.occupant_id,
        x=position[0] if position else None,
        y=position[1] if position else None,
        floor_id=floor_id,
        zone_id=node_id if node_id and position_index.is_zone(node_id) else None,
        current_stair_id=None,
        current_node_id=node_id,
        state=state,
        speed=0.0,
    )


def interpolate_occupant_position(
    record: OccupantRouteRecord,
    start_zone_id: Optional[str],
    time: float,
    position_index: BuildingPositionIndex,
) -> OccupantPosition:

    # Pure function of already-recorded data: never advances, re-decides,
    # or re-simulates anything -- exactly locates where this occupant's
    # own already-fixed route places them at `time`. `start_zone_id` is
    # the occupant's own Scenario-authored starting zone (ScenarioOccupant
    # /ScenarioFirefighter.zone_id/entry_zone_id) -- the one honest
    # position source for a record with no hops at all (STATIONARY/
    # UNREACHABLE, whose OccupantTimeline.route is None -- see
    # OccupantRouteRecord's own docstring) and for the gap between
    # `depart_time` and the first hop's own start_time (pre-movement
    # delay/initial queueing, still physically at the starting zone).

    if not record.hops:

        # STATIONARY or UNREACHABLE -- the only two OccupantTimeline
        # states with route=None (see OccupantRouteRecord's own
        # docstring), both already terminal; there is no "live" state
        # to compute, record.state already is the honest, whole-run
        # answer.
        return _at_node_position(record, start_zone_id, position_index, record.state)

    if time < record.depart_time:
        return _at_node_position(record, start_zone_id, position_index, OccupantState.PENDING.name)

    hops = record.hops

    if time < hops[0].start_time:

        # Departed (pre-movement delay elapsed) but not yet admitted onto
        # the first edge -- holding at the starting zone, exactly what
        # OccupantState.AT_NODE itself documents ("about to attempt the
        # next edge").
        return _at_node_position(record, start_zone_id, position_index, OccupantState.AT_NODE.name)

    if time >= hops[-1].end_time:

        # Every hop in a completed run's own event-driven coordinator
        # (simulator.coordinator.MultiAgentSimulation.run() drains its
        # event heap fully) always ends in ARRIVED once hops exist at
        # all -- QUEUED/TRAVERSING/AT_NODE are never this record's own
        # final, persisted state.
        last_hop = hops[-1]
        return _at_node_position(
            record, last_hop.to_node_id, position_index, OccupantState.ARRIVED.name,
        )

    for hop in hops:

        if hop.start_time <= time < hop.end_time:
            return _position_within_hop(record, hop, time, position_index)

    # Between two hops (queue_wait_time on the next hop covers exactly
    # this gap -- see simulator.coordinator._handle_try_enter_edge()) --
    # holding at the most recently completed hop's own destination node.
    completed = [hop for hop in hops if hop.end_time <= time]
    holding_hop = completed[-1]

    return _at_node_position(
        record, holding_hop.to_node_id, position_index, OccupantState.AT_NODE.name,
    )


# =====================================================


def _position_within_hop(record, hop, time, position_index) -> OccupantPosition:

    duration = hop.end_time - hop.start_time
    speed = (hop.distance / duration) if (hop.distance is not None and duration > 0) else None

    is_stair = hop.edge_type == Edge.STAIR

    from_position, from_floor_id = position_index.node_position(hop.from_node_id)

    if is_stair:

        # Stair edges connect two different floors' coordinate spaces --
        # no single meaningful straight-line interpolation exists between
        # them (mirrors sandbox.manager._sync_position()'s own identical
        # rule). Position holds at the entrance until the hop completes.
        return OccupantPosition(
            occupant_id=record.occupant_id,
            x=from_position[0] if from_position else None,
            y=from_position[1] if from_position else None,
            floor_id=from_floor_id,
            zone_id=hop.from_node_id if position_index.is_zone(hop.from_node_id) else None,
            current_stair_id=hop.edge_id,
            current_node_id=hop.from_node_id,
            state=OccupantState.TRAVERSING.name,
            speed=speed,
        )

    to_position, _to_floor_id = position_index.node_position(hop.to_node_id, arriving_edge_id=hop.edge_id)

    if from_position is None or to_position is None:

        return OccupantPosition(
            occupant_id=record.occupant_id,
            x=from_position[0] if from_position else None,
            y=from_position[1] if from_position else None,
            floor_id=from_floor_id,
            zone_id=hop.from_node_id if position_index.is_zone(hop.from_node_id) else None,
            current_stair_id=None,
            current_node_id=hop.from_node_id,
            state=OccupantState.TRAVERSING.name,
            speed=speed,
        )

    t = (time - hop.start_time) / duration if duration > 0 else 1.0

    via_position = position_index.edge_position(hop.edge_id)
    x, y = _position_via_point(from_position, via_position, to_position, t)

    return OccupantPosition(
        occupant_id=record.occupant_id,
        x=x,
        y=y,
        floor_id=from_floor_id,
        zone_id=hop.from_node_id if position_index.is_zone(hop.from_node_id) else None,
        current_stair_id=None,
        current_node_id=hop.from_node_id,
        state=OccupantState.TRAVERSING.name,
        speed=speed,
    )
