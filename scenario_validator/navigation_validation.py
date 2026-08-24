from navigation.edge import Edge
from navigation.graph_builder import NavigationGraphGenerator
from navigation.node import Node
from navigation.reachability import bfs_reachable

from scenario import StairAvailability

from scenario_validator.issue import FailureCategory, ScenarioValidationIssue
from scenario_validator.report import ScenarioValidationReport


# Navigation Validation -- architecture doc §5.3, module 6, the
# category this pass's brief calls "extremely important." Reuses
# NavigationGraphGenerator (topology -- which zones/doors/exits/stairs
# connect to which) from navigation/, the same machinery Sandbox/
# Building Analysis already build on. Deliberately does NOT reuse
# Edge.traversable or PathfindingEngine's search directly: both are
# hardwired to read the *Building's own live Door/Exit state*
# (Edge.traversable -> reference.locked/active/is_blocked, see
# navigation/edge.py), never a candidate Scenario's *sampled* door/
# exit/stair state. Mutating the Building's engineering objects to
# match the candidate before querying would violate this Validator's
# own "never modifies anything" invariant (§5.1) -- so this module
# builds its own traversability predicate from the candidate's
# resolved states instead, and reuses only the genuinely
# Building-state-independent parts of Navigation's public API
# (NavigationGraph.find_node()/find_neighbors(), Node.OUTSIDE_NODE_ID,
# Edge.edge_type) to walk it.
#
# The Generator must never perform these checks (§4.2/§4.13, unchanged)
# -- this module is the only place in the whole Scenario Engine that
# ever asks "does this state actually enable evacuation."
#
# Scenario Campaign Feasibility Preflight Phase 1 -- the BFS itself
# (previously a private `_bfs_reachable()` defined in this file) moved,
# unchanged, to navigation.reachability.bfs_reachable() so
# campaign_feasibility/ (a new, separate, pre-generation analysis layer)
# can reuse the identical algorithm instead of a second, independently
# written one. This module now imports it rather than defining its own
# copy; nothing about this module's own checks, order, or semantics
# changed.


def _door_traversable_map(candidate):

    # DoorState.is_traversable is the single, canonical interpretation
    # of OPEN/CLOSED/LOCKED (scenario/engineering_state.py) -- OPEN and
    # CLOSED (unlocked) both permit evacuation; only LOCKED blocks it.
    return {state.door_id: state.state.is_traversable for state in candidate.door_states}


def _exit_open_map(candidate):

    return {state.exit_id: state.is_open for state in candidate.exit_states}


def _stair_traversable_map(candidate):

    return {
        state.stair_id: state.availability == StairAvailability.AVAILABLE
        for state in candidate.stair_states
    }


def _make_traversable_predicate(candidate, include_exits=True):

    door_open = _door_traversable_map(candidate)
    exit_open = _exit_open_map(candidate)
    stair_available = _stair_traversable_map(candidate)

    def is_traversable(edge):

        if edge.edge_type == Edge.DOOR:
            return door_open.get(edge.id, True)

        if edge.edge_type == Edge.EXIT:
            return include_exits and exit_open.get(edge.id, True)

        if edge.edge_type == Edge.STAIR:
            return stair_available.get(edge.id, True)

        return True

    return is_traversable


def _occupied_zone_ids(candidate):

    return {occupant.zone_id for occupant in candidate.occupants if occupant.zone_id}


def validate_navigation(candidate, definition, building) -> ScenarioValidationReport:

    report = ScenarioValidationReport()

    if building is None:
        return report

    graph = NavigationGraphGenerator().build(building)
    occupied_zone_ids = _occupied_zone_ids(candidate)

    if not occupied_zone_ids:
        return report

    is_traversable = _make_traversable_predicate(candidate)

    # (1) At least one evacuation route exists / (2) building not
    # partitioned into unreachable regions, scoped to occupied zones
    # only -- one computation serves both bullets: since every edge is
    # bidirectional (navigation/edge.py), "reachable from Outside"
    # under the candidate's traversable edges is exactly "can reach an
    # open Exit," and restricting the *zones checked* to occupied ones
    # is what scopes it away from unoccupied, disconnected regions.
    reachable_from_outside = bfs_reachable(graph, [Node.OUTSIDE_NODE_ID], is_traversable)

    for zone_id in sorted(occupied_zone_ids):

        if zone_id not in reachable_from_outside:

            report.add(
                FailureCategory.NAVIGATION, ScenarioValidationReport.ERROR,
                "NO_EVACUATION_ROUTE",
                f"Occupied zone {zone_id!r} has no path to an open Exit under this "
                f"candidate's door/exit/stair state.",
                object_id=zone_id,
            )

    # (3) Fire origin does not create an impossible initial condition:
    # excluding the ignition zone itself from the graph, every *other*
    # occupied zone must still reach an open Exit -- catches an
    # ignition zone that is the sole path out for the rest of the
    # building, without requiring occupants literally standing in the
    # ignition zone to also "escape" through it.
    if candidate.fire is not None and candidate.fire.ignition_zone_id:

        ignition_zone_id = candidate.fire.ignition_zone_id

        reachable_excluding_fire = bfs_reachable(
            graph, [Node.OUTSIDE_NODE_ID], is_traversable,
            excluded_node_ids={ignition_zone_id},
        )

        for zone_id in sorted(occupied_zone_ids - {ignition_zone_id}):

            if zone_id not in reachable_excluding_fire:

                report.add(
                    FailureCategory.NAVIGATION, ScenarioValidationReport.ERROR,
                    "FIRE_ORIGIN_BLOCKS_EVACUATION",
                    f"Occupied zone {zone_id!r} can only reach an open Exit by "
                    f"routing through the fire's ignition zone {ignition_zone_id!r}.",
                    object_id=zone_id,
                )

    # (4) Minimum reachable egress capacity: at least min_open_exits
    # *distinct* open exits are each individually reachable from at
    # least one occupied zone -- not merely that some single exit is
    # reachable from every zone (§5.3's own worked example: a
    # min_open_exits=2 candidate with only one of its two open exits
    # actually reachable must be rejected here, even though checks (1)
    # and (2) above would each pass independently).
    indoor_traversable = _make_traversable_predicate(candidate, include_exits=False)

    reachable_zone_count = 0

    for exit_state in candidate.exit_states:

        if not exit_state.is_open:
            continue

        exit_edge = next(
            (edge for edge in graph.edges if edge.edge_type == Edge.EXIT and edge.id == exit_state.exit_id),
            None,
        )

        if exit_edge is None:
            continue

        exit_zone_id = exit_edge.from_node

        reachable_from_exit = bfs_reachable(graph, [exit_zone_id], indoor_traversable)

        if reachable_from_exit & occupied_zone_ids:
            reachable_zone_count += 1

    min_open_exits = definition.engineering.min_open_exits

    if reachable_zone_count < min_open_exits:

        report.add(
            FailureCategory.NAVIGATION, ScenarioValidationReport.ERROR,
            "INSUFFICIENT_REACHABLE_EGRESS",
            f"Only {reachable_zone_count} open exit(s) are actually reachable from "
            f"an occupied zone; min_open_exits requires {min_open_exits}.",
        )

    return report
