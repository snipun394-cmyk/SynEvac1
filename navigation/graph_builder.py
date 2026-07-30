import math

from models.connectable_space import SPACE_TYPE_LABELS

from navigation.edge import Edge
from navigation.graph import NavigationGraph
from navigation.node import Node
from navigation.validation import ValidationReport

# What a Door may connect. Exit/Stair stay Zone-only (their own
# _resolve_endpoint() calls below don't pass allowed_types, so they
# keep the default) -- this is strictly additive for Door and changes
# nothing about how Zone <-> Zone doors already resolved.
DOOR_ENDPOINT_TYPES = (
    Node.ZONE,
    Node.ASSEMBLY_POINT,
)


class NavigationGraphGenerator:

    # Builds a NavigationGraph from a Building -- a derived, read-only
    # view over the Designer model. Never mutates the Building, never
    # copies engineering-object data onto Nodes/Edges (only holds a
    # `reference` to the original object), and never infers
    # connectivity from geometry: only explicit ids already stored on
    # Door/Exit/Staircase (zone_a_id/zone_b_id, zone_id,
    # from_zone_id/to_zone_id/to_floor_id) ever produce an edge.
    #
    # Every problem found while walking the Building is recorded via
    # graph.record_issue() and building continues regardless -- a
    # missing/invalid reference just means that one edge (or node) is
    # skipped, not that build() stops early. graph.validate() surfaces
    # all of it afterwards, together with a few whole-graph checks
    # (isolated zones, disconnected floors) that can only be answered
    # once the full graph exists.

    def build(self, building):

        graph = NavigationGraph()

        if building is None:
            return graph

        floors = building.ordered_floors()

        graph.floor_ids = [floor.id for floor in floors]

        graph.floor_names = {
            floor.id: floor.name
            for floor in floors
        }

        graph.add_node(
            Node(
                id=Node.OUTSIDE_NODE_ID,
                name="Outside",
                floor_id="",
                node_type=Node.OUTSIDE,
                reference=None,
            )
        )

        for floor in floors:
            self._add_zone_nodes(graph, floor)

        for floor in floors:
            self._add_assembly_point_nodes(graph, floor)

        for floor in floors:
            self._add_door_edges(graph, floor)

        for floor in floors:
            self._add_exit_edges(graph, floor)

        for floor in floors:
            self._add_stair_edges(graph, building, floor)

        return graph

    # =====================================================
    # Nodes
    # =====================================================

    def _add_zone_nodes(self, graph, floor):

        for zone in floor.zones:

            graph.add_node(
                Node(
                    id=zone.id,
                    name=zone.name,
                    floor_id=floor.id,
                    node_type=Node.ZONE,
                    reference=zone,
                )
            )

    def _add_assembly_point_nodes(self, graph, floor):

        for assembly_point in floor.assembly_points:

            graph.add_node(
                Node(
                    id=assembly_point.id,
                    name=assembly_point.name,
                    floor_id=floor.id,
                    node_type=Node.ASSEMBLY_POINT,
                    reference=assembly_point,
                )
            )

    # =====================================================
    # Edges -- Door (any connectable space <-> any connectable
    # space, same floor -- Zone and Assembly Point today)
    # =====================================================

    def _add_door_edges(self, graph, floor):

        for door in floor.doors:

            endpoint_a = self._resolve_endpoint(
                graph,
                door.zone_a_id,
                floor,
                owner=door,
                missing_code="door_missing_zone_a",
                missing_message=(
                    f"Door '{door.name}' has no Zone A "
                    f"assigned."
                ),
                allowed_types=DOOR_ENDPOINT_TYPES,
            )

            endpoint_b = self._resolve_endpoint(
                graph,
                door.zone_b_id,
                floor,
                owner=door,
                missing_code="door_missing_zone_b",
                missing_message=(
                    f"Door '{door.name}' has no Zone B "
                    f"assigned."
                ),
                allowed_types=DOOR_ENDPOINT_TYPES,
            )

            if endpoint_a is None or endpoint_b is None:
                continue

            if endpoint_a.id == endpoint_b.id:

                graph.record_issue(
                    "invalid_reference",
                    (
                        f"Door '{door.name}' has the same "
                        f"space assigned to both Zone A and "
                        f"Zone B."
                    ),
                    severity=ValidationReport.ERROR,
                    object_id=door.id,
                    floor_id=floor.id,
                )

                continue

            graph.add_edge(
                Edge(
                    id=door.id,
                    edge_type=Edge.DOOR,
                    from_node=endpoint_a.id,
                    to_node=endpoint_b.id,
                    reference=door,
                    walking_distance=self._door_walking_distance(
                        endpoint_a,
                        door,
                        endpoint_b,
                    ),
                    blocking_obstacles=tuple(floor.obstacles),
                )
            )

    # =====================================================
    # Edges -- Exit (Zone <-> Outside)
    # =====================================================

    def _add_exit_edges(self, graph, floor):

        for exit_obj in floor.exits:

            zone = self._resolve_endpoint(
                graph,
                exit_obj.zone_id,
                floor,
                owner=exit_obj,
                missing_code="exit_missing_zone",
                missing_message=(
                    f"Exit '{exit_obj.name}' is not "
                    f"connected to a Zone."
                ),
            )

            if zone is None:
                continue

            graph.add_edge(
                Edge(
                    id=exit_obj.id,
                    edge_type=Edge.EXIT,
                    from_node=zone.id,
                    to_node=Node.OUTSIDE_NODE_ID,
                    reference=exit_obj,
                    walking_distance=self._exit_walking_distance(
                        zone,
                        exit_obj,
                    ),
                    blocking_obstacles=tuple(floor.obstacles),
                )
            )

    # =====================================================
    # Edges -- Stair (Zone <-> Zone, different floors)
    # =====================================================

    def _add_stair_edges(self, graph, building, floor):

        for stair in floor.stairs:

            destination_floor = None

            if not stair.to_floor_id:

                graph.record_issue(
                    "stair_missing_destination_floor",
                    (
                        f"Stair '{stair.name}' has no "
                        f"destination floor assigned."
                    ),
                    severity=ValidationReport.WARNING,
                    object_id=stair.id,
                    floor_id=floor.id,
                )

            else:

                destination_floor = building.get_floor(
                    stair.to_floor_id
                )

                if destination_floor is None:

                    graph.record_issue(
                        "invalid_reference",
                        (
                            f"Stair '{stair.name}' has a "
                            f"destination floor id that "
                            f"doesn't match any floor in "
                            f"this Building."
                        ),
                        severity=ValidationReport.ERROR,
                        object_id=stair.id,
                        floor_id=floor.id,
                    )

            from_zone = self._resolve_endpoint(
                graph,
                stair.from_zone_id,
                floor,
                owner=stair,
                missing_code="stair_missing_origin_zone",
                missing_message=(
                    f"Stair '{stair.name}' has no origin "
                    f"Zone assigned."
                ),
            )

            to_zone = self._resolve_endpoint(
                graph,
                stair.to_zone_id,
                destination_floor,
                owner=stair,
                missing_code="stair_missing_destination_zone",
                missing_message=(
                    f"Stair '{stair.name}' has no "
                    f"destination Zone assigned."
                ),
                # destination_floor may itself be missing/invalid --
                # that's already reported above, don't also blame
                # the (unset) zone for it landing on the wrong floor.
                skip_floor_check=destination_floor is None,
            )

            if (
                destination_floor is None
                or from_zone is None
                or to_zone is None
            ):
                continue

            # Stair Simulation Reliability & Multi-Floor Reachability
            # Audit milestone, Phase 17 -- permanent regression guard for
            # the historical bug: a Staircase whose `from_floor_id` does
            # not actually match the floor it is placed on (`floor` here,
            # since `from_zone` above is resolved against `floor`, not
            # against `stair.from_floor_id` -- connectivity is therefore
            # unaffected by this mismatch, only the DISTANCE computation,
            # which reads from_floor_id/to_floor_id independently via
            # Staircase.vertical_height()) silently computed
            # vertical_height()==0 -> travel_distance()==0 ->
            # Edge.walking_distance==0.0 (never None) -> Edge.
            # traversal_cost==0.0 (never DEFAULT_TRAVERSAL_COST) ->
            # MultiAgentSimulation's own `duration = distance /
            # effective_speed` computed an INSTANTANEOUS
            # (start_time == end_time) traversal, with no validation
            # signal anywhere. Both degenerate cases below are surfaced
            # via graph.record_issue() (never silent, never fatal to
            # building the graph -- this floor's Stair edge still gets
            # built, exactly like every other validation issue in this
            # module) AND, when the computed distance itself is <= 0,
            # `walking_distance` is left None rather than 0.0 -- reusing
            # Edge.walking_distance's OWN ALREADY-EXISTING "not derivable"
            # contract (its own docstring: "None means 'not derivable'"),
            # so Edge.traversal_cost/traversal_time fall back to the
            # SAME DEFAULT_TRAVERSAL_COST/None-time behavior every other
            # edge with undeterminable geometry already gets -- no new
            # concept introduced, no edge silently dropped, no change to
            # graph connectivity or routing reachability.

            if floor.id == destination_floor.id:

                graph.record_issue(
                    "stair_same_floor_both_ends",
                    (
                        f"Stair '{stair.name}' has the same floor "
                        f"('{floor.name}') resolved for both ends -- "
                        f"a Staircase must connect two distinct floors."
                    ),
                    severity=ValidationReport.WARNING,
                    object_id=stair.id,
                    floor_id=floor.id,
                )

            travel_distance = stair.travel_distance(building)

            if travel_distance is None or travel_distance <= 0:

                graph.record_issue(
                    "stair_zero_traversal_distance",
                    (
                        f"Stair '{stair.name}' connects floor "
                        f"'{floor.name}' to floor '{destination_floor.name}' "
                        f"but its computed travel distance is zero or "
                        f"invalid (commonly caused by from_floor_id not "
                        f"matching the floor this Stair is actually placed "
                        f"on) -- falling back to Edge.DEFAULT_TRAVERSAL_COST "
                        f"instead of simulating an instantaneous traversal."
                    ),
                    severity=ValidationReport.WARNING,
                    object_id=stair.id,
                    floor_id=floor.id,
                )

                walking_distance = None

            else:

                walking_distance = travel_distance

            graph.add_edge(
                Edge(
                    id=stair.id,
                    edge_type=Edge.STAIR,
                    from_node=from_zone.id,
                    to_node=to_zone.id,
                    reference=stair,
                    walking_distance=walking_distance,
                )
            )

    # =====================================================
    # Walking distance -- geometric, computed once at build time since
    # it needs both endpoints' positions (or, for Stair, the
    # Building's floor heights) rather than just one `reference`.
    # Never stored on Door/Exit/Staircase themselves.
    # =====================================================

    def _position_of(self, space):

        # Zone exposes `center`, AssemblyPoint exposes `position` --
        # either way, a single (x, y) point to measure from/to.
        if hasattr(space, "center"):
            return space.center

        if hasattr(space, "position"):
            return space.position

        return None

    def _door_walking_distance(self, endpoint_a, door, endpoint_b):

        position_a = self._position_of(endpoint_a.reference)
        position_b = self._position_of(endpoint_b.reference)

        if position_a is None or position_b is None:
            return None

        door_position = door.center

        return (
            math.dist(position_a, door_position)
            + math.dist(door_position, position_b)
        )

    def _exit_walking_distance(self, zone, exit_obj):

        zone_position = self._position_of(zone.reference)

        if zone_position is None:
            return None

        return math.dist(zone_position, exit_obj.center)

    # =====================================================
    # Shared endpoint resolution
    #
    # Every explicit space reference (Door.zone_a_id/zone_b_id,
    # Exit.zone_id, Staircase.from_zone_id/to_zone_id) goes through
    # this one path so "missing" vs. "points at something that isn't
    # a real, allowed space on the expected floor" are reported
    # consistently. `allowed_types` defaults to Zone-only, exactly
    # what every call site used before Door endpoints were
    # generalized -- Exit and Stair never pass it, so their behavior
    # (including this method's messages, which read identically to
    # before whenever allowed_types is just Zone) is unchanged.
    # =====================================================

    def _resolve_endpoint(
        self,
        graph,
        space_id,
        expected_floor,
        owner,
        missing_code,
        missing_message,
        skip_floor_check=False,
        allowed_types=(Node.ZONE,),
    ):

        if not space_id:

            graph.record_issue(
                missing_code,
                missing_message,
                severity=ValidationReport.WARNING,
                object_id=owner.id,
                floor_id=(
                    expected_floor.id
                    if expected_floor is not None
                    else ""
                ),
            )

            return None

        node = graph.find_node(space_id)

        type_names = " or ".join(
            SPACE_TYPE_LABELS.get(node_type, node_type)
            for node_type in allowed_types
        )

        if node is None or node.node_type not in allowed_types:

            graph.record_issue(
                "invalid_reference",
                (
                    f"'{owner.name}' references an id that "
                    f"doesn't match any {type_names} in "
                    f"this Building."
                ),
                severity=ValidationReport.ERROR,
                object_id=owner.id,
                floor_id=(
                    expected_floor.id
                    if expected_floor is not None
                    else ""
                ),
            )

            return None

        if (
            not skip_floor_check
            and expected_floor is not None
            and node.floor_id != expected_floor.id
        ):

            graph.record_issue(
                "invalid_reference",
                (
                    f"'{owner.name}' references a "
                    f"{SPACE_TYPE_LABELS.get(node.node_type, node.node_type)} "
                    f"that belongs to a different floor "
                    f"than expected."
                ),
                severity=ValidationReport.ERROR,
                object_id=owner.id,
                floor_id=expected_floor.id,
            )

            return None

        return node
