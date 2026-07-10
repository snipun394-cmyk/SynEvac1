from navigation.edge import Edge
from navigation.graph import NavigationGraph
from navigation.node import Node
from navigation.validation import ValidationReport


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

    # =====================================================
    # Edges -- Door (Zone <-> Zone, same floor)
    # =====================================================

    def _add_door_edges(self, graph, floor):

        for door in floor.doors:

            zone_a = self._resolve_zone(
                graph,
                door.zone_a_id,
                floor,
                owner=door,
                missing_code="door_missing_zone_a",
                missing_message=(
                    f"Door '{door.name}' has no Zone A "
                    f"assigned."
                ),
            )

            zone_b = self._resolve_zone(
                graph,
                door.zone_b_id,
                floor,
                owner=door,
                missing_code="door_missing_zone_b",
                missing_message=(
                    f"Door '{door.name}' has no Zone B "
                    f"assigned."
                ),
            )

            if zone_a is None or zone_b is None:
                continue

            if zone_a.id == zone_b.id:

                graph.record_issue(
                    "invalid_reference",
                    (
                        f"Door '{door.name}' has the same "
                        f"Zone assigned to both Zone A and "
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
                    from_node=zone_a.id,
                    to_node=zone_b.id,
                    reference=door,
                )
            )

    # =====================================================
    # Edges -- Exit (Zone <-> Outside)
    # =====================================================

    def _add_exit_edges(self, graph, floor):

        for exit_obj in floor.exits:

            zone = self._resolve_zone(
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

            from_zone = self._resolve_zone(
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

            to_zone = self._resolve_zone(
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

            graph.add_edge(
                Edge(
                    id=stair.id,
                    edge_type=Edge.STAIR,
                    from_node=from_zone.id,
                    to_node=to_zone.id,
                    reference=stair,
                )
            )

    # =====================================================
    # Shared zone-reference resolution
    #
    # Every explicit zone reference (Door.zone_a_id/zone_b_id,
    # Exit.zone_id, Staircase.from_zone_id/to_zone_id) goes through
    # this one path so "missing" vs. "points at something that isn't
    # a real Zone on the expected floor" are reported consistently.
    # =====================================================

    def _resolve_zone(
        self,
        graph,
        zone_id,
        expected_floor,
        owner,
        missing_code,
        missing_message,
        skip_floor_check=False,
    ):

        if not zone_id:

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

        node = graph.find_node(zone_id)

        if node is None or node.node_type != Node.ZONE:

            graph.record_issue(
                "invalid_reference",
                (
                    f"'{owner.name}' references a Zone id "
                    f"that doesn't match any Zone in this "
                    f"Building."
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
                    f"'{owner.name}' references a Zone "
                    f"that belongs to a different floor "
                    f"than expected."
                ),
                severity=ValidationReport.ERROR,
                object_id=owner.id,
                floor_id=expected_floor.id,
            )

            return None

        return node
