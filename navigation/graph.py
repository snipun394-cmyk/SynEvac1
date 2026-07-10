from navigation.edge import Edge
from navigation.node import Node
from navigation.validation import ValidationReport


class NavigationGraph:

    # A NavigationGraph is a derived, in-memory view over a Building --
    # it never owns or duplicates engineering objects, only references
    # them (see Node.reference/Edge.reference). The Designer project
    # remains the single source of truth; rebuilding the graph from the
    # same Building always produces the same graph.

    def __init__(self):

        self.nodes = {}
        self.edges = []

        # All floor ids the Building actually has, in Designer order --
        # set once by NavigationGraphGenerator.build(). Kept separate
        # from `nodes` because a floor with zero Zones on it still
        # needs to be seen by validate() (e.g. for "disconnected
        # floor"), even though it will never contribute a Node.
        self.floor_ids = []

        # floor_id -> Floor.name, for readable validation messages
        # only -- never consulted for graph logic itself.
        self.floor_names = {}

        # Issues recorded while the graph was being built (duplicate
        # ids caught by add_node()/add_edge(), and missing/invalid
        # references caught by NavigationGraphGenerator while walking
        # the Building). validate() adds its own structural/topology
        # checks on top of these every time it is called.
        self._build_issues = []

    # =====================================================
    # Construction
    # =====================================================

    def add_node(self, node):

        if node.id in self.nodes:

            self._build_issues.append(
                (
                    "duplicate_node_id",
                    ValidationReport.ERROR,
                    (
                        f"Duplicate node id '{node.id}' "
                        f"('{node.name}' collides with an "
                        f"existing node) -- keeping the first "
                        f"one, this node was ignored."
                    ),
                    node.id,
                    node.floor_id,
                )
            )

            return

        self.nodes[node.id] = node

    # =====================================================

    def add_edge(self, edge):

        if any(
            existing.id == edge.id
            for existing in self.edges
        ):

            self._build_issues.append(
                (
                    "duplicate_edge_id",
                    ValidationReport.ERROR,
                    (
                        f"Duplicate edge id '{edge.id}' "
                        f"({edge.edge_type}) -- an edge with "
                        f"this id already exists."
                    ),
                    edge.id,
                    "",
                )
            )

        self.edges.append(edge)

    # =====================================================

    def record_issue(
        self,
        code,
        message,
        severity=ValidationReport.WARNING,
        object_id="",
        floor_id="",
    ):

        # Used by NavigationGraphGenerator while walking the Building,
        # for problems that stop an edge/node from being created in
        # the first place (e.g. a Door with no Zone A) -- there is no
        # node or edge object yet at that point for add_node()/
        # add_edge() to reject.

        self._build_issues.append(
            (
                code,
                severity,
                message,
                object_id,
                floor_id,
            )
        )

    # =====================================================
    # Lookups
    # =====================================================

    def find_node(self, node_id):

        return self.nodes.get(node_id)

    # =====================================================

    def find_neighbors(self, node):

        node_id = (
            node.id
            if isinstance(node, Node)
            else node
        )

        neighbors = []

        for edge in self.edges:

            if edge.from_node == node_id:
                other_id = edge.to_node

            elif edge.to_node == node_id:
                other_id = edge.from_node

            else:
                continue

            other_node = self.nodes.get(other_id)

            if other_node is not None:

                neighbors.append(
                    (
                        other_node,
                        edge,
                    )
                )

        return neighbors

    # =====================================================
    # Validation
    # =====================================================

    def validate(self):

        report = ValidationReport()

        for (
            code,
            severity,
            message,
            object_id,
            floor_id,
        ) in self._build_issues:

            report.add(
                code,
                message,
                severity=severity,
                object_id=object_id,
                floor_id=floor_id,
            )

        self._validate_zone_connectivity(report)
        self._validate_floor_connectivity(report)

        return report

    # =====================================================

    def _validate_zone_connectivity(self, report):

        zone_nodes = [
            node
            for node in self.nodes.values()
            if node.node_type == Node.ZONE
        ]

        degree = {
            node.id: 0
            for node in zone_nodes
        }

        outside_ids = {
            node.id
            for node in self.nodes.values()
            if node.node_type == Node.OUTSIDE
        }

        adjacency = {
            node.id: set()
            for node in self.nodes.values()
        }

        for edge in self.edges:

            if (
                edge.from_node in adjacency
                and edge.to_node in adjacency
            ):

                adjacency[edge.from_node].add(edge.to_node)
                adjacency[edge.to_node].add(edge.from_node)

            if edge.from_node in degree:
                degree[edge.from_node] += 1

            if edge.to_node in degree:
                degree[edge.to_node] += 1

        # "Zones without any connections" -- a Zone touched by no
        # edge at all. The simplest, purely local check.
        for node in zone_nodes:

            if degree[node.id] == 0:

                report.add(
                    "zone_without_connections",
                    (
                        f"Zone '{node.name}' has no Door, Exit, "
                        f"or Stair connected to it."
                    ),
                    severity=ValidationReport.WARNING,
                    object_id=node.id,
                    floor_id=node.floor_id,
                )

        # "Isolated Zones" -- reachable from nothing that leads
        # Outside, even if they do have local connections (e.g. a
        # closed loop of Zones/Doors that never reaches an Exit).
        # Zones with zero connections are already reported above and
        # are trivially unreachable too, so they are not repeated here.
        reachable = set()
        frontier = list(outside_ids)

        while frontier:

            current = frontier.pop()

            if current in reachable:
                continue

            reachable.add(current)

            for neighbor in adjacency.get(current, ()):

                if neighbor not in reachable:
                    frontier.append(neighbor)

        for node in zone_nodes:

            if (
                degree[node.id] > 0
                and node.id not in reachable
            ):

                report.add(
                    "isolated_zone",
                    (
                        f"Zone '{node.name}' has no path to "
                        f"Outside through the current Door/Exit/"
                        f"Stair connections."
                    ),
                    severity=ValidationReport.WARNING,
                    object_id=node.id,
                    floor_id=node.floor_id,
                )

    # =====================================================

    def _validate_floor_connectivity(self, report):

        if len(self.floor_ids) <= 1:
            return

        floors_with_stairs = set()

        for edge in self.edges:

            if edge.edge_type != Edge.STAIR:
                continue

            from_node = self.nodes.get(edge.from_node)
            to_node = self.nodes.get(edge.to_node)

            if from_node is not None:
                floors_with_stairs.add(from_node.floor_id)

            if to_node is not None:
                floors_with_stairs.add(to_node.floor_id)

        for floor_id in self.floor_ids:

            if floor_id not in floors_with_stairs:

                floor_name = self.floor_names.get(
                    floor_id,
                    floor_id,
                )

                report.add(
                    "disconnected_floor",
                    (
                        f"Floor '{floor_name}' has no Stair "
                        f"connecting it to any other floor."
                    ),
                    severity=ValidationReport.WARNING,
                    object_id="",
                    floor_id=floor_id,
                )
