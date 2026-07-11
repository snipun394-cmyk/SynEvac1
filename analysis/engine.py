from navigation.edge import Edge
from navigation.node import Node

from analysis.report import (
    ConnectivityReport,
    CriticalConnectorFinding,
    ExitServiceArea,
    ExitServiceAreaReport,
    TravelDistanceReport,
)


class BuildingAnalysisEngine:

    # A read-only reporting layer over a PathfindingEngine -- never
    # over Door/Exit/Staircase/Zone directly. Every analysis below
    # reduces to one or more distances_from(Node.OUTSIDE_NODE_ID)
    # calls (single-source Dijkstra), reusing exactly the same
    # public Navigation Graph / CostModel contract PathfindingEngine
    # itself is built on -- no new graph algorithms, no duplicated
    # connectivity logic (graph.validate() is reused as-is inside
    # connectivity_report(), not reimplemented).

    def __init__(self, engine):

        self.engine = engine
        self.graph = engine.graph

    # =====================================================

    def exit_service_areas(self):

        # Partitions every ZONE by which Exit edge its shortest route
        # to Outside actually leaves through. distances_from() returns
        # routes FROM Outside TO each zone, so route.edges[0] (the
        # edge leaving Outside) is always the specific Exit used --
        # Outside connects to the graph only via Exit edges, so this
        # is exact, not a heuristic.

        routes = self.engine.distances_from(Node.OUTSIDE_NODE_ID)

        by_exit = {}
        unserved_node_ids = []

        for node in self._zone_nodes():

            route = routes.get(node.id)

            if route is None:
                unserved_node_ids.append(node.id)
                continue

            exit_edge_id = route.edges[0].id

            area = by_exit.get(exit_edge_id)

            if area is None:

                area = ExitServiceArea(
                    exit_edge_id=exit_edge_id,
                    served_node_ids=[],
                    distances={},
                )
                by_exit[exit_edge_id] = area

            area.served_node_ids.append(node.id)
            area.distances[node.id] = route.total_distance

        return ExitServiceAreaReport(
            by_exit=by_exit,
            unserved_node_ids=unserved_node_ids,
        )

    # =====================================================

    def max_travel_distance(self, threshold=None):

        # The classic life-safety "travel distance to exit" metric,
        # computed for every ZONE in one distances_from() pass. No
        # code-mandated threshold is assumed -- callers may pass one
        # (meters) to also get the list of zones exceeding it.

        routes = self.engine.distances_from(Node.OUTSIDE_NODE_ID)

        distances = {}
        unreachable_node_ids = []
        exceeding_threshold_node_ids = []

        for node in self._zone_nodes():

            route = routes.get(node.id)

            if route is None:
                unreachable_node_ids.append(node.id)
                continue

            distances[node.id] = route.total_distance

            if (
                threshold is not None
                and route.total_distance is not None
                and route.total_distance > threshold
            ):
                exceeding_threshold_node_ids.append(node.id)

        measured = {
            node_id: distance
            for node_id, distance in distances.items()
            if distance is not None
        }

        if measured:
            max_node_id = max(measured, key=measured.get)
            max_distance = measured[max_node_id]
        else:
            max_node_id = None
            max_distance = None

        return TravelDistanceReport(
            distances=distances,
            max_distance=max_distance,
            max_distance_node_id=max_node_id,
            unreachable_node_ids=unreachable_node_ids,
            exceeding_threshold_node_ids=exceeding_threshold_node_ids,
        )

    # =====================================================

    def critical_connectors(self):

        # For every edge, compare the baseline set of nodes reachable
        # from Outside against the same query with just that edge
        # excluded. Removing an edge can only shrink reachability, so
        # (baseline - without_edge) is exactly the set of nodes
        # stranded by that edge's failure. O(E) distances_from()
        # calls -- fine at building scale; Tarjan's bridge-finding
        # would be the answer if this ever needed to scale to a
        # multi-building campus graph.

        baseline = set(
            self.engine.distances_from(Node.OUTSIDE_NODE_ID).keys()
        )

        findings = []

        for edge in self.graph.edges:

            reachable_without = set(
                self.engine.distances_from(
                    Node.OUTSIDE_NODE_ID,
                    excluded_edge_ids={edge.id},
                ).keys()
            )

            stranded_node_ids = sorted(baseline - reachable_without)

            if stranded_node_ids:

                findings.append(
                    CriticalConnectorFinding(
                        edge_id=edge.id,
                        edge_type=edge.edge_type,
                        stranded_node_ids=stranded_node_ids,
                    )
                )

        findings.sort(
            key=lambda finding: len(finding.stranded_node_ids),
            reverse=True,
        )

        return findings

    # =====================================================

    def stair_dependency(self):

        # Not a separate algorithm -- exactly critical_connectors(),
        # filtered to Stair edges.

        return [
            finding
            for finding in self.critical_connectors()
            if finding.edge_type == Edge.STAIR
        ]

    # =====================================================

    def connectivity_report(self):

        findings = self.critical_connectors()

        stair_findings = [
            finding
            for finding in findings
            if finding.edge_type == Edge.STAIR
        ]

        return ConnectivityReport(
            validation=self.graph.validate(),
            travel_distance=self.max_travel_distance(),
            critical_connectors=findings,
            stair_dependencies=stair_findings,
        )

    # =====================================================

    def _zone_nodes(self):

        return [
            node
            for node in self.graph.nodes.values()
            if node.node_type == Node.ZONE
        ]
