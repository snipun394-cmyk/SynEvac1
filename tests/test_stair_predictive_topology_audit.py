import unittest

from navigation.edge import Edge
from navigation.graph_builder import NavigationGraphGenerator
from navigation.node import Node

from pathfinding.engine import PathfindingEngine

from predictive_dataset.topologies_v4 import all_structural_variants_v4


# =====================================================
# Stair Simulation Reliability & Multi-Floor Reachability Audit
# milestone, Phase 11/13 -- mechanically audits EVERY currently-active
# predictive-dataset structural topology (all_structural_variants_v4(),
# V3's 16 reused + V4's 8 new = 24 templates across 6 families) for:
#
#   - degenerate Stairs (walking_distance <= 0, same-floor-both-ends) --
#     the historical zero-duration bug's own mechanical detector, reused
#     as-is (navigation.graph_builder.NavigationGraphGenerator's own new
#     validation codes).
#   - whether every Zone on every floor has a structural path to SOME
#     Exit (occupancy is scenario-random per topology; this checks the
#     STRUCTURE every scenario shares, not one randomly-sampled instance).
#   - chained Stair traversal, for any topology whose floor count is 3+.
#
# Read-only audit -- builds each topology's own Building exactly as the
# predictive-dataset campaign scripts already do, never modifies
# predictive_dataset itself, never trains or touches a model.
# =====================================================


class PredictiveTopologyStairAuditTests(unittest.TestCase):

    def test_no_active_topology_has_a_degenerate_stair(self):

        variants = all_structural_variants_v4()
        self.assertGreater(len(variants), 0)

        for variant in variants:

            with self.subTest(family=variant.family, variant=variant.variant_id):

                graph = NavigationGraphGenerator().build(variant.topology.building)
                report = graph.validate()

                zero_distance_issues = report.by_code("stair_zero_traversal_distance")
                same_floor_issues = report.by_code("stair_same_floor_both_ends")

                self.assertEqual(
                    zero_distance_issues, [],
                    f"{variant.family}/{variant.variant_id} has a Stair with zero/invalid travel distance",
                )
                self.assertEqual(
                    same_floor_issues, [],
                    f"{variant.family}/{variant.variant_id} has a Stair connecting a floor to itself",
                )

    def test_every_stair_edge_has_positive_walking_distance(self):

        for variant in all_structural_variants_v4():

            with self.subTest(family=variant.family, variant=variant.variant_id):

                graph = NavigationGraphGenerator().build(variant.topology.building)
                stair_edges = [e for e in graph.edges if e.edge_type == Edge.STAIR]

                for edge in stair_edges:
                    self.assertIsNotNone(edge.walking_distance, f"{variant.variant_id}: Stair {edge.id} has walking_distance=None")
                    self.assertGreater(edge.walking_distance, 0.0, f"{variant.variant_id}: Stair {edge.id} has walking_distance <= 0")

    def test_every_zone_has_a_structural_path_to_some_exit(self):

        for variant in all_structural_variants_v4():

            with self.subTest(family=variant.family, variant=variant.variant_id):

                building = variant.topology.building
                graph = NavigationGraphGenerator().build(building)
                engine = PathfindingEngine(graph)

                zone_node_ids = [
                    node.id for node in graph.nodes.values()
                    if node.node_type == Node.ZONE
                ]

                unreachable = [
                    zone_id for zone_id in zone_node_ids
                    if engine.nearest_exit(zone_id) is None
                ]

                self.assertEqual(
                    unreachable, [],
                    f"{variant.family}/{variant.variant_id}: zones with no path to any Exit: {unreachable}",
                )

    def test_multi_floor_topologies_support_chained_stair_traversal(self):

        checked_any_multi_floor = False

        for variant in all_structural_variants_v4():

            building = variant.topology.building
            floors = building.ordered_floors()

            if len(floors) < 3:
                continue

            checked_any_multi_floor = True

            with self.subTest(family=variant.family, variant=variant.variant_id):

                graph = NavigationGraphGenerator().build(building)
                engine = PathfindingEngine(graph)

                top_floor = floors[-1]
                top_zone_nodes = [
                    node for node in graph.nodes.values()
                    if node.node_type == Node.ZONE and node.floor_id == top_floor.id
                ]

                if not top_zone_nodes:
                    continue

                route = engine.nearest_exit(top_zone_nodes[0].id)

                self.assertIsNotNone(
                    route, f"{variant.family}/{variant.variant_id}: top floor cannot reach any Exit",
                )

                stair_hops = [e for e in route.edges if e.edge_type == Edge.STAIR]

                for hop in stair_hops:
                    self.assertGreater(hop.walking_distance, 0.0)

        # If this ever becomes False (no active topology has 3+ floors),
        # that is itself worth knowing -- not a silent no-op pass.
        if not checked_any_multi_floor:
            self.skipTest("No currently-active topology has 3+ floors -- chained-Stair coverage is currently untested by this suite.")

    def test_reports_family_and_floor_summary(self):

        # Not an assertion -- a readable audit artifact for
        # docs/architecture/stair_simulation_reliability_audit.md.
        lines = []

        for variant in all_structural_variants_v4():

            building = variant.topology.building
            floors = building.ordered_floors()
            graph = NavigationGraphGenerator().build(building)
            stair_edges = [e for e in graph.edges if e.edge_type == Edge.STAIR]

            lines.append(
                f"{variant.family:24s} {variant.variant_id:34s} floors={len(floors)} "
                f"stairs={len(stair_edges)} "
                f"distances={[round(e.walking_distance, 2) for e in stair_edges]}"
            )

        print("\n[predictive topology stair audit]\n" + "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
