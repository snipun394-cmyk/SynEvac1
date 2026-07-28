import unittest

from predictive_dataset.candidate import edges_by_candidate_id, enumerate_candidates
from predictive_dataset.topologies_v3 import all_structural_variants_v3
from predictive_dataset.topologies_v4 import (
    ALL_FAMILIES_V4,
    CANONICAL_FAMILIES_V4,
    all_structural_variants_v4,
)
from predictive_dataset.topology_diversity_v3 import structural_diversity_report


# =====================================================
# Predictive Dataset V4 milestone, Phase 30 -- focused tests for the 8
# new structural-variant fixtures (multi_wing x4, ring_corridor x4) and
# the combined 24-variant registry. No full-campaign scenario
# generation here (instant, structural-only checks), matching
# tests/test_predictive_dataset_topologies_v3.py's own discipline.
# =====================================================


class StructuralVariantRegistryV4Tests(unittest.TestCase):

    def test_exactly_twenty_four_variants(self):
        self.assertEqual(len(all_structural_variants_v4()), 24)

    def test_v3_sixteen_variants_reused_unmodified(self):
        """The old 16 variants must appear FIRST, structurally identical
        to calling all_structural_variants_v3() directly -- V4 extends,
        never re-derives or duplicates, V3's own registry. Compared by
        (family, variant_id, structural signature) rather than raw
        dataclass equality, since each call constructs fresh Building/
        Zone/etc. objects with their own creation timestamps -- a
        harmless identity difference, not a structural one."""

        from predictive_dataset.topology_signature import compute_structural_signature

        v4_variants = all_structural_variants_v4()[:16]
        v3_variants = all_structural_variants_v3()

        for v4_variant, v3_variant in zip(v4_variants, v3_variants):
            with self.subTest(variant=v3_variant.variant_id):
                self.assertEqual(v4_variant.family, v3_variant.family)
                self.assertEqual(v4_variant.variant_id, v3_variant.variant_id)
                sig_v4 = compute_structural_signature(v4_variant.family, v4_variant.variant_id, v4_variant.topology.building)
                sig_v3 = compute_structural_signature(v3_variant.family, v3_variant.variant_id, v3_variant.topology.building)
                self.assertEqual(sig_v4.structural_key(), sig_v3.structural_key())

    def test_two_new_families_with_four_variants_each(self):

        self.assertEqual(CANONICAL_FAMILIES_V4, ("multi_wing", "ring_corridor"))

        variants = all_structural_variants_v4()
        for family in CANONICAL_FAMILIES_V4:
            with self.subTest(family=family):
                count = sum(1 for v in variants if v.family == family)
                self.assertEqual(count, 4)

    def test_six_families_total(self):
        self.assertEqual(len(ALL_FAMILIES_V4), 6)

    def test_variant_ids_are_unique(self):

        variant_ids = [v.variant_id for v in all_structural_variants_v4()]
        self.assertEqual(len(variant_ids), len(set(variant_ids)))

    def test_topology_spec_name_matches_variant_id(self):

        for v in all_structural_variants_v4():
            with self.subTest(variant=v.variant_id):
                self.assertEqual(v.topology.name, v.variant_id)


class StructuralDiversityV4Tests(unittest.TestCase):

    def test_all_twenty_four_variants_are_genuinely_distinct(self):
        """The same mechanical diversity gate V3's own Phase 4 required
        -- extended to all 24 (16 reused + 8 new), not just the new 8."""

        report = structural_diversity_report(all_structural_variants_v4())

        self.assertEqual(report["requested_variant_count"], 24)
        self.assertEqual(report["distinct_structural_signature_count"], 24)
        self.assertEqual(report["duplicate_structural_signature_groups"], [])
        self.assertTrue(report["all_genuinely_distinct"])

    def test_new_variants_are_distinct_from_every_v3_variant(self):
        """A narrower, more targeted check: even if V3's 16 were somehow
        NOT included, the 8 new variants alone must still be pairwise
        distinct from each other AND from all 16 old ones."""

        v3_only = structural_diversity_report(all_structural_variants_v3())
        combined = structural_diversity_report(all_structural_variants_v4())

        self.assertEqual(v3_only["distinct_structural_signature_count"], 16)
        self.assertEqual(combined["distinct_structural_signature_count"], 24)


class StairVerticalHeightRegressionV4Tests(unittest.TestCase):
    """The V1-bug-fix discipline (every Staircase must set BOTH
    from_floor_id and to_floor_id) must hold for multi_wing_vertical,
    the one new variant with a Stair."""

    def test_every_stair_candidate_has_nonzero_walking_distance(self):

        for v in all_structural_variants_v4():

            edges = edges_by_candidate_id(v.topology.building)

            for candidate in enumerate_candidates(v.topology.building):

                if candidate.candidate_type != "Stair":
                    continue

                edge = edges[candidate.candidate_id]
                with self.subTest(variant=v.variant_id, candidate=candidate.candidate_id):
                    self.assertIsNotNone(edge.walking_distance)
                    self.assertGreater(edge.walking_distance, 0.0)


class GenuineZoneCycleCoverageTests(unittest.TestCase):
    """Mechanically confirms the specific structural gap ring_corridor
    was built to close: a genuine zone-to-zone cycle NOT mediated by
    routing out through OUTSIDE and back."""

    def test_every_ring_corridor_variant_has_a_genuine_zone_only_cycle(self):

        import networkx as nx
        from navigation.node import Node
        from predictive_dataset.graph_context_v4 import _build_undirected_graph

        outside = Node.OUTSIDE_NODE_ID

        for v in all_structural_variants_v4():

            if v.family != "ring_corridor":
                continue

            graph, _ = _build_undirected_graph(v.topology.building)
            zone_only = graph.copy()
            if outside in zone_only:
                zone_only.remove_node(outside)

            n_components = nx.number_connected_components(zone_only) if zone_only.number_of_nodes() else 0
            cyclomatic = (
                zone_only.number_of_edges() - zone_only.number_of_nodes() + n_components
                if zone_only.number_of_nodes() else 0
            )

            with self.subTest(variant=v.variant_id):
                self.assertGreaterEqual(cyclomatic, 1)


if __name__ == "__main__":
    unittest.main()
