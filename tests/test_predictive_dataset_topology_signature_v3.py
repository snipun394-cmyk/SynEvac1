import unittest

from predictive_dataset.topologies_v2 import build_single_exit_lowrise
from predictive_dataset.topologies_v3 import all_structural_variants_v3
from predictive_dataset.topology_signature import (
    StructuralTopologySignature,
    compute_all_signatures,
    compute_structural_signature,
)


class ComputeStructuralSignatureTests(unittest.TestCase):

    def test_single_exit_lowrise_signature_matches_known_shape(self):

        spec = build_single_exit_lowrise()
        sig = compute_structural_signature("single_exit_lowrise", "single_exit_lowrise", spec.building)

        self.assertEqual(sig.floor_count, 1)
        self.assertEqual(sig.zone_count, 2)
        self.assertEqual(sig.door_count, 1)
        self.assertEqual(sig.exit_count, 1)
        self.assertEqual(sig.stair_count, 0)
        self.assertEqual(sig.candidate_count, 2)  # 1 door + 1 exit
        self.assertEqual(sig.graph_node_count, sig.zone_count + 1)
        self.assertEqual(sig.graph_edge_count, sig.candidate_count)
        self.assertGreater(sig.mean_candidate_walking_distance, 0.0)

    def test_to_dict_round_trips_all_fields(self):

        spec = build_single_exit_lowrise()
        sig = compute_structural_signature("fam", "var", spec.building)
        as_dict = sig.to_dict()

        for field_name in (
            "family", "variant_id", "floor_count", "zone_count", "door_count", "exit_count",
            "stair_count", "candidate_count", "graph_node_count", "graph_edge_count",
            "mean_candidate_walking_distance", "max_candidate_walking_distance",
            "mean_alternative_route_count", "max_alternative_route_count",
        ):
            with self.subTest(field=field_name):
                self.assertIn(field_name, as_dict)

    def test_structural_key_excludes_identity_fields(self):

        spec = build_single_exit_lowrise()
        sig_a = compute_structural_signature("family_a", "variant_a", spec.building)
        sig_b = compute_structural_signature("family_b", "variant_b", spec.building)

        # Same underlying Building -> same structural_key, even though
        # family/variant_id identity differs.
        self.assertEqual(sig_a.structural_key(), sig_b.structural_key())

    def test_structural_key_differs_for_genuinely_different_buildings(self):

        variants = all_structural_variants_v3()
        single_exit = next(v for v in variants if v.variant_id == "single_exit_lowrise")
        multi_exit = next(v for v in variants if v.variant_id == "multi_exit_wide")

        sig_a = compute_structural_signature(single_exit.family, single_exit.variant_id, single_exit.topology.building)
        sig_b = compute_structural_signature(multi_exit.family, multi_exit.variant_id, multi_exit.topology.building)

        self.assertNotEqual(sig_a.structural_key(), sig_b.structural_key())

    def test_empty_building_does_not_crash(self):
        """A defensive fixture: a Building with a Floor but no doors/
        exits/stairs/zones should degrade gracefully (mean=0.0), not
        raise a ZeroDivisionError."""

        from models.building import Building
        from models.floor import Floor

        empty_building = Building(id="empty", name="Empty", floors=[
            Floor(id="empty-floor", name="Empty", display_order=0, zones=[]),
        ])

        sig = compute_structural_signature("fam", "var", empty_building)

        self.assertEqual(sig.candidate_count, 0)
        self.assertEqual(sig.mean_candidate_walking_distance, 0.0)
        self.assertEqual(sig.mean_alternative_route_count, 0.0)


class ComputeAllSignaturesTests(unittest.TestCase):

    def test_returns_one_signature_per_variant(self):

        variants = all_structural_variants_v3()
        signatures = compute_all_signatures(variants)

        self.assertEqual(len(signatures), len(variants))
        self.assertTrue(all(isinstance(s, StructuralTopologySignature) for s in signatures))

    def test_signature_variant_ids_match_input_variant_ids(self):

        variants = all_structural_variants_v3()
        signatures = compute_all_signatures(variants)

        self.assertEqual(
            {s.variant_id for s in signatures},
            {v.variant_id for v in variants},
        )


if __name__ == "__main__":
    unittest.main()
