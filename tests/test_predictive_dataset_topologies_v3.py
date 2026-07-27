import unittest

from predictive_dataset.candidate import edges_by_candidate_id, enumerate_candidates
from predictive_dataset.topologies_v3 import (
    CANONICAL_FAMILIES,
    all_structural_variants_v3,
    with_scenario_count,
)


# =====================================================
# Predictive Dataset V3 milestone, Phase 23 -- focused tests for the 16
# structural-variant fixtures. No full-campaign scenario generation
# (instant, structural-only checks), matching
# tests/test_predictive_dataset_topologies_v2.py's own discipline.
# =====================================================


class StructuralVariantRegistryTests(unittest.TestCase):

    def test_exactly_sixteen_variants(self):

        variants = all_structural_variants_v3()
        self.assertEqual(len(variants), 16)

    def test_four_variants_per_canonical_family(self):

        variants = all_structural_variants_v3()
        self.assertEqual(len(CANONICAL_FAMILIES), 4)

        for family in CANONICAL_FAMILIES:
            with self.subTest(family=family):
                count = sum(1 for v in variants if v.family == family)
                self.assertEqual(count, 4)

    def test_every_family_is_canonical(self):

        variants = all_structural_variants_v3()
        for v in variants:
            with self.subTest(variant=v.variant_id):
                self.assertIn(v.family, CANONICAL_FAMILIES)

    def test_variant_ids_are_unique(self):

        variants = all_structural_variants_v3()
        variant_ids = [v.variant_id for v in variants]
        self.assertEqual(len(variant_ids), len(set(variant_ids)))

    def test_topology_spec_name_matches_variant_id(self):

        for v in all_structural_variants_v3():
            with self.subTest(variant=v.variant_id):
                self.assertEqual(v.topology.name, v.variant_id)


class StairVerticalHeightRegressionTests(unittest.TestCase):
    """The V1-bug-fix discipline topologies_v2.py established (every
    Staircase must set BOTH from_floor_id and to_floor_id) must hold
    for every new V3 structural variant too, including the ones with
    CHAINED stairs (twin_stair_chained_core, v1_fixed_three_floor) and
    MULTIPLE stairs converging on one zone (v1_fixed_dual_stair)."""

    def test_every_stair_candidate_has_nonzero_walking_distance(self):

        for v in all_structural_variants_v3():

            edges = edges_by_candidate_id(v.topology.building)

            for candidate in enumerate_candidates(v.topology.building):

                if candidate.candidate_type != "Stair":
                    continue

                edge = edges[candidate.candidate_id]
                with self.subTest(variant=v.variant_id, candidate=candidate.candidate_id):
                    self.assertGreater(edge.walking_distance, 0.0)

    def test_every_candidate_has_nonzero_walking_distance(self):
        """Not just stairs -- doors/exits too, across every variant."""

        for v in all_structural_variants_v3():

            edges = edges_by_candidate_id(v.topology.building)

            for candidate in enumerate_candidates(v.topology.building):

                edge = edges[candidate.candidate_id]
                with self.subTest(variant=v.variant_id, candidate=candidate.candidate_id):
                    self.assertIsNotNone(edge.walking_distance)
                    self.assertGreater(edge.walking_distance, 0.0)


class StructuralPropertiesTests(unittest.TestCase):

    def _counts(self, building):
        floor_count = len(building.floors)
        zone_count = sum(len(f.zones) for f in building.floors)
        door_count = sum(len(f.doors) for f in building.floors)
        exit_count = sum(len(f.exits) for f in building.floors)
        stair_count = sum(len(f.stairs) for f in building.floors)
        return floor_count, zone_count, door_count, exit_count, stair_count

    def test_single_exit_vertical_has_two_floors_and_one_stair(self):

        v = next(v for v in all_structural_variants_v3() if v.variant_id == "single_exit_vertical")
        floor_count, _, _, exit_count, stair_count = self._counts(v.topology.building)
        self.assertEqual(floor_count, 2)
        self.assertEqual(stair_count, 1)
        self.assertEqual(exit_count, 1)

    def test_twin_stair_chained_core_stair_lands_on_intermediate_floor(self):

        v = next(v for v in all_structural_variants_v3() if v.variant_id == "twin_stair_chained_core")
        stair3 = next(s for f in v.topology.building.floors for s in f.stairs if s.id == "tshc-stair-3")
        self.assertEqual(stair3.to_floor_id, "tshc-floor-2")
        self.assertNotEqual(stair3.to_floor_id, "tshc-floor-ground")

    def test_v1_fixed_dual_stair_has_two_stairs_converging_on_same_zone(self):

        v = next(v for v in all_structural_variants_v3() if v.variant_id == "v1_fixed_dual_stair")
        stairs = [s for f in v.topology.building.floors for s in f.stairs]
        self.assertEqual(len(stairs), 2)
        self.assertEqual({s.to_zone_id for s in stairs}, {"v1fd-zone-upper"})

    def test_multi_exit_reduced_redundancy_has_fewer_exits_than_spokes(self):

        v = next(v for v in all_structural_variants_v3() if v.variant_id == "multi_exit_reduced_redundancy")
        _, zone_count, _, exit_count, _ = self._counts(v.topology.building)
        # 5 zones (hub + 4 spokes) but only 2 exits -- 2 spokes have none.
        self.assertEqual(zone_count, 5)
        self.assertEqual(exit_count, 2)

    def test_all_sixteen_variants_have_distinct_floor_zone_door_exit_stair_tuples_or_are_documented_exceptions(self):
        """Sanity check that the registry as a whole spans a real range
        of structural shapes -- not a substitute for
        predictive_dataset.topology_diversity_v3's own mechanical
        signature-based duplicate check (tested separately)."""

        shapes = {v.variant_id: self._counts(v.topology.building) for v in all_structural_variants_v3()}
        distinct_shapes = set(shapes.values())
        # 16 variants should not collapse to fewer than, say, 10 distinct
        # (floor,zone,door,exit,stair) tuples -- a loose sanity floor,
        # NOT the authoritative diversity check (see topology_diversity_v3).
        self.assertGreaterEqual(len(distinct_shapes), 10)


class WithScenarioCountTests(unittest.TestCase):

    def test_returns_new_spec_with_overridden_count(self):

        v = all_structural_variants_v3()[0]
        original_count = v.topology.scenario_count

        overridden = with_scenario_count(v.topology, 999)

        self.assertEqual(overridden.scenario_count, 999)
        self.assertNotEqual(overridden.scenario_count, original_count)
        # everything else about the spec is unchanged
        self.assertEqual(overridden.name, v.topology.name)
        self.assertIs(overridden.building, v.topology.building)


if __name__ == "__main__":
    unittest.main()
