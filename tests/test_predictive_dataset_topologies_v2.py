import unittest

from predictive_dataset.candidate import edges_by_candidate_id, enumerate_candidates
from predictive_dataset.topologies_v2 import (
    all_topology_specs,
    build_multi_exit_wide,
    build_single_exit_lowrise,
    build_twin_stair_highrise,
    build_v1_topology_fixed,
)


# =====================================================
# Predictive Dataset V2 milestone, Phase 19 -- focused tests for the
# new topology fixtures. Deliberately NO full-campaign scenario
# generation here (Phase 19's own "do not create tests that require
# running the entire campaign") -- these check structural properties of
# the Building/candidate objects themselves, which is instant.
# =====================================================


class StaircaseVerticalHeightRegressionTests(unittest.TestCase):
    """Phase 1's root-cause finding, mechanically guarded: every V2
    Staircase must resolve a nonzero vertical_height/travel_distance --
    the exact bug ai_registry.training_scenario.make_training_building()
    has (Staircase(...) never sets from_floor_id, so vertical_height()
    silently returns 0.0) must never recur in any V2 topology."""

    def test_every_v2_stair_candidate_has_nonzero_walking_distance(self):

        for spec in all_topology_specs():

            edges = edges_by_candidate_id(spec.building)

            for candidate in enumerate_candidates(spec.building):

                if candidate.candidate_type != "Stair":
                    continue

                edge = edges[candidate.candidate_id]
                with self.subTest(topology=spec.name, candidate=candidate.candidate_id):
                    self.assertGreater(
                        edge.walking_distance, 0.0,
                        f"{spec.name}'s {candidate.candidate_id} has zero walking_distance -- "
                        f"almost certainly a from_floor_id/to_floor_id authoring bug (see "
                        f"docs/architecture/predictive_dataset_campaign_v2.md Section 1).",
                    )

    def test_v1_fixed_topology_stair_resolves_the_same_vertical_height_as_v1_should_have(self):

        spec = build_v1_topology_fixed()
        stair = spec.building.floors[0].stairs[0]

        self.assertEqual(stair.vertical_height(spec.building), 3.0)
        self.assertGreater(stair.travel_distance(spec.building), 0.0)


class TopologyFamilyStructureTests(unittest.TestCase):

    def test_single_exit_lowrise_has_exactly_one_exit(self):

        spec = build_single_exit_lowrise()
        exit_count = sum(len(floor.exits) for floor in spec.building.floors)
        self.assertEqual(exit_count, 1)

    def test_single_exit_lowrise_candidates_extract_without_error(self):

        spec = build_single_exit_lowrise()
        candidates = enumerate_candidates(spec.building)
        candidate_types = {c.candidate_type for c in candidates}

        self.assertEqual(candidate_types, {"Door", "Exit"})
        self.assertEqual(sum(1 for c in candidates if c.candidate_type == "Exit"), 1)

    def test_twin_stair_highrise_has_three_floors_and_two_stairs(self):

        spec = build_twin_stair_highrise()

        self.assertEqual(len(spec.building.floors), 3)

        stair_candidates = [c for c in enumerate_candidates(spec.building) if c.candidate_type == "Stair"]
        self.assertEqual(len(stair_candidates), 2)

    def test_twin_stair_highrise_stairs_are_on_separate_floors_from_ground(self):

        spec = build_twin_stair_highrise()
        edges = edges_by_candidate_id(spec.building)

        stair_candidates = [c for c in enumerate_candidates(spec.building) if c.candidate_type == "Stair"]
        floor_ids = {c.floor_id for c in stair_candidates}

        self.assertEqual(len(floor_ids), 2)  # each stair originates on its own distinct upper floor
        for candidate in stair_candidates:
            self.assertGreater(edges[candidate.candidate_id].walking_distance, 0.0)

    def test_multi_exit_wide_has_three_exits_and_four_doors(self):

        spec = build_multi_exit_wide()

        candidates = enumerate_candidates(spec.building)
        self.assertEqual(sum(1 for c in candidates if c.candidate_type == "Exit"), 3)
        self.assertEqual(sum(1 for c in candidates if c.candidate_type == "Door"), 4)

    def test_all_topology_families_have_distinct_names_and_candidate_ids(self):

        specs = all_topology_specs()
        names = [spec.name for spec in specs]
        self.assertEqual(len(names), len(set(names)))

        all_candidate_ids = []
        for spec in specs:
            all_candidate_ids.extend(c.candidate_id for c in enumerate_candidates(spec.building))

        self.assertEqual(len(all_candidate_ids), len(set(all_candidate_ids)))  # no id collisions across families

    def test_all_topology_families_generate_valid_scenarios(self):

        from scenario_generator.batch_generator import iter_batch
        from scenario_generator.request import BatchGenerationRequest

        for spec in all_topology_specs():

            request = BatchGenerationRequest(
                definition=spec.definition, definition_id=f"test-{spec.name}",
                building=spec.building, master_seed=1, count=3,
            )

            scenarios = list(iter_batch(request))
            with self.subTest(topology=spec.name):
                self.assertEqual(len(scenarios), 3)
                for scenario in scenarios:
                    self.assertTrue(scenario.metadata.scenario_id)


if __name__ == "__main__":
    unittest.main()
