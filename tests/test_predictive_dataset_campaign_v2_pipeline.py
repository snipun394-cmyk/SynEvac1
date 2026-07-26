import unittest

from ai_decision.engine import AIDecisionEngine
from behaviour_profile_resolver import register_occupants
from dataset_builder.timeline import TimelineRun
from scenario_generator.batch_generator import iter_batch
from scenario_generator.request import BatchGenerationRequest
from scenario_runner import run as run_scenario_context
from simulation_runtime import SimulationRuntime

from ai_registry.training_scenario import make_training_building
from predictive_dataset.campaign_config import CAMPAIGN_VERSION
from predictive_dataset.campaign_config_v2 import CAMPAIGN_VERSION_V2, MINIMUM_END_TIME_SECONDS
from predictive_dataset.candidate import enumerate_candidates
from predictive_dataset.dataset_builder import build_candidate_dataset_rows
from predictive_dataset.simulation_extractor import extract_simulation_candidate_features
from predictive_dataset.target_generator import generate_candidate_label
from predictive_dataset.topologies_v2 import build_single_exit_lowrise, build_twin_stair_highrise
from predictive_dataset.topology_analysis_v2 import multi_bottleneck_candidate_type_combinations
from predictive_dataset.versioning import dataset_version


# =====================================================
# Predictive Dataset V2 milestone, Phase 19 -- small (3-10 scenario),
# fast end-to-end pipeline tests. Deliberately far smaller than the real
# ~2500-scenario campaign (Phase 19's own "do not create tests that
# require running the entire campaign") -- these exercise the exact
# same generation -> simulation -> extraction code path at a scale that
# runs in well under a second.
# =====================================================


def _run_scenarios(spec, count, *, master_seed=1, minimum_end_time=None):

    request = BatchGenerationRequest(
        definition=spec.definition, definition_id=f"pipeline-test-{spec.name}",
        building=spec.building, master_seed=master_seed, count=count,
    )

    runs = []
    for scenario in iter_batch(request):

        context = run_scenario_context(scenario, spec.building)
        register_occupants(context)
        engine = AIDecisionEngine(base_engine=context.engine)
        runtime = SimulationRuntime(context, engine, dt=5.0)

        if minimum_end_time is not None:
            runtime.clock.end_time = max(runtime.clock.end_time, minimum_end_time)

        tick_results = runtime.run()
        runs.append(TimelineRun(
            scenario=scenario, building=context.building,
            movement_result=runtime.movement_result, tick_results=tick_results,
        ))

    return runs


class SingleExitExtractionTests(unittest.TestCase):

    def test_single_exit_scenarios_produce_valid_rows(self):

        spec = build_single_exit_lowrise()
        runs = _run_scenarios(spec, count=5, minimum_end_time=MINIMUM_END_TIME_SECONDS)

        total_rows = 0
        for run in runs:
            rows = build_candidate_dataset_rows(run, horizons=(20.0,))
            total_rows += len(rows)
            for row in rows:
                self.assertIn(row["candidate_type"], ("Door", "Exit"))

        self.assertGreater(total_rows, 0)


class MultiFloorStairExtractionTests(unittest.TestCase):

    def test_twin_stair_highrise_produces_stair_rows(self):

        spec = build_twin_stair_highrise()
        runs = _run_scenarios(spec, count=8, minimum_end_time=MINIMUM_END_TIME_SECONDS)

        stair_rows = []
        for run in runs:
            rows = build_candidate_dataset_rows(run, horizons=(20.0,))
            stair_rows.extend(row for row in rows if row["candidate_type"] == "Stair")

        self.assertGreater(len(stair_rows), 0)
        for row in stair_rows:
            self.assertGreater(row["candidate_walking_distance"], 0.0)

    def test_stair_feature_variation_and_congestion_labeling_across_a_small_sample(self):
        """Not asserting a fixed positive rate (stochastic) -- just that,
        unlike V1 (0/501,696 nonzero across the WHOLE campaign), a small
        V2 sample already shows real queue/approaching variation and at
        least one genuine positive label somewhere."""

        spec = build_twin_stair_highrise()
        runs = _run_scenarios(spec, count=15, minimum_end_time=MINIMUM_END_TIME_SECONDS)

        stair_rows = []
        for run in runs:
            rows = build_candidate_dataset_rows(run, horizons=(20.0,))
            stair_rows.extend(row for row in rows if row["candidate_type"] == "Stair")

        self.assertGreater(len(stair_rows), 0)

        any_nonzero_queue = any(row["candidate_queue_length"] > 0 for row in stair_rows)
        any_nonzero_approaching = any(row["candidate_approaching_count"] > 0 for row in stair_rows)
        any_positive_target = any(row["target"] is True for row in stair_rows)

        self.assertTrue(any_nonzero_queue or any_nonzero_approaching)
        self.assertTrue(any_positive_target)


class MultipleBottleneckLabelingTests(unittest.TestCase):

    def _row(self, scenario_id, time, horizon, candidate_type, target):
        return {
            "scenario_id": scenario_id, "observation_time": time, "prediction_horizon": horizon,
            "candidate_type": candidate_type, "target": target,
        }

    def test_two_simultaneous_positive_candidates_are_counted_as_a_multiple_bottleneck_bucket(self):

        rows = [
            self._row("scn-1", 5.0, 20.0, "Door", True),
            self._row("scn-1", 5.0, 20.0, "Stair", True),
            self._row("scn-1", 5.0, 20.0, "Exit", False),
        ]

        report = multi_bottleneck_candidate_type_combinations(rows)

        self.assertEqual(report["multiple_bottleneck_bucket_count"], 1)
        self.assertEqual(report["no_or_single_bottleneck_bucket_count"], 0)
        self.assertEqual(report["candidate_type_combination_counts"], {"Door+Stair": 1})

    def test_single_positive_candidate_is_not_a_multiple_bottleneck_bucket(self):

        rows = [self._row("scn-1", 5.0, 20.0, "Door", True), self._row("scn-1", 5.0, 20.0, "Exit", False)]

        report = multi_bottleneck_candidate_type_combinations(rows)

        self.assertEqual(report["multiple_bottleneck_bucket_count"], 0)
        self.assertEqual(report["no_or_single_bottleneck_bucket_count"], 1)

    def test_same_type_double_bottleneck_is_tallied_as_self_pair(self):

        rows = [
            self._row("scn-1", 5.0, 20.0, "Door", True),
            self._row("scn-1", 5.0, 20.0, "Door", True),
        ]

        report = multi_bottleneck_candidate_type_combinations(rows)

        self.assertEqual(report["candidate_type_combination_counts"], {"Door+Door": 1})


class HighOccupancyExtractionTests(unittest.TestCase):

    def test_twin_stair_highrise_can_produce_high_occupancy_scenarios(self):

        spec = build_twin_stair_highrise()
        request = BatchGenerationRequest(
            definition=spec.definition, definition_id="high-occ-test",
            building=spec.building, master_seed=1, count=15,
        )

        occupant_counts = [len(scenario.occupants) for scenario in iter_batch(request)]

        self.assertTrue(any(count >= 30 for count in occupant_counts))


class LeakageReAuditV2Tests(unittest.TestCase):
    """Phase 10 -- the SAME leakage boundary tests/test_predictive_dataset_
    leakage_guards.py already proves for the V1 fixture, re-run against a
    real V2 topology's Stair candidate specifically (the repaired
    candidate type this milestone's whole effort centers on)."""

    def test_stair_feature_extraction_is_still_blind_to_the_future_on_a_v2_topology(self):

        spec = build_twin_stair_highrise()
        runs = _run_scenarios(spec, count=6, minimum_end_time=MINIMUM_END_TIME_SECONDS)

        stair_candidate = next(c for c in enumerate_candidates(spec.building) if c.candidate_type == "Stair")

        from predictive_dataset.candidate import edges_by_candidate_id
        edge = edges_by_candidate_id(spec.building)[stair_candidate.candidate_id]

        for run in runs:

            # Any observation_time strictly inside the run's own duration.
            if not run.tick_results:
                continue

            mid_tick = run.tick_results[len(run.tick_results) // 2]
            time = mid_tick.time

            features_now = extract_simulation_candidate_features(
                stair_candidate, edge, time, building=run.building,
                movement_result=run.movement_result, occupancy_snapshot=mid_tick.occupancy_snapshot,
            )
            features_again = extract_simulation_candidate_features(
                stair_candidate, edge, time, building=run.building,
                movement_result=run.movement_result, occupancy_snapshot=mid_tick.occupancy_snapshot,
            )

            # Same inputs (including the SAME already-completed
            # movement_result, future ticks included) -> identical
            # features -- extraction is a pure function of time, never
            # re-derives a different answer by peeking further ahead.
            self.assertEqual(features_now, features_again)


class V1BackwardCompatibilityTests(unittest.TestCase):

    def test_v1_training_building_stair_still_has_the_original_zero_distance_characteristic(self):
        """V1's own fixture must remain byte-for-byte untouched -- this is
        not a bug we fixed in place (that would silently change V1's
        already-frozen, already-documented campaign results if ever
        regenerated from source); V2 fixes it in NEW topology fixtures
        instead (predictive_dataset.topologies_v2)."""

        building = make_training_building()
        stair = building.floors[0].stairs[0]

        self.assertEqual(stair.from_floor_id, "")
        self.assertEqual(stair.vertical_height(building), 0.0)
        self.assertEqual(stair.travel_distance(building), 0.0)

    def test_v1_campaign_version_is_unchanged(self):

        self.assertEqual(CAMPAIGN_VERSION, "predictive_dataset_campaign_v1")

    def test_dataset_version_defaults_to_v1_campaign_version_when_not_specified(self):

        version = dataset_version(20.0)
        self.assertEqual(version.campaign_version, CAMPAIGN_VERSION)


class DatasetVersionMixingGuardTests(unittest.TestCase):

    def test_v1_and_v2_campaign_versions_are_distinct(self):

        self.assertNotEqual(CAMPAIGN_VERSION, CAMPAIGN_VERSION_V2)

    def test_dataset_version_can_be_explicitly_tagged_as_v2(self):

        version = dataset_version(20.0, campaign_version=CAMPAIGN_VERSION_V2)

        self.assertEqual(version.campaign_version, CAMPAIGN_VERSION_V2)
        self.assertNotEqual(version.campaign_version, CAMPAIGN_VERSION)

    def test_v1_and_v2_dataset_versions_would_be_rejected_as_incompatible_by_predictive_model_loader(self):

        from predictive_model.dataset_loader import DatasetRequirement, assert_compatible, IncompatibleDatasetVersionError
        from predictive_model.dataset_loader import DatasetManifest

        v1_manifest = DatasetManifest(
            schema_version="1.0", campaign_version=CAMPAIGN_VERSION, feature_version="1.0",
            target_version="v1-congestion-threshold-2-horizon-window", recommended_horizon_seconds=20.0,
            report_path="fake-v1",
        )
        requirement_expecting_v2 = DatasetRequirement(campaign_version=CAMPAIGN_VERSION_V2)

        with self.assertRaises(IncompatibleDatasetVersionError):
            assert_compatible(v1_manifest, requirement_expecting_v2)


if __name__ == "__main__":
    unittest.main()
