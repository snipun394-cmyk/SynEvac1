import unittest

from behaviour_profile_resolver.registry import DEFAULT_PROFILE_REGISTRY
from crowd_intelligence.models import DensityThresholds
from simulator.capacity import DefaultCapacityModel, StairCapacityModel
from simulator.congestion import DefaultCongestionModel

from calibration_benchmark.candidates import CapacityWidthCandidate, WalkingSpeedCandidate
from calibration_benchmark.harness import run_calibration_benchmark
from calibration_benchmark.metrics import METRIC_FIELDS
from calibration_benchmark.optional_metrics import RecommendationEffectivenessMetric

from tests.calibration_benchmark_fixtures import DEFINITION_ID, MASTER_SEED, make_building, make_definition


class RunCalibrationBenchmarkTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()
        self.definition = make_definition(occupant_count=12)

    def test_produces_a_comparison_for_every_metric_field(self):

        candidate = WalkingSpeedCandidate("Adult_Default", 0.6, "test", "test rationale")

        result = run_calibration_benchmark(
            candidate, self.building, self.definition, DEFINITION_ID, MASTER_SEED, n_scenarios=3, dt=1.0,
        )

        self.assertEqual(result.n_completed_pairs, 3)
        for field_name in METRIC_FIELDS:
            self.assertIn(field_name, result.comparisons)

    def test_a_slower_candidate_speed_produces_a_longer_mean_evacuation_time(self):

        candidate = WalkingSpeedCandidate("Adult_Default", 0.4, "test", "test rationale")

        result = run_calibration_benchmark(
            candidate, self.building, self.definition, DEFINITION_ID, MASTER_SEED, n_scenarios=3, dt=1.0,
        )

        comparison = result.comparisons["evacuation_time"]
        self.assertGreater(comparison.candidate_mean, comparison.baseline_mean)

    def test_additional_metrics_are_included_when_supplied(self):

        candidate = WalkingSpeedCandidate("Adult_Default", 0.6, "test", "test rationale")

        result = run_calibration_benchmark(
            candidate, self.building, self.definition, DEFINITION_ID, MASTER_SEED, n_scenarios=2, dt=1.0,
            additional_metrics=[RecommendationEffectivenessMetric()],
        )

        self.assertIn("recommendation_effectiveness", result.additional_comparisons)

    def test_never_leaves_any_production_default_mutated(self):

        original_speed = DEFAULT_PROFILE_REGISTRY["Adult_Default"].walking_speed
        original_capacity = DefaultCapacityModel.PEOPLE_PER_METER_OF_WIDTH
        original_stair_capacity = StairCapacityModel.PEOPLE_PER_METER_OF_WIDTH
        original_speed_factor = DefaultCongestionModel.MINIMUM_SPEED_FACTOR

        walking_candidate = WalkingSpeedCandidate("Adult_Default", 0.1, "test", "test")
        capacity_candidate = CapacityWidthCandidate(9.9, "test", "test")

        run_calibration_benchmark(
            walking_candidate, self.building, self.definition, DEFINITION_ID, MASTER_SEED, n_scenarios=1, dt=1.0,
        )
        run_calibration_benchmark(
            capacity_candidate, self.building, self.definition, DEFINITION_ID, MASTER_SEED, n_scenarios=1, dt=1.0,
        )

        self.assertEqual(DEFAULT_PROFILE_REGISTRY["Adult_Default"].walking_speed, original_speed)
        self.assertEqual(DefaultCapacityModel.PEOPLE_PER_METER_OF_WIDTH, original_capacity)
        self.assertEqual(StairCapacityModel.PEOPLE_PER_METER_OF_WIDTH, original_stair_capacity)
        self.assertEqual(DefaultCongestionModel.MINIMUM_SPEED_FACTOR, original_speed_factor)

    def test_zero_scenarios_produces_an_empty_but_well_formed_result(self):

        candidate = WalkingSpeedCandidate("Adult_Default", 0.6, "test", "test rationale")

        result = run_calibration_benchmark(
            candidate, self.building, self.definition, DEFINITION_ID, MASTER_SEED, n_scenarios=0, dt=1.0,
        )

        self.assertEqual(result.n_completed_pairs, 0)
        for comparison in result.comparisons.values():
            self.assertIsNone(comparison.paired.p_value)


if __name__ == "__main__":
    unittest.main()
