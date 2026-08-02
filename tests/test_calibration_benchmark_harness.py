import unittest

from behaviour_profile_resolver.registry import DEFAULT_PROFILE_REGISTRY
from crowd_intelligence.models import DensityThresholds
from simulator.capacity import DefaultCapacityModel, StairCapacityModel
from simulator.congestion import DefaultCongestionModel

from calibration_benchmark.candidates import CapacityWidthCandidate, FlowRegionCapacityCandidate, WalkingSpeedCandidate
from calibration_benchmark.harness import run_calibration_benchmark
from calibration_benchmark.metrics import METRIC_FIELDS
from calibration_benchmark.optional_metrics import RecommendationEffectivenessMetric
from calibration_benchmark.report import render_markdown_report
from calibration_benchmark.simulation_seam import run_with_overrides

from navigation.graph_builder import NavigationGraphGenerator
from navigation.flow_region import FlowRegion

from scenario_pipeline import run_batch_pipeline

from simulator.capacity import StairCapacityModel
from simulator.congestion import StairAwareCongestionModel

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


class FlowRegionCapacityCandidateEndToEndTests(unittest.TestCase):

    # Milestone 4's own Definition of Done: a full run_calibration_benchmark()
    # call with the Flow-Region candidate as the "candidate" arm, compared
    # against today's behavior as "baseline", executes end-to-end and
    # produces a real report via the already-existing, unmodified
    # calibration_benchmark/report.py. make_building()'s door-a+exit-a
    # are confirmed (see navigation/flow_region_inference.py's own
    # heuristic) to form one real, shared two-edge chain FlowRegion --
    # this is not a trivial single-edge no-op scenario.

    def setUp(self):

        self.building = make_building()
        self.definition = make_definition(occupant_count=12)

    def test_the_fixture_building_forms_a_real_multi_edge_chain_region(self):

        graph = NavigationGraphGenerator().build(self.building)

        self.assertEqual(graph.flow_regions["door-a"].id, graph.flow_regions["exit-a"].id)
        self.assertEqual(graph.flow_regions["door-a"].region_kind, FlowRegion.CHAIN)

    def test_runs_end_to_end_and_produces_every_metric(self):

        candidate = FlowRegionCapacityCandidate("test", "test rationale")

        result = run_calibration_benchmark(
            candidate, self.building, self.definition, DEFINITION_ID, MASTER_SEED, n_scenarios=3, dt=1.0,
        )

        self.assertEqual(result.n_completed_pairs, 3)
        for field_name in METRIC_FIELDS:
            self.assertIn(field_name, result.comparisons)

    def test_report_renders_without_error(self):

        candidate = FlowRegionCapacityCandidate("test", "test rationale")

        result = run_calibration_benchmark(
            candidate, self.building, self.definition, DEFINITION_ID, MASTER_SEED, n_scenarios=2, dt=1.0,
        )

        markdown = render_markdown_report(result)

        self.assertIn("Hybrid Flow Regions", markdown)
        self.assertIn("architecture variant", markdown)

    def test_baseline_arm_matches_todays_production_behavior_exactly(self):

        # The candidate's OWN baseline arm (via run_calibration_benchmark)
        # must be numerically identical to calling run_with_overrides()
        # directly with today's production defaults and no Flow Region
        # involvement at all -- proving the baseline side of this new
        # candidate is a true no-op, not just type-correct in isolation.

        batch = run_batch_pipeline(self.definition, DEFINITION_ID, self.building, MASTER_SEED, 1)
        scenario = batch.scenarios[0]

        direct_movement, _, _ = run_with_overrides(
            scenario, self.building,
            capacity_model=StairCapacityModel(), congestion_model=StairAwareCongestionModel(),
            dt=1.0,
        )

        candidate = FlowRegionCapacityCandidate("test", "test rationale")
        via_candidate_movement, _, _ = run_with_overrides(
            scenario, self.building,
            capacity_model=candidate.baseline_capacity_model(),
            congestion_model=candidate.baseline_congestion_model(),
            dt=1.0,
            use_flow_regions=candidate.baseline_use_flow_regions(),
        )

        self.assertEqual(
            direct_movement.total_evacuation_time,
            via_candidate_movement.total_evacuation_time,
        )

    def test_candidate_arm_differs_from_baseline_because_a_real_region_is_shared(self):

        # Not a strict requirement of every possible building, but for
        # THIS fixture's genuine 2-edge chain, sharing one admission
        # unit across door-a+exit-a is expected to change at least one
        # of the headline metrics relative to today's independent
        # per-edge admission control -- otherwise the wiring would be
        # silently inert even when a real region exists.
        candidate = FlowRegionCapacityCandidate("test", "test rationale")

        result = run_calibration_benchmark(
            candidate, self.building, self.definition, DEFINITION_ID, MASTER_SEED, n_scenarios=5, dt=1.0,
        )

        differs_somewhere = any(
            comparison.baseline_mean != comparison.candidate_mean
            for comparison in result.comparisons.values()
            if comparison.baseline_mean is not None and comparison.candidate_mean is not None
        )

        self.assertTrue(differs_somewhere)


if __name__ == "__main__":
    unittest.main()
