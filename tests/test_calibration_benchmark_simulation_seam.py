import unittest

from scenario_pipeline import run_batch_pipeline
from scenario_runner import run as run_scenario

from calibration_benchmark.simulation_seam import run_with_overrides
from calibration_benchmark.candidates import WalkingSpeedCandidate

from tests.calibration_benchmark_fixtures import DEFINITION_ID, MASTER_SEED, make_building, make_definition


class RunWithOverridesReproducesProductionDefaultsTests(unittest.TestCase):

    # run_with_overrides() with every override left at its default must
    # reproduce scenario_runner.run()'s own composition exactly -- the
    # central non-drift guarantee this milestone's own simulation seam
    # depends on (see simulation_seam.py's own module docstring).

    def setUp(self):

        self.building = make_building()
        batch = run_batch_pipeline(make_definition(occupant_count=6), DEFINITION_ID, self.building, MASTER_SEED, 1)
        self.assertEqual(len(batch.scenarios), 1)
        self.scenario = batch.scenarios[0]

    def test_default_call_matches_scenario_runner_run_ground_truth(self):

        movement_result, ground_truth, _building_copy = run_with_overrides(self.scenario, self.building, dt=1.0)

        context = run_scenario(self.scenario, self.building)
        from behaviour_profile_resolver.registrar import register_occupants
        from ai_decision.engine import AIDecisionEngine
        from ground_truth import SimulationArtifacts, analyze as analyze_ground_truth
        from simulation_runtime import SimulationRuntime

        register_occupants(context)
        decision_engine = AIDecisionEngine(base_engine=context.engine)
        runtime = SimulationRuntime(context, decision_engine, dt=1.0, perception_provider=None)
        runtime.run()

        expected_ground_truth = analyze_ground_truth(
            SimulationArtifacts(scenario=self.scenario, building=self.building, movement_result=runtime.movement_result),
        )

        self.assertEqual(ground_truth.total_evacuation_time, expected_ground_truth.total_evacuation_time)
        self.assertEqual(ground_truth.people_evacuated, expected_ground_truth.people_evacuated)
        self.assertEqual(ground_truth.peak_congestion_value, expected_ground_truth.peak_congestion_value)


class RunWithOverridesInjectsOverridesTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()
        batch = run_batch_pipeline(make_definition(occupant_count=10), DEFINITION_ID, self.building, MASTER_SEED, 1)
        self.scenario = batch.scenarios[0]

    def test_a_much_slower_walking_speed_registry_produces_a_longer_evacuation(self):

        candidate = WalkingSpeedCandidate("Adult_Default", 0.2, "test", "test")

        baseline_result, baseline_gt, _ = run_with_overrides(
            self.scenario, self.building, registry=candidate.baseline_registry(), dt=1.0,
        )
        candidate_result, candidate_gt, _ = run_with_overrides(
            self.scenario, self.building, registry=candidate.candidate_registry(), dt=1.0,
        )

        self.assertGreater(candidate_gt.total_evacuation_time, baseline_gt.total_evacuation_time)

    def test_does_not_mutate_the_default_profile_registry(self):

        from behaviour_profile_resolver.registry import DEFAULT_PROFILE_REGISTRY

        original_speed = DEFAULT_PROFILE_REGISTRY["Adult_Default"].walking_speed
        candidate = WalkingSpeedCandidate("Adult_Default", 0.2, "test", "test")

        run_with_overrides(self.scenario, self.building, registry=candidate.candidate_registry(), dt=1.0)

        self.assertEqual(DEFAULT_PROFILE_REGISTRY["Adult_Default"].walking_speed, original_speed)


if __name__ == "__main__":
    unittest.main()
