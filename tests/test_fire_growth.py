import unittest

from hazard.node_state import HazardNodeState
from hazard.snapshot import HazardSnapshot

from hazard_evolution.contribution import HazardContribution
from hazard_evolution.engine import HazardEvolutionEngine
from hazard_evolution.provider import EvolutionBackedHazardProvider
from hazard_evolution.source import HazardSource

from fire_growth.growth_curve import FireGrowthCurve, TSquaredFireGrowthCurve
from fire_growth.model import FireGrowthModel


class TSquaredFireGrowthCurveTests(unittest.TestCase):

    def test_zero_elapsed_time_is_zero_intensity(self):

        curve = TSquaredFireGrowthCurve(growth_time=100.0)
        self.assertEqual(curve.intensity_at(0.0), 0.0)

    def test_intensity_grows_quadratically(self):

        curve = TSquaredFireGrowthCurve(growth_time=100.0)

        self.assertAlmostEqual(curve.intensity_at(50.0), 0.25)   # (50/100)^2
        self.assertAlmostEqual(curve.intensity_at(100.0), 1.0)   # (100/100)^2

    def test_intensity_saturates_at_one_past_growth_time(self):

        curve = TSquaredFireGrowthCurve(growth_time=100.0)

        self.assertEqual(curve.intensity_at(500.0), 1.0)

    def test_faster_growth_time_reaches_saturation_sooner(self):

        slow = TSquaredFireGrowthCurve(growth_time=600.0)
        fast = TSquaredFireGrowthCurve(growth_time=60.0)

        self.assertLess(slow.intensity_at(60.0), fast.intensity_at(60.0))


class FireGrowthModelTests(unittest.TestCase):

    def test_no_opinion_before_ignition(self):

        model = FireGrowthModel(
            ignition_node_id="z1", ignition_time=30.0,
            growth_curve=TSquaredFireGrowthCurve(growth_time=100.0),
        )

        contribution = model.propose(HazardSnapshot(), time=0.0, dt=10.0)

        self.assertEqual(contribution, HazardContribution())

    def test_produces_a_contribution_only_for_the_ignition_node(self):

        model = FireGrowthModel(
            ignition_node_id="z1", ignition_time=0.0,
            growth_curve=TSquaredFireGrowthCurve(growth_time=100.0),
        )

        contribution = model.propose(HazardSnapshot(), time=0.0, dt=50.0)

        self.assertIn("z1", contribution.node_states)
        self.assertEqual(len(contribution.node_states), 1)
        self.assertEqual(len(contribution.edge_states), 0)

    def test_hazard_score_matches_the_growth_curve_at_elapsed_time(self):

        curve = TSquaredFireGrowthCurve(growth_time=100.0)
        model = FireGrowthModel(ignition_node_id="z1", ignition_time=10.0, growth_curve=curve)

        contribution = model.propose(HazardSnapshot(), time=10.0, dt=40.0)  # step ends at t=50

        self.assertAlmostEqual(
            contribution.node_state("z1").hazard_score,
            curve.intensity_at(40.0),
        )

    def test_does_not_read_previous_snapshot(self):

        # Deliberately stateless w.r.t. history -- an arbitrary,
        # unrelated previous_snapshot must not change the result.
        curve = TSquaredFireGrowthCurve(growth_time=100.0)
        model = FireGrowthModel(ignition_node_id="z1", ignition_time=0.0, growth_curve=curve)

        unrelated_snapshot = HazardSnapshot(
            node_states={"z1": HazardNodeState(hazard_score=0.99)},
        )

        result_with_history = model.propose(unrelated_snapshot, time=0.0, dt=25.0)
        result_without_history = model.propose(HazardSnapshot(), time=0.0, dt=25.0)

        self.assertEqual(result_with_history, result_without_history)

    def test_default_growth_curve_is_used_when_none_supplied(self):

        model = FireGrowthModel(ignition_node_id="z1", ignition_time=0.0)

        self.assertIsInstance(model.growth_curve, TSquaredFireGrowthCurve)

    def test_repeated_stepping_matches_a_single_larger_step_at_the_same_absolute_time(self):

        curve = TSquaredFireGrowthCurve(growth_time=100.0)
        model_a = FireGrowthModel(ignition_node_id="z1", ignition_time=0.0, growth_curve=curve)
        model_b = FireGrowthModel(ignition_node_id="z1", ignition_time=0.0, growth_curve=curve)

        # Five 10-second evolve() steps ...
        engine_a = HazardEvolutionEngine(sources=[model_a])
        snapshot_a = HazardSnapshot(timestamp=0.0)
        for step in range(5):
            snapshot_a = engine_a.evolve(snapshot_a, time=step * 10.0, dt=10.0)

        # ... versus one 50-second step, same source, same absolute end time.
        engine_b = HazardEvolutionEngine(sources=[model_b])
        snapshot_b = HazardSnapshot(timestamp=0.0)
        snapshot_b = engine_b.evolve(snapshot_b, time=0.0, dt=50.0)

        self.assertAlmostEqual(
            snapshot_a.node_state("z1").hazard_score,
            snapshot_b.node_state("z1").hazard_score,
        )


class FireGrowthModelNeverTouchesDownstreamLayersTests(unittest.TestCase):

    def test_model_module_imports_nothing_from_simulation_behavior_or_pathfinding(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "fire_growth"

        forbidden = r"^\s*(from|import)\s+(simulator|behavior|pathfinding|models|designer|navigation)\b"

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"{path.name} imports a downstream/engineering layer directly -- "
                f"FireGrowthModel must only ever produce HazardContribution objects",
            )


class MultipleIgnitionPointsComposeWithoutNewCodeTests(unittest.TestCase):

    def test_two_fire_growth_models_evolve_independently_in_the_same_engine(self):

        curve = TSquaredFireGrowthCurve(growth_time=100.0)

        fire_a = FireGrowthModel(ignition_node_id="z1", ignition_time=0.0, growth_curve=curve)
        fire_b = FireGrowthModel(ignition_node_id="z2", ignition_time=20.0, growth_curve=curve)

        engine = HazardEvolutionEngine(sources=[fire_a, fire_b])

        result = engine.evolve(HazardSnapshot(timestamp=0.0), time=0.0, dt=60.0)

        self.assertAlmostEqual(result.node_state("z1").hazard_score, curve.intensity_at(60.0))
        self.assertAlmostEqual(result.node_state("z2").hazard_score, curve.intensity_at(40.0))

    def test_worst_case_merge_applies_when_two_fires_reach_the_same_node(self):

        # Not a realistic scenario (two independent ignition points at
        # the same node), but confirms the merge strategy -- not
        # FireGrowthModel -- resolves the conflict, exactly as designed
        # for HazardEvolutionEngine.
        fast_curve = TSquaredFireGrowthCurve(growth_time=10.0)
        slow_curve = TSquaredFireGrowthCurve(growth_time=1000.0)

        fast_fire = FireGrowthModel(ignition_node_id="z1", ignition_time=0.0, growth_curve=fast_curve)
        slow_fire = FireGrowthModel(ignition_node_id="z1", ignition_time=0.0, growth_curve=slow_curve)

        engine = HazardEvolutionEngine(sources=[fast_fire, slow_fire])
        result = engine.evolve(HazardSnapshot(timestamp=0.0), time=0.0, dt=10.0)

        self.assertAlmostEqual(result.node_state("z1").hazard_score, fast_curve.intensity_at(10.0))


class FireGrowthModelViaProviderIntegrationTests(unittest.TestCase):

    def test_evolution_backed_provider_surfaces_fire_growth_over_time(self):

        curve = TSquaredFireGrowthCurve(growth_time=200.0)
        model = FireGrowthModel(ignition_node_id="z1", ignition_time=0.0, growth_curve=curve)

        engine = HazardEvolutionEngine(sources=[model])
        provider = EvolutionBackedHazardProvider(engine, HazardSnapshot(timestamp=0.0), dt=10.0)

        early = provider.snapshot_at(20.0).node_state("z1").hazard_score
        later = provider.snapshot_at(150.0).node_state("z1").hazard_score

        self.assertLess(early, later)
        self.assertAlmostEqual(later, curve.intensity_at(150.0))


class CustomHazardSourceCanReplaceFireGrowthModelTests(unittest.TestCase):

    def test_engine_accepts_any_hazard_source_in_place_of_fire_growth_model(self):

        # Proves the second swap point from the design: the whole
        # model can be replaced by a different HazardSource (e.g. a
        # future multi-node Fire Spread model) with zero engine changes.

        class StubFireSpreadModel(HazardSource):
            def propose(self, previous_snapshot, time, dt):
                return HazardContribution(
                    node_states={"z1": HazardNodeState(hazard_score=0.5), "z2": HazardNodeState(hazard_score=0.3)},
                )

        engine = HazardEvolutionEngine(sources=[StubFireSpreadModel()])
        result = engine.evolve(HazardSnapshot(timestamp=0.0), time=0.0, dt=1.0)

        self.assertEqual(result.node_state("z1").hazard_score, 0.5)
        self.assertEqual(result.node_state("z2").hazard_score, 0.3)


if __name__ == "__main__":
    unittest.main()
