import unittest

from hazard.node_state import HazardNodeState
from hazard.snapshot import HazardSnapshot

from hazard_evolution.contribution import HazardContribution
from hazard_evolution.engine import HazardEvolutionEngine
from hazard_evolution.source import HazardSource

from tenability.model import TenabilityModel
from tenability.scoring import SmokeVisibilityTenabilityScorer, TenabilityScorer


class FixedContributionSource(HazardSource):

    def __init__(self, contribution):
        self.contribution = contribution

    def propose(self, previous_snapshot, time, dt):
        return self.contribution


class SmokeVisibilityTenabilityScorerTests(unittest.TestCase):

    def setUp(self):
        self.scorer = SmokeVisibilityTenabilityScorer()

    def test_clear_conditions_score_zero(self):

        score = self.scorer.score(HazardNodeState(smoke_level=0.0, visibility=30.0))
        self.assertEqual(score, 0.0)

    def test_missing_readings_score_zero(self):

        score = self.scorer.score(HazardNodeState())
        self.assertEqual(score, 0.0)

    def test_smoke_level_passes_through_when_visibility_is_fine(self):

        score = self.scorer.score(HazardNodeState(smoke_level=0.4, visibility=30.0))
        self.assertEqual(score, 0.4)

    def test_smoke_level_passes_through_when_visibility_is_unknown(self):

        score = self.scorer.score(HazardNodeState(smoke_level=0.4, visibility=None))
        self.assertEqual(score, 0.4)

    def test_impaired_visibility_dominates_low_smoke(self):

        score = self.scorer.score(HazardNodeState(smoke_level=0.1, visibility=1.0))

        expected_impairment = (3.0 - 1.0) / 3.0
        self.assertAlmostEqual(score, expected_impairment)

    def test_zero_visibility_is_fully_untenable(self):

        score = self.scorer.score(HazardNodeState(smoke_level=0.0, visibility=0.0))
        self.assertEqual(score, 1.0)

    def test_visibility_at_threshold_is_not_impaired(self):

        score = self.scorer.score(HazardNodeState(smoke_level=0.0, visibility=3.0))
        self.assertEqual(score, 0.0)

    def test_score_is_the_worse_of_smoke_and_impairment(self):

        high_smoke = self.scorer.score(HazardNodeState(smoke_level=0.9, visibility=30.0))
        high_impairment = self.scorer.score(HazardNodeState(smoke_level=0.0, visibility=0.1))

        self.assertAlmostEqual(high_smoke, 0.9)
        self.assertGreater(high_impairment, 0.9)

    def test_threshold_is_overridable(self):

        scorer = SmokeVisibilityTenabilityScorer(min_tenable_visibility_m=10.0)
        score = scorer.score(HazardNodeState(smoke_level=0.0, visibility=5.0))

        self.assertAlmostEqual(score, 0.5)


class TenabilityModelTests(unittest.TestCase):

    def test_no_opinion_when_snapshot_has_no_nodes(self):

        model = TenabilityModel()
        contribution = model.propose(HazardSnapshot(), time=0.0, dt=1.0)

        self.assertEqual(contribution, HazardContribution())

    def test_no_opinion_for_a_node_scoring_zero(self):

        snapshot = HazardSnapshot(
            node_states={"z1": HazardNodeState(smoke_level=0.0, visibility=30.0)},
        )
        model = TenabilityModel()
        contribution = model.propose(snapshot, time=0.0, dt=1.0)

        self.assertIsNone(contribution.node_state("z1"))

    def test_produces_hazard_score_matching_the_scorer(self):

        snapshot = HazardSnapshot(
            node_states={"z1": HazardNodeState(smoke_level=0.6, visibility=25.0)},
        )
        model = TenabilityModel()
        contribution = model.propose(snapshot, time=0.0, dt=1.0)

        self.assertEqual(contribution.node_state("z1").hazard_score, 0.6)

    def test_never_invents_node_ids_not_already_in_the_snapshot(self):

        snapshot = HazardSnapshot(
            node_states={"z1": HazardNodeState(smoke_level=0.5, visibility=1.0)},
        )
        model = TenabilityModel()
        contribution = model.propose(snapshot, time=0.0, dt=1.0)

        self.assertEqual(set(contribution.node_states.keys()), {"z1"})

    def test_produces_no_edge_states_ever(self):

        snapshot = HazardSnapshot(
            node_states={"z1": HazardNodeState(smoke_level=0.9, visibility=0.5)},
        )
        model = TenabilityModel()
        contribution = model.propose(snapshot, time=0.0, dt=1.0)

        self.assertEqual(len(contribution.edge_states), 0)

    def test_does_not_mutate_previous_snapshot(self):

        original_state = HazardNodeState(smoke_level=0.9, visibility=0.5)
        snapshot = HazardSnapshot(node_states={"z1": original_state})

        model = TenabilityModel()
        model.propose(snapshot, time=0.0, dt=1.0)

        self.assertEqual(snapshot.node_state("z1"), original_state)

    def test_default_scorer_is_smoke_visibility_scorer(self):

        model = TenabilityModel()
        self.assertIsInstance(model.scorer, SmokeVisibilityTenabilityScorer)

    def test_custom_scorer_is_used_when_supplied(self):

        class StubScorer(TenabilityScorer):
            def score(self, node_state):
                return 0.77

        snapshot = HazardSnapshot(node_states={"z1": HazardNodeState()})
        model = TenabilityModel(scorer=StubScorer())
        contribution = model.propose(snapshot, time=0.0, dt=1.0)

        self.assertEqual(contribution.node_state("z1").hazard_score, 0.77)


class TenabilityModelNeverTouchesDownstreamLayersTests(unittest.TestCase):

    def test_module_imports_nothing_from_navigation_simulation_behavior_or_the_building_model(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "tenability"

        forbidden = r"^\s*(from|import)\s+(simulator|behavior|pathfinding|navigation|models|designer)\b"

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"{path.name} imports a downstream/engineering layer directly -- "
                f"TenabilityModel must only ever transform fields already on "
                f"HazardNodeState",
            )


class TenabilityModelViaEngineIntegrationTests(unittest.TestCase):

    def test_tenability_lags_one_step_behind_the_source_it_derives_from(self):

        # Demonstrates the key architectural distinction from Fire/
        # Smoke: TenabilityModel reads previous_snapshot, so its
        # opinion about a node only appears once that node has
        # actually appeared in a *prior* merged snapshot, one step
        # behind the raw smoke reading itself.
        smoke_source = FixedContributionSource(
            HazardContribution(
                node_states={"z1": HazardNodeState(smoke_level=0.2, visibility=1.0)},
            )
        )
        tenability = TenabilityModel()
        engine = HazardEvolutionEngine(sources=[smoke_source, tenability])

        first = engine.evolve(HazardSnapshot(timestamp=0.0), time=0.0, dt=1.0)

        self.assertEqual(first.node_state("z1").smoke_level, 0.2)
        self.assertEqual(first.node_state("z1").hazard_score, 0.0)  # tenability had nothing to read yet

        second = engine.evolve(first, time=1.0, dt=1.0)

        expected_impairment = (3.0 - 1.0) / 3.0
        self.assertAlmostEqual(second.node_state("z1").hazard_score, expected_impairment)
        self.assertEqual(second.node_state("z1").smoke_level, 0.2)  # preserved from smoke_source, untouched by tenability

    def test_engine_accepts_any_hazard_source_in_place_of_tenability_model(self):

        class StubFEDModel(HazardSource):
            def propose(self, previous_snapshot, time, dt):
                return HazardContribution(
                    node_states={"z1": HazardNodeState(hazard_score=0.42)},
                )

        engine = HazardEvolutionEngine(sources=[StubFEDModel()])
        result = engine.evolve(HazardSnapshot(timestamp=0.0), time=0.0, dt=1.0)

        self.assertEqual(result.node_state("z1").hazard_score, 0.42)


if __name__ == "__main__":
    unittest.main()
