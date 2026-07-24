import dataclasses
import pathlib
import re
import unittest

from evacuation_recommendation.evidence import ai_bottleneck_probability
from evacuation_recommendation.models import RecommendationConfig, RecommendationWeights
from evacuation_recommendation.scoring import score_candidate

from ai_registry.inference_service import BottleneckOccurrencePrediction, EvacuationTimePrediction

from live_system.live_ai_gateway import AISystemStatus, LiveAIPredictionSnapshot

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from navigation.graph_builder import NavigationGraphGenerator

from live_occupants.manager import LiveOccupantManager

from evacuation_recommendation.engine import EvacuationRecommendationEngine


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# =====================================================
# Define AI's Real Operational Role milestone -- mechanically proves
# the specific architectural claims docs/architecture/
# ai_operational_role.md makes, rather than merely asserting them in
# prose. No model trained, no scoring code changed by this file.
# =====================================================


class _Bottleneck:

    def __init__(self, probability):
        self.probability = probability


class _Snapshot:

    def __init__(self, probability):
        self.bottleneck = _Bottleneck(probability) if probability is not None else None


class GlobalBottleneckCannotAffectRankingTests(unittest.TestCase):

    # Phase 16 -- "current global bottleneck AI cannot change relative
    # exit ranking." Two otherwise-identical candidates, scored once
    # with a LOW ai_bottleneck_probability and once with a HIGH one --
    # the relative order (which candidate scores higher) must be
    # identical either way, because the SAME probability is applied to
    # both.

    def _score_two_candidates(self, probability):

        weights = RecommendationWeights()
        config = RecommendationConfig()

        # Candidate A: closer (shorter distance) than Candidate B --
        # A should always outrank B on distance_goodness alone,
        # regardless of what AI says, since AI is uniform.
        score_a, _ = score_candidate(
            route_distance_m=10.0, min_distance=10.0, max_distance=50.0,
            congestion_level=None, queue_count=None, throughput=None,
            trajectory_support=None, emergency_response_elevated=False,
            ai_bottleneck_probability=probability, weights=weights, config=config,
        )
        score_b, _ = score_candidate(
            route_distance_m=50.0, min_distance=10.0, max_distance=50.0,
            congestion_level=None, queue_count=None, throughput=None,
            trajectory_support=None, emergency_response_elevated=False,
            ai_bottleneck_probability=probability, weights=weights, config=config,
        )

        return score_a, score_b

    def test_relative_order_unchanged_across_low_and_high_ai_probability(self):

        score_a_low, score_b_low = self._score_two_candidates(0.05)
        score_a_high, score_b_high = self._score_two_candidates(0.95)

        # A outranks B in BOTH cases -- AI probability never flips it.
        self.assertGreater(score_a_low, score_b_low)
        self.assertGreater(score_a_high, score_b_high)

        # The MARGIN between A and B is identical regardless of AI
        # probability -- proving AI contributed the SAME amount to
        # both, never a differentiating amount.
        self.assertAlmostEqual(score_a_low - score_b_low, score_a_high - score_b_high, places=9)

    def test_absolute_score_does_change_with_ai_probability(self):

        # AI is not a no-op -- it changes the ABSOLUTE score (and
        # therefore the explanation/reason codes), just never the
        # RELATIVE order. Confirming this distinguishes "rank-inert by
        # construction" from "AI does literally nothing."

        score_low, _ = score_candidate(
            route_distance_m=10.0, min_distance=10.0, max_distance=10.0,
            congestion_level=None, queue_count=None, throughput=None,
            trajectory_support=None, emergency_response_elevated=False,
            ai_bottleneck_probability=0.05, weights=RecommendationWeights(), config=RecommendationConfig(),
        )
        score_high, _ = score_candidate(
            route_distance_m=10.0, min_distance=10.0, max_distance=10.0,
            congestion_level=None, queue_count=None, throughput=None,
            trajectory_support=None, emergency_response_elevated=False,
            ai_bottleneck_probability=0.95, weights=RecommendationWeights(), config=RecommendationConfig(),
        )

        self.assertNotEqual(score_low, score_high)

    def test_uniform_across_an_arbitrary_number_of_candidates(self):

        # Generalizes the two-candidate proof above to N candidates
        # with distinct distances -- the SAME AI probability applied to
        # every one of them can only ever shift every score by the
        # identical constant, so sorting by score is invariant to it.

        weights = RecommendationWeights()
        config = RecommendationConfig()
        distances = [5.0, 12.0, 20.0, 33.0, 41.0]

        def scores_for(probability):
            return [
                score_candidate(
                    route_distance_m=d, min_distance=min(distances), max_distance=max(distances),
                    congestion_level=None, queue_count=None, throughput=None,
                    trajectory_support=None, emergency_response_elevated=False,
                    ai_bottleneck_probability=probability, weights=weights, config=config,
                )[0]
                for d in distances
            ]

        scores_low = scores_for(0.0)
        scores_high = scores_for(1.0)

        order_low = sorted(range(len(distances)), key=lambda i: -scores_low[i])
        order_high = sorted(range(len(distances)), key=lambda i: -scores_high[i])

        self.assertEqual(order_low, order_high)


class BottleneckPredictionHasNoLocationFieldTests(unittest.TestCase):

    # Phase 16 -- "current AI output has no location field." Mechanical
    # introspection of the actual dataclass, not a doc claim.

    def test_bottleneck_occurrence_prediction_has_no_location_field(self):

        field_names = {f.name for f in dataclasses.fields(BottleneckOccurrencePrediction)}

        location_like = {name for name in field_names if any(
            token in name.lower() for token in ("zone", "exit", "door", "stair", "location", "asset")
        )}

        self.assertEqual(
            location_like, set(),
            f"BottleneckOccurrencePrediction has location-like field(s) {location_like} -- "
            f"the model's own output type must carry no zone/exit/door/stair/location field "
            f"until a genuinely localized model exists (docs/architecture/ai_operational_role.md).",
        )

        self.assertEqual(
            field_names,
            {"probability", "predicted_occurrence", "threshold", "model_id", "model_version",
             "feature_schema_version", "timestamp"},
        )

    def test_ai_bottleneck_probability_evidence_helper_returns_a_single_scalar(self):

        # evacuation_recommendation.evidence.ai_bottleneck_probability()
        # is the ONE function that extracts AI evidence for scoring --
        # confirms it returns a bare float (a single building-wide
        # number), never a per-exit mapping.

        snapshot = _Snapshot(0.42)
        result = ai_bottleneck_probability(snapshot)

        self.assertIsInstance(result, float)
        self.assertEqual(result, 0.42)


class SafetyExclusionIgnoresAIProbabilityTests(unittest.TestCase):

    # Phase 16 -- "deterministic safety excludes unsafe exits
    # regardless of AI probability." evacuation_recommendation/ never
    # imports decision_policy at all (already mechanically enforced by
    # tests/test_evacuation_recommendation_architecture_guards.py,
    # re-verified here directly for this milestone's own record), and
    # score_candidate() itself has no code path through which
    # ai_bottleneck_probability could ever re-include an excluded
    # candidate -- AI is applied ONLY inside score_candidate(), which is
    # only ever called on candidates ranking.rank_exits_for_zone()
    # already resolved as reachable/safe.

    def test_evacuation_recommendation_never_imports_decision_policy(self):

        package_dir = REPO_ROOT / "evacuation_recommendation"
        forbidden = re.compile(r"^\s*(from|import)\s+decision_policy\b", re.MULTILINE)

        for path in package_dir.glob("*.py"):

            text = path.read_text(encoding="utf-8")

            self.assertIsNone(
                forbidden.search(text),
                f"evacuation_recommendation/{path.name} imports decision_policy -- AI/deterministic "
                f"safety exclusion must remain structurally separate from any AI-influenced scoring.",
            )

    def test_score_candidate_has_no_parameter_capable_of_re_including_an_excluded_candidate(self):

        # score_candidate()'s only inputs are already-resolved per-
        # candidate evidence values -- it has no "building" / "hazard" /
        # "excluded_zone_ids" parameter at all, so it could not
        # override safety exclusion even if a caller wanted it to.

        import inspect

        signature = inspect.signature(score_candidate)
        parameter_names = set(signature.parameters.keys())

        forbidden_params = {"hazard_summary", "excluded_zone_ids", "building", "building_state"}

        self.assertEqual(parameter_names & forbidden_params, set())


class AIAbsenceDoesNotPreventRecommendationsTests(unittest.TestCase):

    # Phase 16 -- "AI absence does not prevent recommendations."

    def setUp(self):

        floor = Floor(
            id="floor-1", name="Ground Floor",
            zones=[Zone(id="zone-1", name="Z1", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="floor-1")],
            doors=[],
            exits=[Exit(id="EXIT-1", name="Main Exit", floor_id="floor-1", zone_id="zone-1",
                        start_point=(0.0, 3.0), end_point=(0.0, 7.0))],
        )
        self.building = Building(id="ai-role-test-building", name="AI Role Test Building", floors=[floor])
        self.graph = NavigationGraphGenerator().build(self.building)
        self.manager = LiveOccupantManager()
        self.manager.update("OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (5.0, 5.0), None, None, 0.9, 1.0)

        self.engine = EvacuationRecommendationEngine(self.building, self.graph, self.manager)

    def test_compute_succeeds_with_no_ai_prediction_snapshot_at_all(self):

        snapshot = self.engine.compute(1.0)  # every optional param, including ai_prediction_snapshot, omitted

        self.assertIn("zone-1", snapshot.zones)
        recommendation = snapshot.zones["zone-1"]
        self.assertTrue(recommendation.candidates)
        self.assertEqual(recommendation.candidates[0].exit_id, "EXIT-1")

    def test_compute_succeeds_with_an_explicit_none_ai_prediction_snapshot(self):

        snapshot = self.engine.compute(1.0, ai_prediction_snapshot=None)

        self.assertIn("zone-1", snapshot.zones)
        self.assertTrue(snapshot.zones["zone-1"].candidates)

    def test_compute_succeeds_with_an_ai_snapshot_carrying_no_bottleneck(self):

        # AISystemStatus.UNAVAILABLE with bottleneck=None -- exactly
        # what RegistryLiveAIInferenceGateway produces when no
        # compatible model is registered.
        unavailable = LiveAIPredictionSnapshot(
            timestamp=1.0, building_state_timestamp=1.0, feature_schema_version=None,
            system_status=AISystemStatus.UNAVAILABLE, bottleneck=None,
        )

        snapshot = self.engine.compute(1.0, ai_prediction_snapshot=unavailable)

        self.assertTrue(snapshot.zones["zone-1"].candidates)


class ExperimentalEvacuationTimeCannotInfluenceRoutingTests(unittest.TestCase):

    # Phase 16 -- "experimental evacuation-time output cannot influence
    # routing." Mechanically proves evacuation_time_experimental/
    # EvacuationTimePrediction is never referenced anywhere in the
    # routing-capable packages.

    _ROUTING_PACKAGES = (
        "evacuation_recommendation", "evacuation_guidance", "dynamic_signage",
        "navigation", "pathfinding", "decision_policy",
    )

    def test_no_routing_package_references_evacuation_time_experimental(self):

        forbidden = re.compile(r"evacuation_time_experimental|EvacuationTimePrediction")

        offenders = []

        for package_name in self._ROUTING_PACKAGES:

            package_dir = REPO_ROOT / package_name

            if not package_dir.is_dir():
                continue

            for path in package_dir.glob("*.py"):

                text = path.read_text(encoding="utf-8")

                if forbidden.search(text):
                    offenders.append(f"{package_name}/{path.name}")

        self.assertEqual(
            offenders, [],
            f"The following routing-capable files reference the experimental evacuation-time "
            f"prediction: {offenders}. It must remain surfaced only as diagnostic/UI information "
            f"(command_center.live_ai_panel), never consumed by anything that chooses a route.",
        )

    def test_evacuation_time_prediction_has_no_location_field_either(self):

        field_names = {f.name for f in dataclasses.fields(EvacuationTimePrediction)}

        location_like = {name for name in field_names if any(
            token in name.lower() for token in ("zone", "exit", "door", "stair", "location", "asset")
        )}

        self.assertEqual(location_like, set())


if __name__ == "__main__":
    unittest.main()
