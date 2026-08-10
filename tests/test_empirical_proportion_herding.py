import math
import random
import unittest

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.zone import Zone

from navigation.edge import Edge
from navigation.graph_builder import NavigationGraphGenerator

from pathfinding.engine import PathfindingEngine

from behavior.context import DecisionContext
from behavior.route_choice import RouteChoice, ShortestRouteChoiceStrategy

from behavior_library import kinateder_warren_2021_herding_evidence as evidence
from behavior_library.route_choice_strategies import (
    EmpiricalProportionHerdingRouteChoiceStrategy,
    StaticHerdingRouteChoiceStrategy,
)

from simulator.decision import ActionType, BehaviorDecision


# Empirically Parameterized Proportion-Conditioned Herding --
# Kinateder & Warren (2021), Physica A 569, 125746,
# DOI 10.1016/j.physa.2021.125746. Mechanism fidelity tests only --
# verifies the implementation correctly represents the published
# coefficients and the domain policy; does NOT claim behavioral or
# whole-building validation (see this milestone's own chat report).


def make_zone(name, **kwargs):

    fields = dict(x=0.0, y=0.0, width=2.0, height=2.0)
    fields.update(kwargs)

    return Zone(name=name, **fields)


def _build_two_exit_building():

    # The "small isolated synthetic route-choice environment" the
    # milestone asks for: two exits, no congestion, no hazard, no
    # walking simulation -- MultiAgentSimulation.run() is never called
    # anywhere in this file.

    building = Building(name="B")
    floor = building.create_floor(name="Ground Floor")

    start = make_zone("Start", x=0.0, y=0.0)
    # Deliberately asymmetric: WingA (the "majority" exit in every test
    # below) is farther than WingB (the fallback/nearest exit) -- so
    # ShortestRouteChoiceStrategy's own fallback provably lands on a
    # DIFFERENT exit than the herding-majority one, giving the Monte
    # Carlo tests below a genuine, distinguishable signal. A symmetric
    # layout would make "follow majority" and "shortest-path fallback"
    # converge on the same exit regardless of follow_probability,
    # silently masking the very thing these tests exist to measure.
    wing_a = make_zone("WingA", x=-50.0, y=0.0)
    wing_b = make_zone("WingB", x=8.0, y=0.0)

    floor.add_zone(start)
    floor.add_zone(wing_a)
    floor.add_zone(wing_b)

    floor.add_door(Door(name="DA", zone_a_id=start.id, zone_b_id=wing_a.id, floor_id=floor.id, width=5.0))
    floor.add_door(Door(name="DB", zone_a_id=start.id, zone_b_id=wing_b.id, floor_id=floor.id, width=5.0))

    floor.add_exit(Exit(name="ExitA", zone_id=wing_a.id, floor_id=floor.id))
    floor.add_exit(Exit(name="ExitB", zone_id=wing_b.id, floor_id=floor.id))

    graph = NavigationGraphGenerator().build(building)
    engine = PathfindingEngine(graph)

    exit_a_edge = next(e for e in graph.edges if e.edge_type == Edge.EXIT and e.reference.name == "ExitA")
    exit_b_edge = next(e for e in graph.edges if e.edge_type == Edge.EXIT and e.reference.name == "ExitB")

    return engine, start, exit_a_edge, exit_b_edge


class _FakeRoute:

    # The only thing either herding strategy ever reads off a
    # decisions_so_far route is `.edges[-1].id` -- this stand-in avoids
    # needing a real, pathfinding-generated Route object for the
    # already-resolved "crowd" fed into decisions_so_far.

    def __init__(self, edge):
        self.edges = (edge,)


def _crowd_decisions(majority_edge, minority_edge, majority_count, minority_count):

    decisions = {}

    for i in range(majority_count):
        decisions[f"maj{i}"] = BehaviorDecision(
            occupant_id=f"maj{i}", action_type=ActionType.EVACUATE, start_id="Start",
            goal_id="Outside", route=_FakeRoute(majority_edge),
        )

    for i in range(minority_count):
        decisions[f"min{i}"] = BehaviorDecision(
            occupant_id=f"min{i}", action_type=ActionType.EVACUATE, start_id="Start",
            goal_id="Outside", route=_FakeRoute(minority_edge),
        )

    return decisions


def _monte_carlo_follow_rate(engine, start, decisions_so_far, majority_edge, n_trials=4000, **strategy_kwargs):

    context = DecisionContext(graph=None, engine=engine, profile=None, start_id=start.id, decisions_so_far=decisions_so_far)

    rng = random.Random(4242)
    strategy = EmpiricalProportionHerdingRouteChoiceStrategy(rng=rng, **strategy_kwargs)

    followed = 0
    for _ in range(n_trials):
        choice = strategy.choose(context)
        if choice.route is not None and choice.route.edges and choice.route.edges[-1].id == majority_edge.id:
            followed += 1

    return followed / n_trials


# =====================================================
# Evidence module tests -- published coefficients, domain policy
# =====================================================


class EvidenceModuleTests(unittest.TestCase):

    def test_nearest_level_exact_matches(self):

        for level in evidence.SUPPORTED_PROPORTION_LEVELS_PERCENT:
            self.assertEqual(evidence.nearest_supported_proportion_level(level), level)

    def test_nearest_level_ties_break_toward_the_lower_more_conservative_category(self):

        self.assertEqual(evidence.nearest_supported_proportion_level(75), 70)
        self.assertEqual(evidence.nearest_supported_proportion_level(65), 60)
        self.assertEqual(evidence.nearest_supported_proportion_level(85), 80)
        self.assertEqual(evidence.nearest_supported_proportion_level(95), 90)

    def test_model_implied_probability_size_10_matches_hand_computed_logit(self):

        expected = 1.0 / (1.0 + math.exp(-(evidence.INTERCEPT + evidence.PROPORTION_COEFFICIENTS[80])))
        self.assertAlmostEqual(evidence.model_implied_probability(10, 80), expected, places=12)

    def test_model_implied_probability_size_20_includes_the_interaction_term(self):

        expected = 1.0 / (1.0 + math.exp(-(
            evidence.INTERCEPT + evidence.PROPORTION_COEFFICIENTS[80] + evidence.SIZE_20_INTERACTION_COEFFICIENTS[80]
        )))
        self.assertAlmostEqual(evidence.model_implied_probability(20, 80), expected, places=12)

    def test_unsupported_crowd_size_returns_none(self):

        self.assertIsNone(evidence.model_implied_probability(15, 80))
        self.assertIsNone(evidence.follow_probability_for(15, 0.8))

    def test_below_reference_proportion_returns_none(self):

        self.assertIsNone(evidence.follow_probability_for(10, 0.55))
        self.assertIsNone(evidence.follow_probability_for(10, 0.0))

    def test_above_100_percent_fails_safe_not_fabricated(self):

        # Structurally impossible via real exit_edge_counts (majority_count
        # can never exceed total_observed_decisions), but must never
        # raise or invent a value if it ever occurred.
        self.assertIsNone(evidence.follow_probability_for(10, 1.5))

    def test_exactly_100_percent_is_supported(self):

        self.assertIsNotNone(evidence.follow_probability_for(10, 1.0))


# =====================================================
# Strategy tests -- items 1-14
# =====================================================


class ProportionAndSizeProbabilityTests(unittest.TestCase):

    def setUp(self):
        self.engine, self.start, self.exit_a, self.exit_b = _build_two_exit_building()

    def _rate(self, size, majority_percent, n_trials=4000):

        majority_count = size * majority_percent // 100
        minority_count = size - majority_count
        decisions = _crowd_decisions(self.exit_a, self.exit_b, majority_count, minority_count)
        expected = evidence.follow_probability_for(size, majority_count / size)

        observed = _monte_carlo_follow_rate(self.engine, self.start, decisions, self.exit_a, n_trials=n_trials)

        return observed, expected

    def test_60_percent_size_10_matches_published_model_baseline(self):

        observed, expected = self._rate(10, 60)
        self.assertAlmostEqual(observed, expected, delta=0.04)

    def test_70_percent_size_10_small_increase(self):

        observed, expected = self._rate(10, 70)
        self.assertAlmostEqual(observed, expected, delta=0.04)
        self.assertGreater(expected, evidence.follow_probability_for(10, 0.60))

    def test_80_percent_size_10_stronger_increase(self):

        observed, expected = self._rate(10, 80)
        self.assertAlmostEqual(observed, expected, delta=0.04)
        self.assertGreater(expected, evidence.follow_probability_for(10, 0.70))

    def test_90_percent_size_10_stronger_still(self):

        observed, expected = self._rate(10, 90)
        self.assertAlmostEqual(observed, expected, delta=0.04)
        self.assertGreater(expected, evidence.follow_probability_for(10, 0.80))

    def test_100_percent_size_10_strongest(self):

        observed, expected = self._rate(10, 100)
        self.assertAlmostEqual(observed, expected, delta=0.04)
        self.assertGreater(expected, evidence.follow_probability_for(10, 0.90))

    def test_80_percent_size_20_reduced_relative_to_size_10(self):

        observed, expected = self._rate(20, 80)
        self.assertAlmostEqual(observed, expected, delta=0.04)
        self.assertLess(expected, evidence.follow_probability_for(10, 0.80))

    def test_90_percent_size_20_reduced_relative_to_size_10(self):

        observed, expected = self._rate(20, 90)
        self.assertAlmostEqual(observed, expected, delta=0.04)
        self.assertLess(expected, evidence.follow_probability_for(10, 0.90))

    def test_100_percent_size_20_converges_toward_size_10(self):

        observed, expected = self._rate(20, 100)
        self.assertAlmostEqual(observed, expected, delta=0.04)

        gap_at_100 = abs(evidence.follow_probability_for(10, 1.0) - evidence.follow_probability_for(20, 1.0))
        gap_at_90 = abs(evidence.follow_probability_for(10, 0.9) - evidence.follow_probability_for(20, 0.9))

        self.assertLess(gap_at_100, gap_at_90)


class DomainPolicyTests(unittest.TestCase):

    def setUp(self):
        self.engine, self.start, self.exit_a, self.exit_b = _build_two_exit_building()

    def test_unsupported_crowd_size_uses_exact_legacy_fallback(self):

        # size 15 is not in {10, 20} -- must reduce to the legacy
        # constant, not the empirical model.
        decisions = _crowd_decisions(self.exit_a, self.exit_b, 12, 3)

        observed = _monte_carlo_follow_rate(
            self.engine, self.start, decisions, self.exit_a, legacy_follow_probability=0.3,
        )

        self.assertAlmostEqual(observed, 0.3, delta=0.04)

    def test_proportion_below_60_percent_uses_exact_legacy_fallback(self):

        decisions = _crowd_decisions(self.exit_a, self.exit_b, 5, 5)  # size 10, 50% -- below reference

        observed = _monte_carlo_follow_rate(
            self.engine, self.start, decisions, self.exit_a, legacy_follow_probability=0.7,
        )

        self.assertAlmostEqual(observed, 0.7, delta=0.04)

    def test_no_interpolation_75_percent_snaps_to_the_lower_70_percent_category(self):

        # size 20, 15/20 = 75% -- exactly between the 70% and 80%
        # categories. Must resolve to 70%'s own probability, never an
        # interpolated value between the two.
        decisions = _crowd_decisions(self.exit_a, self.exit_b, 15, 5)

        expected_70 = evidence.follow_probability_for(20, 0.70)
        expected_80 = evidence.follow_probability_for(20, 0.80)
        actual = evidence.follow_probability_for(20, 0.75)

        self.assertEqual(actual, expected_70)
        self.assertNotEqual(actual, expected_80)
        # Not an interpolated midpoint either.
        self.assertNotAlmostEqual(actual, (expected_70 + expected_80) / 2, places=6)


class RouteSelectionEquivalenceTests(unittest.TestCase):

    def setUp(self):
        self.engine, self.start, self.exit_a, self.exit_b = _build_two_exit_building()

    def test_majority_exit_selection_matches_plain_static_herding(self):

        decisions = _crowd_decisions(self.exit_a, self.exit_b, 8, 2)  # size 10, 80%
        context = DecisionContext(
            graph=None, engine=self.engine, profile=None, start_id=self.start.id, decisions_so_far=decisions,
        )

        empirical = EmpiricalProportionHerdingRouteChoiceStrategy(rng=random.Random(1))
        plain = StaticHerdingRouteChoiceStrategy(follow_probability=1.0, rng=random.Random(1))

        empirical_choice = empirical.choose(context)
        plain_choice = plain.choose(context)

        self.assertEqual(empirical_choice.route.edges[-1].id, self.exit_a.id)
        self.assertEqual(empirical_choice.route.edges[-1].id, plain_choice.route.edges[-1].id)

    def test_fallback_route_choice_is_used_unmodified_when_not_following(self):

        class _MarkerFallback:
            def choose(self, context):
                return RouteChoice(goal_id="MARKER", route=None)

        decisions = _crowd_decisions(self.exit_a, self.exit_b, 5, 5)  # size 10, 50% -> out of domain, legacy=0.0

        strategy = EmpiricalProportionHerdingRouteChoiceStrategy(
            legacy_follow_probability=0.0, rng=random.Random(1), fallback=_MarkerFallback(),
        )
        context = DecisionContext(
            graph=None, engine=self.engine, profile=None, start_id=self.start.id, decisions_so_far=decisions,
        )

        choice = strategy.choose(context)

        self.assertEqual(choice.goal_id, "MARKER")

    def test_capacity_and_width_are_never_accessed(self):

        class _CapacityTrapEdge:

            def __init__(self, real_edge):
                self._real_edge = real_edge

            @property
            def id(self):
                return self._real_edge.id

            @property
            def edge_type(self):
                return self._real_edge.edge_type

            @property
            def capacity(self):
                raise AssertionError("capacity must never be accessed by herding strategies")

            @property
            def width(self):
                raise AssertionError("width must never be accessed by herding strategies")

        trap_a = _CapacityTrapEdge(self.exit_a)
        trap_b = _CapacityTrapEdge(self.exit_b)

        decisions = _crowd_decisions(trap_a, trap_b, 8, 2)
        context = DecisionContext(
            graph=None, engine=self.engine, profile=None, start_id=self.start.id, decisions_so_far=decisions,
        )

        strategy = EmpiricalProportionHerdingRouteChoiceStrategy(rng=random.Random(1))

        # Must not raise -- exercises many rolls to cover both the
        # follow and fallback branches, neither of which may touch
        # capacity/width.
        for _ in range(200):
            strategy.choose(context)
