import random
import unittest

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from behavior.context import DecisionContext
from behavior.orchestrator import HumanBehaviorLayer
from behavior.profile import BehaviorProfile

from behavior_library.decision_strategies import ComplianceDecisionStrategy
from behavior_library.pre_movement_strategies import ProbabilisticPreMovementDelay
from behavior_library.route_choice_strategies import StaticHerdingRouteChoiceStrategy

from simulator.decision import BehaviorDecision

from scenario import Scenario, ScenarioFire, ScenarioMetadata, ScenarioOccupant
from scenario_runner import run

from behaviour_profile_resolver import register_occupants
from behaviour_profile_resolver.registrar import _derive_occupant_seed

from ai_decision.engine import AIDecisionEngine

from simulation_runtime import SimulationRuntime


# docs/architecture/reproducibility_review.md's own approved fix, verified
# here end to end: DecisionContext.rng (§7.1), HumanBehaviorLayer.
# register()'s rng parameter (§7.3), register_occupants()'s per-occupant
# seed derivation (§7.4), and the three behavior_library strategies
# preferring context.rng over self.rng (§7.2).


def make_building():

    # zone-2 -> door-1 -> zone-1 -> exit-1: a real, non-trivial route --
    # without a door connecting them, every occupant placed at zone-2
    # would be topologically isolated and end up STATIONARY regardless
    # of any behavioural draw, making the draw's effect unobservable in
    # depart_time/arrival_time (the exact pitfall this comment exists
    # to warn against reintroducing).
    floor = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-1", name="Lobby", x=0.0, y=0.0, width=10.0, height=8.0),
            Zone(id="zone-2", name="Office", x=20.0, y=0.0, width=6.0, height=6.0),
        ],
        doors=[Door(id="door-1", normally_open=True, zone_a_id="zone-1", zone_b_id="zone-2")],
        exits=[Exit(id="exit-1", zone_id="zone-1")],
    )

    return Building(name="Test Building", id="building-1", floors=[floor])


def make_metadata(**overrides):

    defaults = dict(
        scenario_id="scn-1", definition_id="def-1", definition_content_hash="hash-abc",
        generation_version="scenario_generator/1", seed=42, created_at="2026-07-13T00:00:00",
    )
    defaults.update(overrides)

    return ScenarioMetadata(**defaults)


def make_occupants(profile_id, count=5, zone_id="zone-2"):

    return tuple(
        ScenarioOccupant(
            occupant_id=f"occ-{zone_id}-{index}", zone_id=zone_id, floor_id="floor-1",
            position=(21.0, 1.0), behaviour_profile_id=profile_id,
        )
        for index in range(count)
    )


def make_scenario(profile_id="Adult_Default", seed=42, count=5):

    return Scenario(
        metadata=make_metadata(seed=seed),
        occupants=make_occupants(profile_id, count=count),
        fire=ScenarioFire(
            ignition_zone_id="zone-2", ignition_floor_id="floor-1",
            fire_profile="Electrical", growth_parameters={"growth_time": 200.0},
        ),
    )


def depart_and_arrival_times(movement_result):

    return tuple(sorted(
        (timeline.depart_time, timeline.arrival_time)
        for timeline in movement_result.occupants.values()
    ))


class DecisionContextRngFieldTests(unittest.TestCase):

    def test_rng_defaults_to_none(self):

        context = DecisionContext(
            graph=None, engine=None, profile=BehaviorProfile(occupant_id="occ-1"), start_id="zone-1",
        )

        self.assertIsNone(context.rng)

    def test_rng_accepts_a_random_instance(self):

        seeded = random.Random(1)
        context = DecisionContext(
            graph=None, engine=None, profile=BehaviorProfile(occupant_id="occ-1"), start_id="zone-1",
            rng=seeded,
        )

        self.assertIs(context.rng, seeded)


class HumanBehaviorLayerRngThreadingTests(unittest.TestCase):

    def _layer(self):

        # A minimal, duck-typed stand-in -- deliberately NOT a real
        # MultiAgentSimulation and NOT a monkeypatch of that class
        # (mutating a shared, real class would leak into every other
        # test in the process). HumanBehaviorLayer only ever calls
        # simulation.submit_decision(decision) and reads simulation.engine.
        simulation = _StubSimulation()

        return HumanBehaviorLayer(simulation, simulation.engine)

    def test_register_without_rng_still_works(self):

        # Backward compatibility -- an existing caller that never
        # passes rng= must keep working exactly as before this change.
        layer = self._layer()
        profile = BehaviorProfile(occupant_id="occ-1")

        decision_id = layer.register(
            start_id="zone-1", profile=profile, decision_strategy=_AlwaysWait(),
        )

        self.assertEqual(decision_id, "submitted:occ-1")

    def test_register_threads_rng_into_decision_context(self):

        layer = self._layer()
        profile = BehaviorProfile(occupant_id="occ-1")
        seeded = random.Random(7)

        capturing_strategy = _CapturingDecisionStrategy()

        layer.register(
            start_id="zone-1", profile=profile, decision_strategy=capturing_strategy, rng=seeded,
        )

        self.assertIs(capturing_strategy.captured_context.rng, seeded)


class _StubEngine:

    def __init__(self):
        self.graph = None


class _StubSimulation:

    def __init__(self):
        self.engine = _StubEngine()

    def submit_decision(self, decision):
        return f"submitted:{decision.occupant_id}"


class _AlwaysWait:

    def decide(self, context):

        from behavior.intent import ActionIntent
        from simulator.decision import ActionType

        return ActionIntent(
            occupant_id=context.profile.occupant_id, action_type=ActionType.WAIT,
            requires_movement=False,
        )


class _CapturingDecisionStrategy:

    def __init__(self):
        self.captured_context = None

    def decide(self, context):

        from behavior.intent import ActionIntent
        from simulator.decision import ActionType

        self.captured_context = context

        return ActionIntent(
            occupant_id=context.profile.occupant_id, action_type=ActionType.WAIT,
            requires_movement=False,
        )


class StrategyContextRngPreferenceTests(unittest.TestCase):

    # Each strategy: context.rng, when supplied, must be preferred over
    # self.rng, and self.rng must remain the fallback when context.rng
    # is None (docs/architecture/reproducibility_review.md §7.2).

    def _context(self, rng=None):

        return DecisionContext(
            graph=None, engine=None,
            profile=BehaviorProfile(occupant_id="occ-1", compliance_level=0.5), start_id="zone-1",
            rng=rng,
        )

    def test_compliance_strategy_prefers_context_rng(self):

        strategy = ComplianceDecisionStrategy(rng=random.Random(999))

        context_a = self._context(rng=random.Random(1))
        context_b = self._context(rng=random.Random(1))

        intent_a = strategy.decide(context_a)
        intent_b = strategy.decide(context_b)

        self.assertEqual(intent_a.action_type, intent_b.action_type)

    def test_compliance_strategy_falls_back_to_self_rng_when_context_rng_is_none(self):

        strategy_a = ComplianceDecisionStrategy(rng=random.Random(5))
        strategy_b = ComplianceDecisionStrategy(rng=random.Random(5))

        intent_a = strategy_a.decide(self._context(rng=None))
        intent_b = strategy_b.decide(self._context(rng=None))

        self.assertEqual(intent_a.action_type, intent_b.action_type)

    def test_pre_movement_delay_prefers_context_rng(self):

        strategy = ProbabilisticPreMovementDelay(median_delay=30.0, rng=random.Random(999))

        delay_a = strategy.delay(self._context(rng=random.Random(1)))
        delay_b = strategy.delay(self._context(rng=random.Random(1)))

        self.assertEqual(delay_a, delay_b)

    def test_pre_movement_delay_falls_back_to_self_rng_when_context_rng_is_none(self):

        strategy_a = ProbabilisticPreMovementDelay(median_delay=30.0, rng=random.Random(5))
        strategy_b = ProbabilisticPreMovementDelay(median_delay=30.0, rng=random.Random(5))

        delay_a = strategy_a.delay(self._context(rng=None))
        delay_b = strategy_b.delay(self._context(rng=None))

        self.assertEqual(delay_a, delay_b)

    def test_static_herding_prefers_context_rng(self):

        leader_route = _FakeRoute(goal_id="outside", edge_id="exit-1")
        leader_decision = BehaviorDecision(
            occupant_id="leader", action_type=None, start_id="zone-1", goal_id="outside",
            route=leader_route,
        )

        # Two strategy instances with DIFFERENT self.rng seeds (999 vs
        # 111) -- if both nonetheless agree given the SAME context.rng
        # seed (1), self.rng cannot be the thing driving the outcome.
        strategy_a = StaticHerdingRouteChoiceStrategy(
            follow_probability=0.5, rng=random.Random(999), fallback=_FixedFallback(),
        )
        strategy_b = StaticHerdingRouteChoiceStrategy(
            follow_probability=0.5, rng=random.Random(111), fallback=_FixedFallback(),
        )

        engine = _HerdingStubEngine()

        context_a = DecisionContext(
            graph=None, engine=engine, profile=BehaviorProfile(occupant_id="occ-1"),
            start_id="zone-2", decisions_so_far={"leader": leader_decision}, rng=random.Random(1),
        )
        context_b = DecisionContext(
            graph=None, engine=engine, profile=BehaviorProfile(occupant_id="occ-1"),
            start_id="zone-2", decisions_so_far={"leader": leader_decision}, rng=random.Random(1),
        )

        choice_a = strategy_a.choose(context_a)
        choice_b = strategy_b.choose(context_b)

        self.assertEqual(choice_a.goal_id, choice_b.goal_id)


class _FakeGoal:

    def __init__(self, goal_id):
        self.id = goal_id


class _FakeEdge:

    def __init__(self, edge_id):
        self.id = edge_id


class _FakeRoute:

    def __init__(self, goal_id, edge_id):
        self.goal = _FakeGoal(goal_id)
        self.edges = [_FakeEdge(edge_id)]
        self.node_ids = ["zone-1", goal_id]


class _HerdingStubEngine:

    # Always resolves to the same exit the leader used (edge_id
    # "exit-1"), so the "herd" branch of StaticHerdingRouteChoiceStrategy
    # always finds a matching candidate when it is the branch taken.

    def nearest_exit(self, start_id):
        return _FakeRoute(goal_id="outside", edge_id="exit-1")

    def alternative_paths(self, start_id, goal_id, k):
        return [_FakeRoute(goal_id="outside", edge_id="exit-1")]


class _FixedFallback:

    # A deliberately different, fixed outcome from the herd branch, so
    # the two branches are observably distinguishable in a test.

    def choose(self, context):

        from behavior.route_choice import RouteChoice

        return RouteChoice(goal_id="fallback-goal", route=None)


class DeriveOccupantSeedTests(unittest.TestCase):

    def test_same_inputs_always_derive_the_same_seed(self):

        first = _derive_occupant_seed(42, "occ-zone-2-0")
        second = _derive_occupant_seed(42, "occ-zone-2-0")

        self.assertEqual(first, second)

    def test_different_occupant_ids_derive_different_seeds(self):

        seed_a = _derive_occupant_seed(42, "occ-zone-2-0")
        seed_b = _derive_occupant_seed(42, "occ-zone-2-1")

        self.assertNotEqual(seed_a, seed_b)

    def test_different_scenario_seeds_derive_different_seeds(self):

        seed_a = _derive_occupant_seed(42, "occ-zone-2-0")
        seed_b = _derive_occupant_seed(99, "occ-zone-2-0")

        self.assertNotEqual(seed_a, seed_b)

    def test_returns_a_non_negative_integer(self):

        seed = _derive_occupant_seed(42, "occ-zone-2-0")

        self.assertIsInstance(seed, int)
        self.assertGreaterEqual(seed, 0)


class EndToEndDeterministicReplayTests(unittest.TestCase):

    # The headline proof this fix exists for: running the identical
    # Scenario through scenario_runner -> behaviour_profile_resolver
    # -> MultiAgentSimulation.run() twice must produce identical
    # results, for every default profile -- including the five
    # previously-flaky ones docs/architecture/reproducibility_review.md
    # §3/§4.1 identified (only Staff_Default was deterministic before
    # this fix).

    PROFILES = (
        "Adult_Default", "Child_Default", "Wheelchair_Default",
        "Visitor_Default", "FireWarden_Default", "Staff_Default",
    )

    def test_repeated_simulation_of_the_same_scenario_is_identical(self):

        for profile_id in self.PROFILES:

            with self.subTest(profile=profile_id):

                scenario = make_scenario(profile_id=profile_id)
                building = make_building()

                results = []

                for _ in range(2):

                    context = run(scenario, building)
                    register_occupants(context)
                    movement = context.simulation.run()

                    results.append(depart_and_arrival_times(movement))

                self.assertEqual(results[0], results[1])

    def test_two_independently_built_contexts_from_the_same_scenario_agree(self):

        for profile_id in self.PROFILES:

            with self.subTest(profile=profile_id):

                scenario = make_scenario(profile_id=profile_id)

                first_context = run(scenario, make_building())
                register_occupants(first_context)
                first_movement = first_context.simulation.run()

                second_context = run(scenario, make_building())
                register_occupants(second_context)
                second_movement = second_context.simulation.run()

                self.assertEqual(
                    depart_and_arrival_times(first_movement),
                    depart_and_arrival_times(second_movement),
                )

    def test_different_scenario_seeds_produce_different_behaviour_draws(self):

        # A sanity check on the other direction -- the fix must not
        # accidentally collapse every scenario onto the same draws
        # regardless of seed.
        scenario_a = make_scenario(profile_id="Adult_Default", seed=1)
        scenario_b = make_scenario(profile_id="Adult_Default", seed=2)

        context_a = run(scenario_a, make_building())
        register_occupants(context_a)
        movement_a = context_a.simulation.run()

        context_b = run(scenario_b, make_building())
        register_occupants(context_b)
        movement_b = context_b.simulation.run()

        self.assertNotEqual(
            depart_and_arrival_times(movement_a), depart_and_arrival_times(movement_b),
        )

    def test_occupants_in_the_same_scenario_do_not_share_identical_draws(self):

        # Regression guard for the root cause itself (docs/architecture/
        # reproducibility_review.md §5): before the fix, every occupant
        # of a profile shared one process-lifetime rng, so distinct
        # occupants' pre-movement delays were drawn from a single
        # advancing stream rather than independent, occupant-specific
        # ones. This does not assert full statistical independence, only
        # that occupants are not all getting identical depart times
        # (which a bug re-seeding the same value per occupant would
        # produce).
        scenario = make_scenario(profile_id="Adult_Default", count=8)

        context = run(scenario, make_building())
        register_occupants(context)
        movement = context.simulation.run()

        depart_times = [timeline.depart_time for timeline in movement.occupants.values()]

        self.assertGreater(len(set(depart_times)), 1)


class SimulationRuntimeEndToEndDeterminismTests(unittest.TestCase):

    # Extends tests/test_simulation_runtime.py's own determinism
    # coverage (which deliberately used only the always-deterministic
    # Staff_Default profile) to Adult_Default specifically -- the exact
    # profile whose non-determinism originally forced that workaround.

    def test_two_independent_runtimes_agree_using_adult_default(self):

        scenario = make_scenario(profile_id="Adult_Default", count=3)

        results = []

        for _ in range(2):

            context = run(scenario, make_building())
            register_occupants(context)

            engine = AIDecisionEngine(base_engine=context.engine)
            runtime = SimulationRuntime(context, engine, dt=10.0)

            tick_results = runtime.run()

            results.append(tuple(
                (tick.time, tick.decision.unsafe_zone_ids) for tick in tick_results
            ))

        self.assertEqual(results[0], results[1])


class BackwardCompatibilityTests(unittest.TestCase):

    # docs/architecture/reproducibility_review.md §7.5: existing
    # behavior must be preserved when no rng is supplied anywhere in
    # the chain -- a direct caller of these strategies/HumanBehaviorLayer
    # that predates this change keeps working unmodified.

    def test_compliance_strategy_still_works_when_constructed_and_called_the_old_way(self):

        strategy = ComplianceDecisionStrategy(rng=random.Random(3))
        context = DecisionContext(
            graph=None, engine=None,
            profile=BehaviorProfile(occupant_id="occ-1", compliance_level=1.0), start_id="zone-1",
        )

        # compliance_level=1.0 -- always complies, regardless of the
        # roll -- proves the call succeeds and returns a real intent
        # without needing context.rng at all.
        intent = strategy.decide(context)

        self.assertIsNotNone(intent)

    def test_pre_movement_delay_still_works_when_constructed_and_called_the_old_way(self):

        strategy = ProbabilisticPreMovementDelay(median_delay=30.0, rng=random.Random(3))
        context = DecisionContext(
            graph=None, engine=None, profile=BehaviorProfile(occupant_id="occ-1"), start_id="zone-1",
        )

        delay = strategy.delay(context)

        self.assertIsInstance(delay, float)
        self.assertGreater(delay, 0.0)

    def test_register_occupants_public_signature_is_unchanged(self):

        import inspect

        signature = inspect.signature(register_occupants)

        self.assertEqual(list(signature.parameters), ["context", "registry"])

    def test_human_behavior_layer_register_signature_gains_only_a_trailing_optional_rng(self):

        import inspect

        signature = inspect.signature(HumanBehaviorLayer.register)
        parameters = list(signature.parameters)

        self.assertEqual(parameters[-1], "rng")
        self.assertIsNone(signature.parameters["rng"].default)


if __name__ == "__main__":
    unittest.main()
