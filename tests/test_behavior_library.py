import random
import unittest

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.zone import Zone

from navigation.graph_builder import NavigationGraphGenerator

from pathfinding.engine import PathfindingEngine

from simulator.coordinator import MultiAgentSimulation
from simulator.decision import ActionType
from simulator.occupant import OccupantState

from behavior.context import DecisionContext
from behavior.intent import AlwaysEvacuateDecisionStrategy
from behavior.orchestrator import HumanBehaviorLayer
from behavior.pre_movement import NoPreMovementDelay
from behavior.profile import BehaviorProfile, Role
from behavior.route_choice import ShortestRouteChoiceStrategy

from behavior_library.decision_strategies import (
    AlwaysIgnoreDecisionStrategy,
    AlwaysWaitDecisionStrategy,
    BasicHelpingDecisionStrategy,
    ComplianceDecisionStrategy,
)
from behavior_library.pre_movement_strategies import ProbabilisticPreMovementDelay
from behavior_library.route_choice_strategies import (
    FamiliarityBasedRouteChoiceStrategy,
    FollowLeaderRouteChoiceStrategy,
    HelpTargetRouteChoiceStrategy,
    StaticHerdingRouteChoiceStrategy,
)


def make_zone(name, **kwargs):

    fields = dict(x=0.0, y=0.0, width=2.0, height=2.0)
    fields.update(kwargs)

    return Zone(name=name, **fields)


def build_two_zone_building():

    # Room -- door --> Corridor -- exit --> Outside

    building = Building(name="B")
    floor = building.create_floor(name="Ground Floor")

    room = make_zone("Room", x=0.0, y=0.0)
    corridor = make_zone("Corridor", x=10.0, y=0.0)

    floor.add_zone(room)
    floor.add_zone(corridor)

    door = Door(name="D1", zone_a_id=room.id, zone_b_id=corridor.id, floor_id=floor.id)
    floor.add_door(door)

    exit_obj = Exit(name="Ex", zone_id=corridor.id, floor_id=floor.id)
    floor.add_exit(exit_obj)

    graph = NavigationGraphGenerator().build(building)
    engine = PathfindingEngine(graph)

    return building, floor, room, corridor, door, exit_obj, engine


def build_two_exit_building():

    # Start -- door_b --> RoomB -- exit_b --> Outside   (short)
    # Start -- door_c --> RoomC -- exit_c --> Outside   (long)

    building = Building(name="B")
    floor = building.create_floor(name="Ground Floor")

    start = make_zone("Start", x=0.0, y=0.0)
    room_b = make_zone("RoomB", x=5.0, y=0.0)
    room_c = make_zone("RoomC", x=30.0, y=0.0)

    floor.add_zone(start)
    floor.add_zone(room_b)
    floor.add_zone(room_c)

    door_b = Door(name="DB", zone_a_id=start.id, zone_b_id=room_b.id, floor_id=floor.id)
    door_c = Door(name="DC", zone_a_id=start.id, zone_b_id=room_c.id, floor_id=floor.id)
    floor.add_door(door_b)
    floor.add_door(door_c)

    exit_b = Exit(name="ExB", zone_id=room_b.id, floor_id=floor.id)
    exit_c = Exit(name="ExC", zone_id=room_c.id, floor_id=floor.id)
    floor.add_exit(exit_b)
    floor.add_exit(exit_c)

    graph = NavigationGraphGenerator().build(building)
    engine = PathfindingEngine(graph)

    return building, floor, start, room_b, room_c, door_b, door_c, exit_b, exit_c, engine


def make_context(engine, start_id, profile=None, decisions_so_far=None):

    return DecisionContext(
        graph=engine.graph if engine is not None else None, engine=engine,
        profile=profile or BehaviorProfile(occupant_id="p1"),
        start_id=start_id,
        decisions_so_far=decisions_so_far or {},
    )


class AlwaysWaitAndIgnoreTests(unittest.TestCase):

    def test_always_wait_produces_a_non_movement_wait_intent(self):

        context = make_context(None, "z1")
        # engine unused by these trivial strategies -- pass None safely
        intent = AlwaysWaitDecisionStrategy().decide(context)

        self.assertEqual(intent.action_type, ActionType.WAIT)
        self.assertFalse(intent.requires_movement)

    def test_always_ignore_produces_a_non_movement_ignore_intent(self):

        context = make_context(None, "z1")
        intent = AlwaysIgnoreDecisionStrategy().decide(context)

        self.assertEqual(intent.action_type, ActionType.IGNORE)
        self.assertFalse(intent.requires_movement)


class ComplianceDecisionStrategyTests(unittest.TestCase):

    def test_low_roll_delegates_to_compliant_strategy(self):

        strategy = ComplianceDecisionStrategy(rng=_FixedRandom(0.1))
        profile = BehaviorProfile(occupant_id="p1", compliance_level=0.5)
        context = make_context(None, "z1", profile=profile)

        intent = strategy.decide(context)

        self.assertEqual(intent.action_type, ActionType.EVACUATE)

    def test_high_roll_delegates_to_noncompliant_strategy(self):

        strategy = ComplianceDecisionStrategy(rng=_FixedRandom(0.9))
        profile = BehaviorProfile(occupant_id="p1", compliance_level=0.5)
        context = make_context(None, "z1", profile=profile)

        intent = strategy.decide(context)

        self.assertEqual(intent.action_type, ActionType.WAIT)

    def test_full_compliance_always_evacuates_regardless_of_roll(self):

        strategy = ComplianceDecisionStrategy(rng=_FixedRandom(0.999))
        profile = BehaviorProfile(occupant_id="p1", compliance_level=1.0)
        context = make_context(None, "z1", profile=profile)

        self.assertEqual(strategy.decide(context).action_type, ActionType.EVACUATE)

    def test_zero_compliance_never_evacuates(self):

        profile = BehaviorProfile(occupant_id="p1", compliance_level=0.0)
        context = make_context(None, "z1", profile=profile)

        # roll (0.0) <= compliance_level (0.0) would still count as
        # compliant, so use a roll strictly greater than zero to
        # exercise the noncompliant branch.
        strategy = ComplianceDecisionStrategy(rng=_FixedRandom(0.0001))
        self.assertEqual(strategy.decide(context).action_type, ActionType.WAIT)

    def test_composes_with_a_custom_compliant_strategy(self):

        profile = BehaviorProfile(occupant_id="p1", compliance_level=1.0)
        profile.traits["helping_occupant_id"] = "victim-1"
        context = make_context(None, "z1", profile=profile)

        strategy = ComplianceDecisionStrategy(
            compliant_strategy=BasicHelpingDecisionStrategy(),
            rng=_FixedRandom(0.0),
        )

        intent = strategy.decide(context)

        self.assertEqual(intent.action_type, ActionType.HELP)


class _FixedRandom(random.Random):

    # Deterministic stand-in for random.Random() -- always returns the
    # same value from random(), used to make compliance/herding rolls
    # reproducible in tests without depending on a real seed's
    # internals.

    def __init__(self, value):
        super().__init__()
        self.value = value

    def random(self):
        return self.value


class BasicHelpingDecisionStrategyTests(unittest.TestCase):

    def test_no_target_falls_back_to_evacuate(self):

        context = make_context(None, "z1")
        intent = BasicHelpingDecisionStrategy().decide(context)

        self.assertEqual(intent.action_type, ActionType.EVACUATE)

    def test_target_present_produces_a_help_intent_requiring_movement(self):

        profile = BehaviorProfile(occupant_id="helper")
        profile.traits["helping_occupant_id"] = "victim-1"
        context = make_context(None, "z1", profile=profile)

        intent = BasicHelpingDecisionStrategy().decide(context)

        self.assertEqual(intent.action_type, ActionType.HELP)
        self.assertTrue(intent.requires_movement)
        self.assertEqual(intent.metadata["helping_occupant_id"], "victim-1")


class FamiliarityBasedRouteChoiceStrategyTests(unittest.TestCase):

    def setUp(self):

        (
            self.building, self.floor, self.start, self.room_b, self.room_c,
            self.door_b, self.door_c, self.exit_b, self.exit_c, self.engine,
        ) = build_two_exit_building()

    def test_no_familiarity_data_matches_shortest_route_choice(self):

        context = make_context(self.engine, self.start.id)

        familiar_choice = FamiliarityBasedRouteChoiceStrategy().choose(context)
        shortest_choice = ShortestRouteChoiceStrategy().choose(context)

        self.assertEqual(familiar_choice.route.edge_ids, shortest_choice.route.edge_ids)

    def test_prefers_the_familiar_exit_even_when_longer(self):

        profile = BehaviorProfile(occupant_id="p1", familiarity={self.room_c.id: 1.0})
        context = make_context(self.engine, self.start.id, profile=profile)

        choice = FamiliarityBasedRouteChoiceStrategy(max_alternatives=2).choose(context)

        self.assertIn(self.door_c.id, choice.route.edge_ids)
        self.assertIn(self.exit_c.id, choice.route.edge_ids)

    def test_falls_back_when_unreachable(self):

        isolated_building = Building(name="Isolated")
        floor = isolated_building.create_floor(name="F")
        zone = make_zone("Z", x=0.0, y=0.0)
        floor.add_zone(zone)

        graph = NavigationGraphGenerator().build(isolated_building)
        engine = PathfindingEngine(graph)

        context = make_context(engine, zone.id)
        choice = FamiliarityBasedRouteChoiceStrategy().choose(context)

        self.assertIsNone(choice.goal_id)
        self.assertIsNone(choice.route)


class StaticHerdingRouteChoiceStrategyTests(unittest.TestCase):

    def setUp(self):

        (
            self.building, self.floor, self.start, self.room_b, self.room_c,
            self.door_b, self.door_c, self.exit_b, self.exit_c, self.engine,
        ) = build_two_exit_building()

    def _decision_via(self, exit_edge_id, other_edge_id, start_id="start-decoy"):

        from pathfinding.route import Route
        from simulator.decision import BehaviorDecision

        node = self.engine.graph.find_node(self.start.id)
        edge = next(e for e in self.engine.graph.edges if e.id == exit_edge_id)

        route = Route(nodes=[node], edges=[edge], total_cost=1.0, total_distance=1.0)

        return BehaviorDecision(
            occupant_id="peer", action_type=ActionType.EVACUATE,
            start_id=start_id, goal_id="outside", route=route,
        )

    def test_no_peers_decided_yet_falls_back_to_shortest(self):

        context = make_context(self.engine, self.start.id)

        herd_choice = StaticHerdingRouteChoiceStrategy(rng=_FixedRandom(0.0)).choose(context)
        shortest_choice = ShortestRouteChoiceStrategy().choose(context)

        self.assertEqual(herd_choice.route.edge_ids, shortest_choice.route.edge_ids)

    def test_follows_the_majority_chosen_exit(self):

        decisions_so_far = {
            "peer1": self._decision_via(self.exit_c.id, self.exit_b.id),
            "peer2": self._decision_via(self.exit_c.id, self.exit_b.id),
            "peer3": self._decision_via(self.exit_b.id, self.exit_c.id),
        }
        context = make_context(self.engine, self.start.id, decisions_so_far=decisions_so_far)

        choice = StaticHerdingRouteChoiceStrategy(
            follow_probability=1.0, rng=_FixedRandom(0.0),
        ).choose(context)

        self.assertIn(self.exit_c.id, choice.route.edge_ids)

    def test_follow_probability_zero_never_herds(self):

        decisions_so_far = {"peer1": self._decision_via(self.exit_c.id, self.exit_b.id)}
        context = make_context(self.engine, self.start.id, decisions_so_far=decisions_so_far)

        choice = StaticHerdingRouteChoiceStrategy(
            follow_probability=0.0, rng=_FixedRandom(0.5),
        ).choose(context)

        shortest_choice = ShortestRouteChoiceStrategy().choose(context)
        self.assertEqual(choice.route.edge_ids, shortest_choice.route.edge_ids)


class FollowLeaderRouteChoiceStrategyTests(unittest.TestCase):

    def setUp(self):

        (
            self.building, self.floor, self.room, self.corridor,
            self.door, self.exit_obj, self.engine,
        ) = build_two_zone_building()

    def test_no_leader_trait_falls_back_to_shortest(self):

        context = make_context(self.engine, self.room.id)

        choice = FollowLeaderRouteChoiceStrategy().choose(context)
        shortest_choice = ShortestRouteChoiceStrategy().choose(context)

        self.assertEqual(choice.route.edge_ids, shortest_choice.route.edge_ids)

    def test_leader_not_yet_decided_falls_back(self):

        profile = BehaviorProfile(occupant_id="follower", role=Role.FOLLOWER)
        profile.traits["leader_occupant_id"] = "leader-1"
        context = make_context(self.engine, self.room.id, profile=profile)

        choice = FollowLeaderRouteChoiceStrategy().choose(context)
        shortest_choice = ShortestRouteChoiceStrategy().choose(context)

        self.assertEqual(choice.route.edge_ids, shortest_choice.route.edge_ids)

    def test_leader_decided_copies_leader_route(self):

        leader_route = self.engine.nearest_exit(self.room.id)

        from simulator.decision import BehaviorDecision

        leader_decision = BehaviorDecision(
            occupant_id="leader-1", action_type=ActionType.EVACUATE,
            start_id=self.room.id, goal_id=leader_route.goal.id, route=leader_route,
        )

        profile = BehaviorProfile(occupant_id="follower", role=Role.FOLLOWER)
        profile.traits["leader_occupant_id"] = "leader-1"
        context = make_context(
            self.engine, self.room.id, profile=profile,
            decisions_so_far={"leader-1": leader_decision},
        )

        choice = FollowLeaderRouteChoiceStrategy().choose(context)

        self.assertEqual(choice.route.edge_ids, leader_route.edge_ids)


class HelpTargetRouteChoiceStrategyTests(unittest.TestCase):

    def setUp(self):

        (
            self.building, self.floor, self.room, self.corridor,
            self.door, self.exit_obj, self.engine,
        ) = build_two_zone_building()

    def test_no_target_falls_back_to_shortest(self):

        context = make_context(self.engine, self.room.id)

        choice = HelpTargetRouteChoiceStrategy().choose(context)
        shortest_choice = ShortestRouteChoiceStrategy().choose(context)

        self.assertEqual(choice.route.edge_ids, shortest_choice.route.edge_ids)

    def test_target_not_decided_yet_falls_back(self):

        profile = BehaviorProfile(occupant_id="helper")
        profile.traits["helping_occupant_id"] = "victim-1"
        context = make_context(self.engine, self.room.id, profile=profile)

        choice = HelpTargetRouteChoiceStrategy().choose(context)
        shortest_choice = ShortestRouteChoiceStrategy().choose(context)

        self.assertEqual(choice.route.edge_ids, shortest_choice.route.edge_ids)

    def test_routes_toward_the_targets_registered_location(self):

        from simulator.decision import BehaviorDecision

        victim_decision = BehaviorDecision(
            occupant_id="victim-1", action_type=ActionType.WAIT,
            start_id=self.corridor.id,
        )

        profile = BehaviorProfile(occupant_id="helper")
        profile.traits["helping_occupant_id"] = "victim-1"
        context = make_context(
            self.engine, self.room.id, profile=profile,
            decisions_so_far={"victim-1": victim_decision},
        )

        choice = HelpTargetRouteChoiceStrategy().choose(context)

        self.assertEqual(choice.goal_id, self.corridor.id)
        self.assertEqual(choice.route.goal.id, self.corridor.id)


class ProbabilisticPreMovementDelayTests(unittest.TestCase):

    def test_rejects_nonpositive_median_delay(self):

        with self.assertRaises(ValueError):
            ProbabilisticPreMovementDelay(median_delay=0.0)

    def test_delay_is_always_non_negative(self):

        strategy = ProbabilisticPreMovementDelay(median_delay=10.0, rng=random.Random(1))
        context = make_context(None, "z1")

        for _ in range(200):
            self.assertGreaterEqual(strategy.delay(context), 0.0)

    def test_seeded_rng_is_reproducible(self):

        context = make_context(None, "z1")

        strategy_a = ProbabilisticPreMovementDelay(median_delay=10.0, rng=random.Random(42))
        strategy_b = ProbabilisticPreMovementDelay(median_delay=10.0, rng=random.Random(42))

        sequence_a = [strategy_a.delay(context) for _ in range(20)]
        sequence_b = [strategy_b.delay(context) for _ in range(20)]

        self.assertEqual(sequence_a, sequence_b)

    def test_median_delay_shapes_the_distribution(self):

        context = make_context(None, "z1")
        strategy = ProbabilisticPreMovementDelay(median_delay=30.0, spread=0.3, rng=random.Random(7))

        samples = sorted(strategy.delay(context) for _ in range(2000))
        sample_median = samples[len(samples) // 2]

        self.assertAlmostEqual(sample_median, 30.0, delta=5.0)


class RealisticEvacuationScenarioTests(unittest.TestCase):

    # Demonstrates behavior profiles actually diverging in outcome --
    # not just each strategy in isolation, but composed through
    # HumanBehaviorLayer exactly as a real scenario would use them.

    def test_familiar_and_unfamiliar_occupants_take_different_exits(self):

        (
            building, floor, start, room_b, room_c,
            door_b, door_c, exit_b, exit_c, engine,
        ) = build_two_exit_building()

        sim = MultiAgentSimulation(engine)
        layer = HumanBehaviorLayer(sim)

        layer.register(
            start.id,
            BehaviorProfile(occupant_id="local", familiarity={room_c.id: 1.0}),
            decision_strategy=AlwaysEvacuateDecisionStrategy(),
            route_choice_strategy=FamiliarityBasedRouteChoiceStrategy(max_alternatives=2),
        )
        layer.register(
            start.id,
            BehaviorProfile(occupant_id="stranger"),
            decision_strategy=AlwaysEvacuateDecisionStrategy(),
            route_choice_strategy=ShortestRouteChoiceStrategy(),
        )

        result = sim.run()

        self.assertIn(exit_c.id, result.occupants["local"].traversed_edge_ids)
        self.assertIn(exit_b.id, result.occupants["stranger"].traversed_edge_ids)
        self.assertNotEqual(
            result.occupants["local"].traversed_edge_ids,
            result.occupants["stranger"].traversed_edge_ids,
        )

    def test_compliant_occupant_evacuates_while_noncompliant_stays_put(self):

        _, _, room, corridor, door, exit_obj, engine = build_two_zone_building()

        sim = MultiAgentSimulation(engine)
        layer = HumanBehaviorLayer(sim)

        layer.register(
            room.id,
            BehaviorProfile(occupant_id="rule_follower", compliance_level=1.0),
            decision_strategy=ComplianceDecisionStrategy(rng=_FixedRandom(0.0)),
        )
        layer.register(
            room.id,
            BehaviorProfile(occupant_id="rule_breaker", compliance_level=0.0),
            decision_strategy=ComplianceDecisionStrategy(rng=_FixedRandom(0.9999)),
        )

        result = sim.run()

        self.assertEqual(result.occupants["rule_follower"].state, OccupantState.ARRIVED)
        self.assertEqual(result.occupants["rule_breaker"].state, OccupantState.STATIONARY)

    def test_follower_ends_up_on_the_identical_route_as_the_leader(self):

        _, _, room, corridor, door, exit_obj, engine = build_two_zone_building()

        sim = MultiAgentSimulation(engine)
        layer = HumanBehaviorLayer(sim)

        layer.register(
            room.id,
            BehaviorProfile(occupant_id="leader", role=Role.LEADER),
            decision_strategy=AlwaysEvacuateDecisionStrategy(),
        )

        follower_profile = BehaviorProfile(occupant_id="follower", role=Role.FOLLOWER)
        follower_profile.traits["leader_occupant_id"] = "leader"

        layer.register(
            room.id,
            follower_profile,
            decision_strategy=AlwaysEvacuateDecisionStrategy(),
            route_choice_strategy=FollowLeaderRouteChoiceStrategy(),
        )

        result = sim.run()

        self.assertEqual(
            result.occupants["leader"].traversed_edge_ids,
            result.occupants["follower"].traversed_edge_ids,
        )

    def test_helper_travels_to_the_victim_instead_of_straight_to_the_exit(self):

        _, _, room, corridor, door, exit_obj, engine = build_two_zone_building()

        sim = MultiAgentSimulation(engine)
        layer = HumanBehaviorLayer(sim)

        # Victim is stationary, waiting in the Corridor zone.
        layer.register(
            corridor.id,
            BehaviorProfile(occupant_id="victim"),
            decision_strategy=AlwaysWaitDecisionStrategy(),
        )

        helper_profile = BehaviorProfile(occupant_id="helper")
        helper_profile.traits["helping_occupant_id"] = "victim"

        layer.register(
            room.id,
            helper_profile,
            decision_strategy=BasicHelpingDecisionStrategy(),
            route_choice_strategy=HelpTargetRouteChoiceStrategy(),
        )

        result = sim.run()

        self.assertEqual(result.occupants["helper"].route.goal.id, corridor.id)
        self.assertEqual(result.occupants["victim"].state, OccupantState.STATIONARY)

    def test_probabilistic_delay_staggers_departure_relative_to_no_delay(self):

        _, _, room, corridor, door, exit_obj, engine = build_two_zone_building()

        sim = MultiAgentSimulation(engine)
        layer = HumanBehaviorLayer(sim)

        layer.register(
            room.id,
            BehaviorProfile(occupant_id="immediate"),
            decision_strategy=AlwaysEvacuateDecisionStrategy(),
            pre_movement_strategy=NoPreMovementDelay(),
        )
        layer.register(
            room.id,
            BehaviorProfile(occupant_id="delayed"),
            decision_strategy=AlwaysEvacuateDecisionStrategy(),
            pre_movement_strategy=ProbabilisticPreMovementDelay(
                median_delay=20.0, rng=random.Random(3),
            ),
        )

        result = sim.run()

        self.assertLess(
            result.occupants["immediate"].arrival_time,
            result.occupants["delayed"].arrival_time,
        )


class BehaviorLibraryIndependenceTests(unittest.TestCase):

    def test_behavior_library_never_touches_reference_or_designer(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "behavior_library"

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertNotIn(
                ".reference", text, f"{path.name} touches .reference directly"
            )
            self.assertIsNone(
                re.search(r"^\s*(from|import)\s+(models|designer)\b", text, re.MULTILINE),
                f"{path.name} imports models/designer directly",
            )

    def test_frozen_behavior_package_never_imports_behavior_library(self):

        import pathlib
        import re

        behavior_dir = pathlib.Path(__file__).resolve().parent.parent / "behavior"

        for path in behavior_dir.glob("*.py"):

            text = path.read_text()

            self.assertIsNone(
                re.search(r"^\s*(from|import)\s+behavior_library\b", text, re.MULTILINE),
                f"behavior/{path.name} imports behavior_library/ -- reverses "
                f"the dependency direction; behavior/ (frozen) must stay "
                f"unaware of behavior_library/",
            )

    def test_frozen_subsystems_never_import_behavior_library(self):

        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parent.parent

        frozen_dirs = (
            "navigation", "pathfinding", "simulator", "analysis",
            "hazard", "hazard_evolution", "fire_growth",
        )

        for dir_name in frozen_dirs:

            for path in (root / dir_name).glob("*.py"):

                text = path.read_text()

                self.assertIsNone(
                    re.search(r"^\s*(from|import)\s+behavior_library\b", text, re.MULTILINE),
                    f"{dir_name}/{path.name} imports behavior_library/ -- "
                    f"reverses the dependency direction",
                )


if __name__ == "__main__":
    unittest.main()
