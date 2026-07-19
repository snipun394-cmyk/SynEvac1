import unittest

from dataclasses import FrozenInstanceError

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.zone import Zone

from navigation.graph_builder import NavigationGraphGenerator

from pathfinding.engine import PathfindingEngine

from simulator.coordinator import MultiAgentSimulation
from simulator.decision import ActionType, BehaviorDecision
from simulator.occupant import OccupantState

from behavior.context import DecisionContext
from behavior.group import BehaviorGroup
from behavior.intent import ActionIntent, AlwaysEvacuateDecisionStrategy, DecisionStrategy
from behavior.orchestrator import HumanBehaviorLayer
from behavior.pre_movement import NoPreMovementDelay, PreMovementDelayStrategy
from behavior.profile import BehaviorProfile, Role
from behavior.route_choice import RouteChoice, RouteChoiceStrategy, ShortestRouteChoiceStrategy


def make_zone(name, **kwargs):

    fields = dict(x=0.0, y=0.0, width=2.0, height=2.0)
    fields.update(kwargs)

    return Zone(name=name, **fields)


def build_two_zone_building():

    building = Building(name="B")
    floor = building.create_floor(name="Ground Floor")

    zone_a = make_zone("A", x=0.0, y=0.0)
    zone_b = make_zone("B", x=5.0, y=0.0)
    floor.add_zone(zone_a)
    floor.add_zone(zone_b)

    door = Door(
        name="D1", zone_a_id=zone_a.id, zone_b_id=zone_b.id, floor_id=floor.id,
    )
    floor.add_door(door)
    exit_obj = Exit(name="Ex", zone_id=zone_b.id, floor_id=floor.id)
    floor.add_exit(exit_obj)

    graph = NavigationGraphGenerator().build(building)
    engine = PathfindingEngine(graph)

    return building, floor, zone_a, zone_b, door, exit_obj, engine


class BehaviorProfileTests(unittest.TestCase):

    def test_defaults_are_sensible(self):

        profile = BehaviorProfile(occupant_id="p1")

        self.assertIsNone(profile.walking_speed)
        self.assertEqual(profile.familiarity, {})
        self.assertEqual(profile.compliance_level, 1.0)
        self.assertEqual(profile.role, Role.INDEPENDENT)
        self.assertIsNone(profile.group_id)
        self.assertEqual(profile.traits, {})

    def test_profile_is_mutable_unlike_behavior_decision(self):

        # BehaviorProfile was never asked to become immutable -- only
        # BehaviorDecision was. Confirms it still behaves as ordinary,
        # mutable data.
        profile = BehaviorProfile(occupant_id="p1")
        profile.compliance_level = 0.2
        self.assertEqual(profile.compliance_level, 0.2)


class BehaviorDecisionImmutabilityTests(unittest.TestCase):

    def test_field_reassignment_raises(self):

        decision = BehaviorDecision(
            occupant_id="p1", action_type=ActionType.EVACUATE, start_id="z1",
        )

        with self.assertRaises(FrozenInstanceError):
            decision.goal_id = "outside"

    def test_metadata_mapping_is_truly_read_only(self):

        decision = BehaviorDecision(
            occupant_id="p1", action_type=ActionType.EVACUATE, start_id="z1",
            metadata={"target": "z9"},
        )

        with self.assertRaises(TypeError):
            decision.metadata["target"] = "z8"

    def test_changed_behavior_is_a_new_instance_not_a_mutation(self):

        first = BehaviorDecision(
            occupant_id="p1", action_type=ActionType.WAIT, start_id="z1",
        )
        second = BehaviorDecision(
            occupant_id="p1", action_type=ActionType.EVACUATE,
            start_id="z1", goal_id="outside",
        )

        self.assertIsNot(first, second)
        # The first decision is completely untouched by the second's
        # creation -- this is the "timeline of immutable snapshots"
        # property the design relies on.
        self.assertEqual(first.action_type, ActionType.WAIT)
        self.assertIsNone(first.goal_id)


class DecisionStrategyTests(unittest.TestCase):

    def test_always_evacuate_produces_a_movement_intent(self):

        context = DecisionContext(
            graph=None, engine=None, profile=BehaviorProfile(occupant_id="p1"),
            start_id="z1",
        )

        intent = AlwaysEvacuateDecisionStrategy().decide(context)

        self.assertEqual(intent.action_type, ActionType.EVACUATE)
        self.assertTrue(intent.requires_movement)
        self.assertEqual(intent.occupant_id, "p1")

    def test_custom_strategy_can_produce_a_non_movement_intent(self):

        class AlwaysWaitStrategy(DecisionStrategy):
            def decide(self, context):
                return ActionIntent(
                    occupant_id=context.profile.occupant_id,
                    action_type=ActionType.WAIT,
                    requires_movement=False,
                )

        context = DecisionContext(
            graph=None, engine=None, profile=BehaviorProfile(occupant_id="p1"),
            start_id="z1",
        )

        intent = AlwaysWaitStrategy().decide(context)

        self.assertEqual(intent.action_type, ActionType.WAIT)
        self.assertFalse(intent.requires_movement)


class RouteChoiceStrategyTests(unittest.TestCase):

    def test_shortest_route_choice_matches_nearest_exit(self):

        _, _, zone_a, _, _, exit_obj, engine = build_two_zone_building()

        context = DecisionContext(
            graph=engine.graph, engine=engine,
            profile=BehaviorProfile(occupant_id="p1"), start_id=zone_a.id,
        )

        choice = ShortestRouteChoiceStrategy().choose(context)
        expected = engine.nearest_exit(zone_a.id)

        self.assertEqual(choice.goal_id, expected.goal.id)
        self.assertEqual(choice.route.edge_ids, expected.edge_ids)


class PreMovementDelayStrategyTests(unittest.TestCase):

    def test_no_delay_returns_zero(self):

        context = DecisionContext(
            graph=None, engine=None, profile=BehaviorProfile(occupant_id="p1"),
            start_id="z1",
        )

        self.assertEqual(NoPreMovementDelay().delay(context), 0.0)


class HumanBehaviorLayerOrchestrationTests(unittest.TestCase):

    def setUp(self):

        (
            self.building, self.floor, self.zone_a, self.zone_b,
            self.door, self.exit_obj, self.engine,
        ) = build_two_zone_building()

    def test_default_strategies_match_plain_evacuate(self):

        sim = MultiAgentSimulation(self.engine)
        layer = HumanBehaviorLayer(sim)

        layer.register(
            self.zone_a.id,
            BehaviorProfile(occupant_id="alice"),
            decision_strategy=AlwaysEvacuateDecisionStrategy(),
        )

        result = sim.run()

        expected = self.engine.nearest_exit(self.zone_a.id)
        self.assertEqual(result.occupants["alice"].traversed_edge_ids, expected.edge_ids)
        self.assertEqual(result.occupants["alice"].state, OccupantState.ARRIVED)

    def test_wait_decision_strategy_produces_a_stationary_occupant(self):

        class AlwaysWaitStrategy(DecisionStrategy):
            def decide(self, context):
                return ActionIntent(
                    occupant_id=context.profile.occupant_id,
                    action_type=ActionType.WAIT,
                    requires_movement=False,
                )

        sim = MultiAgentSimulation(self.engine)
        layer = HumanBehaviorLayer(sim)

        layer.register(
            self.zone_a.id,
            BehaviorProfile(occupant_id="bob"),
            decision_strategy=AlwaysWaitStrategy(),
        )

        result = sim.run()

        self.assertEqual(result.occupants["bob"].state, OccupantState.STATIONARY)
        self.assertIsNone(result.occupants["bob"].arrival_time)

    def test_movement_required_but_unreachable_registers_unreachable_not_stationary(self):

        # Ground-truth-classification correctness fix (validation Phase
        # 5 finding, docs/validation/technical_report.md §6): a zone
        # with no route to any exit under the default
        # ShortestRouteChoiceStrategy used to register as STATIONARY --
        # indistinguishable from a genuine WAIT/IGNORE decision -- even
        # though intent.requires_movement was True. It must now
        # register as UNREACHABLE and be counted in
        # unreachable_occupant_ids.

        isolated_building = Building(name="Isolated")
        isolated_floor = isolated_building.create_floor(name="F")
        isolated_zone = make_zone("Trapped", x=0.0, y=0.0)
        isolated_floor.add_zone(isolated_zone)

        isolated_graph = NavigationGraphGenerator().build(isolated_building)
        isolated_engine = PathfindingEngine(isolated_graph)

        sim = MultiAgentSimulation(isolated_engine)
        layer = HumanBehaviorLayer(sim)

        layer.register(
            isolated_zone.id,
            BehaviorProfile(occupant_id="bob"),
            decision_strategy=AlwaysEvacuateDecisionStrategy(),
        )

        result = sim.run()

        self.assertEqual(result.occupants["bob"].state, OccupantState.UNREACHABLE)
        self.assertIsNone(result.occupants["bob"].arrival_time)
        self.assertIn("bob", result.unreachable_occupant_ids)

    def test_custom_pre_movement_delay_shifts_depart_time(self):

        class FixedDelay(PreMovementDelayStrategy):
            def delay(self, context):
                return 10.0

        sim = MultiAgentSimulation(self.engine)
        layer = HumanBehaviorLayer(sim)

        layer.register(
            self.zone_a.id,
            BehaviorProfile(occupant_id="carol"),
            decision_strategy=AlwaysEvacuateDecisionStrategy(),
            pre_movement_strategy=FixedDelay(),
        )

        result = sim.run()

        # First step's start_time reflects the 10s pre-movement delay.
        self.assertEqual(result.occupants["carol"].steps[0].start_time, 10.0)

    def test_follower_copies_leader_route_via_decisions_so_far(self):

        group = BehaviorGroup(group_id="family-1", leader_occupant_id="leader")

        class CopyLeaderRouteChoice(RouteChoiceStrategy):
            def __init__(self, leader_id):
                self.leader_id = leader_id

            def choose(self, context):
                leader_decision = context.decisions_so_far[self.leader_id]
                return RouteChoice(
                    goal_id=leader_decision.goal_id, route=leader_decision.route,
                )

        sim = MultiAgentSimulation(self.engine)
        layer = HumanBehaviorLayer(sim)

        layer.register(
            self.zone_a.id,
            BehaviorProfile(occupant_id="leader", role=Role.LEADER, group_id=group.group_id),
            decision_strategy=AlwaysEvacuateDecisionStrategy(),
        )
        layer.register(
            self.zone_a.id,
            BehaviorProfile(occupant_id="follower", role=Role.FOLLOWER, group_id=group.group_id),
            decision_strategy=AlwaysEvacuateDecisionStrategy(),
            route_choice_strategy=CopyLeaderRouteChoice(group.leader_occupant_id),
        )

        result = sim.run()

        self.assertEqual(
            result.occupants["leader"].traversed_edge_ids,
            result.occupants["follower"].traversed_edge_ids,
        )

    def test_register_uses_submit_decision_not_add_occupant_directly(self):

        # Behavior Layer's only point of contact with Simulation is
        # submit_decision() -- confirmed by inspecting the source, the
        # same way independence from engineering models is confirmed
        # elsewhere.
        import pathlib

        text = (
            pathlib.Path(__file__).resolve().parent.parent
            / "behavior" / "orchestrator.py"
        ).read_text()

        self.assertIn("submit_decision", text)
        self.assertNotIn(".add_occupant(", text)

    def test_walking_speed_from_profile_is_respected(self):

        sim = MultiAgentSimulation(self.engine)
        layer = HumanBehaviorLayer(sim)

        layer.register(
            self.zone_a.id,
            BehaviorProfile(occupant_id="fast", walking_speed=5.0),
            decision_strategy=AlwaysEvacuateDecisionStrategy(),
        )

        default_sim = MultiAgentSimulation(self.engine)
        default_layer = HumanBehaviorLayer(default_sim)
        default_layer.register(
            self.zone_a.id,
            BehaviorProfile(occupant_id="default"),
            decision_strategy=AlwaysEvacuateDecisionStrategy(),
        )

        fast_result = sim.run()
        default_result = default_sim.run()

        self.assertLess(
            fast_result.occupants["fast"].arrival_time,
            default_result.occupants["default"].arrival_time,
        )


class BehaviorLayerIndependenceTests(unittest.TestCase):

    def test_behavior_package_never_touches_reference_or_engineering_models(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "behavior"

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertNotIn(
                ".reference", text, f"{path.name} touches .reference directly"
            )
            self.assertIsNone(
                re.search(r"^\s*(from|import)\s+(models|designer)\b", text, re.MULTILINE),
                f"{path.name} imports models/designer directly",
            )

    def test_behavior_package_never_imports_simulator_internals(self):

        import pathlib

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "behavior"

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertNotIn("_search", text)
            self.assertNotIn("_relax", text)
            self.assertNotIn("_process_next_event", text)
            self.assertNotIn("_register", text)

    def test_dependency_direction_simulator_never_imports_behavior(self):

        import pathlib
        import re

        simulator_dir = pathlib.Path(__file__).resolve().parent.parent / "simulator"

        for path in simulator_dir.glob("*.py"):

            text = path.read_text()

            self.assertIsNone(
                re.search(r"^\s*(from|import)\s+behavior\b", text, re.MULTILINE),
                f"{path.name} imports behavior/ -- reverses the dependency direction",
            )


if __name__ == "__main__":
    unittest.main()
