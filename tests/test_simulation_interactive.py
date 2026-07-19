import unittest
from pathlib import Path

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.staircase import Staircase
from models.zone import Zone

from navigation.edge import Edge

from scenario import Scenario, ScenarioFire, ScenarioMetadata, ScenarioOccupant
from scenario_runner import run

from behaviour_profile_resolver.registry import DEFAULT_PROFILE_REGISTRY

from simulator.occupant import OccupantState

from simulation_interactive import (
    Action,
    ActionResult,
    InteractiveActionType,
    InteractiveSimulation,
)


# =====================================================
# Fixtures -- mirrors tests/test_scenario_event_executor.py's and
# tests/test_simulation_runtime.py's own make_building()/make_metadata()/
# make_scenario() shape. Two zones (zone-a, zone-b) each lead to their
# own exit; zone-a's path is deliberately much shorter (zone-a/exit-a
# positioned close to the origin, zone-b/exit-b positioned far away)
# so ShortestRouteChoiceStrategy's dijkstra deterministically prefers
# the zone-a/exit-a path by default, without depending on any
# undocumented tie-breaking rule.
# =====================================================


def make_building():

    floor1 = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-start", name="Start", x=0.0, y=0.0, width=2.0, height=2.0),
            Zone(id="zone-a", name="A", x=10.0, y=0.0, width=2.0, height=2.0),
            Zone(id="zone-b", name="B", x=200.0, y=0.0, width=2.0, height=2.0),
        ],
        doors=[
            Door(id="door-a", normally_open=True, zone_a_id="zone-start", zone_b_id="zone-a"),
            Door(id="door-b", normally_open=True, zone_a_id="zone-start", zone_b_id="zone-b"),
        ],
        exits=[
            Exit(id="exit-a", zone_id="zone-a"),
            Exit(id="exit-b", zone_id="zone-b"),
        ],
    )

    return Building(name="Interactive Test Building", id="building-1", floors=[floor1])


def make_stair_building():

    # zone-start2's shortest path is via stair-1 (a single-floor-
    # transition, hence small, fixed travel_distance) to zone-c/exit-c;
    # door-a2/zone-a2/exit-a2 is deliberately far away so it is only
    # ever chosen once stair-1 is closed.

    floor1 = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-start2", name="Start2", x=0.0, y=0.0, width=2.0, height=2.0),
            Zone(id="zone-a2", name="A2", x=1000.0, y=0.0, width=2.0, height=2.0),
        ],
        doors=[
            Door(id="door-a2", normally_open=True, zone_a_id="zone-start2", zone_b_id="zone-a2"),
        ],
        exits=[
            Exit(id="exit-a2", zone_id="zone-a2"),
        ],
        stairs=[
            Staircase(id="stair-1", from_zone_id="zone-start2", to_zone_id="zone-c", to_floor_id="floor-2"),
        ],
    )
    floor2 = Floor(
        name="Upper", id="floor-2",
        zones=[Zone(id="zone-c", name="C", x=0.0, y=0.0, width=2.0, height=2.0)],
        exits=[Exit(id="exit-c", zone_id="zone-c")],
    )

    return Building(name="Interactive Stair Test Building", id="building-2", floors=[floor1, floor2])


def make_metadata(**overrides):

    defaults = dict(
        scenario_id="scn-1", definition_id="def-1", definition_content_hash="hash-abc",
        generation_version="scenario_generator/1", seed=42, created_at="2026-07-14T00:00:00",
    )
    defaults.update(overrides)

    return ScenarioMetadata(**defaults)


def make_occupant(**overrides):

    defaults = dict(
        occupant_id="occ-1", zone_id="zone-start", floor_id="floor-1",
        position=(1.0, 1.0), behaviour_profile_id="Staff_Default",
    )
    defaults.update(overrides)

    return ScenarioOccupant(**defaults)


def make_scenario(**overrides):

    defaults = dict(
        metadata=make_metadata(),
        occupants=(make_occupant(),),
        fire=ScenarioFire(
            ignition_zone_id="zone-b", ignition_floor_id="floor-1",
            fire_profile="Electrical", growth_parameters={"growth_time": 200.0},
        ),
        events=(),
    )
    defaults.update(overrides)

    return Scenario(**defaults)


def make_context(scenario=None, building=None):

    return run(scenario or make_scenario(), building or make_building())


def make_simulation(scenario=None, building=None, dt=1.0):

    return InteractiveSimulation(make_context(scenario=scenario, building=building), dt=dt)


class NoUpfrontMovementTests(unittest.TestCase):

    def test_init_never_calls_simulation_run(self):

        context = make_context()

        call_count = {"n": 0}
        original_run = context.simulation.run

        def spy_run():
            call_count["n"] += 1
            return original_run()

        context.simulation.run = spy_run

        simulation = InteractiveSimulation(context, dt=1.0)

        self.assertEqual(call_count["n"], 0)

        simulation.step()
        simulation.step()

        self.assertEqual(call_count["n"], 0)

    def test_occupant_has_not_moved_before_first_step(self):

        simulation = make_simulation()

        result = simulation._stepper.snapshot_result()
        timeline = result.occupants["occ-1"]

        self.assertEqual(timeline.state, OccupantState.AT_NODE)
        self.assertEqual(timeline.steps, [])


class DoorChangeReroutingTests(unittest.TestCase):

    def test_closing_door_before_departure_reroutes_to_alternate_path(self):

        simulation = make_simulation()

        simulation.queue_action(Action(InteractiveActionType.CLOSE_DOOR, target_id="door-a"))
        step_result = simulation.step()

        self.assertIn("occ-1", step_result.replanned_occupant_ids)

        results = simulation.run_to_completion()
        final_timeline = simulation._stepper.snapshot_result().occupants["occ-1"]

        self.assertEqual(final_timeline.state, OccupantState.ARRIVED)
        self.assertIn("door-b", final_timeline.route.edge_ids)
        self.assertIn("exit-b", final_timeline.route.edge_ids)
        self.assertNotIn("door-a", final_timeline.route.edge_ids)


class ReplanningUnreachableTests(unittest.TestCase):

    # Ground-truth-classification correctness fix (validation Phase 5
    # finding, docs/validation/technical_report.md §6), exercised
    # through simulation_interactive.replanning.replan_occupant() --
    # the second, independent orchestration path that reimplements
    # HumanBehaviorLayer.register()'s own dispatch (see that module's
    # own comment for why it doesn't just call register()). Closing
    # every door out of zone-start before departure strands occ-1 with
    # no route to any exit; the replanned decision must register
    # UNREACHABLE, not STATIONARY.

    def test_closing_every_door_before_departure_replans_to_unreachable(self):

        simulation = make_simulation()

        simulation.queue_action(Action(InteractiveActionType.CLOSE_DOOR, target_id="door-a"))
        simulation.queue_action(Action(InteractiveActionType.CLOSE_DOOR, target_id="door-b"))
        step_result = simulation.step()

        self.assertIn("occ-1", step_result.replanned_occupant_ids)

        final_timeline = simulation._stepper.snapshot_result().occupants["occ-1"]

        self.assertEqual(final_timeline.state, OccupantState.UNREACHABLE)
        self.assertIsNone(final_timeline.arrival_time)


class ExitChangeReroutingTests(unittest.TestCase):

    def test_closing_exit_before_departure_reroutes_to_alternate_path(self):

        simulation = make_simulation()

        simulation.queue_action(Action(InteractiveActionType.CLOSE_EXIT, target_id="exit-a"))
        simulation.step()
        simulation.run_to_completion()

        final_timeline = simulation._stepper.snapshot_result().occupants["occ-1"]

        self.assertEqual(final_timeline.state, OccupantState.ARRIVED)
        self.assertIn("exit-b", final_timeline.route.edge_ids)
        self.assertNotIn("exit-a", final_timeline.route.edge_ids)


class StairChangeReroutingTests(unittest.TestCase):

    def _make_stair_occupant_simulation(self):

        scenario = make_scenario(
            occupants=(make_occupant(occupant_id="occ-1", zone_id="zone-start2"),),
        )

        return make_simulation(scenario=scenario, building=make_stair_building())

    def test_default_route_uses_stair(self):

        simulation = self._make_stair_occupant_simulation()
        simulation.run_to_completion()

        final_timeline = simulation._stepper.snapshot_result().occupants["occ-1"]

        self.assertIn("stair-1", final_timeline.route.edge_ids)

    def test_closing_stair_before_departure_reroutes_through_door(self):

        simulation = self._make_stair_occupant_simulation()

        simulation.queue_action(Action(InteractiveActionType.CLOSE_STAIR, target_id="stair-1"))
        step_result = simulation.step()

        self.assertIn("occ-1", step_result.replanned_occupant_ids)

        stair_edge_present = any(
            edge.edge_type == Edge.STAIR and edge.id == "stair-1"
            for edge in simulation._context.graph.edges
        )
        self.assertFalse(stair_edge_present)

        simulation.run_to_completion()
        final_timeline = simulation._stepper.snapshot_result().occupants["occ-1"]

        self.assertEqual(final_timeline.state, OccupantState.ARRIVED)
        self.assertIn("door-a2", final_timeline.route.edge_ids)
        self.assertIn("exit-a2", final_timeline.route.edge_ids)
        self.assertNotIn("stair-1", final_timeline.route.edge_ids)

    def test_reopening_stair_restores_graph_edge(self):

        simulation = self._make_stair_occupant_simulation()

        simulation.queue_action(Action(InteractiveActionType.CLOSE_STAIR, target_id="stair-1"))
        simulation.step()

        simulation.queue_action(Action(InteractiveActionType.OPEN_STAIR, target_id="stair-1"))
        simulation.step()

        stair_edge_present = any(
            edge.edge_type == Edge.STAIR and edge.id == "stair-1"
            for edge in simulation._context.graph.edges
        )
        self.assertTrue(stair_edge_present)


class RecommendationTests(unittest.TestCase):

    def test_recommend_exit_redirects_reachable_occupant(self):

        simulation = make_simulation()

        simulation.queue_action(
            Action(
                InteractiveActionType.RECOMMEND_EXIT, target_id="zone-start",
                parameters={"exit_id": "exit-b"},
            )
        )
        step_result = simulation.step()

        self.assertIn("occ-1", step_result.replanned_occupant_ids)

        simulation.run_to_completion()
        final_timeline = simulation._stepper.snapshot_result().occupants["occ-1"]

        self.assertEqual(final_timeline.state, OccupantState.ARRIVED)
        self.assertIn("exit-b", final_timeline.route.edge_ids)

    def test_recommend_unreachable_exit_falls_back_gracefully(self):

        simulation = make_simulation()

        simulation.queue_action(
            Action(
                InteractiveActionType.RECOMMEND_EXIT, target_id="zone-start",
                parameters={"exit_id": "exit-does-not-exist"},
            )
        )
        simulation.step()
        simulation.run_to_completion()

        final_timeline = simulation._stepper.snapshot_result().occupants["occ-1"]

        self.assertEqual(final_timeline.state, OccupantState.ARRIVED)
        self.assertIn("exit-a", final_timeline.route.edge_ids)


class BroadcastTests(unittest.TestCase):

    def test_broadcast_shelter_in_place_makes_at_node_occupant_stationary(self):

        simulation = make_simulation()

        simulation.queue_action(
            Action(InteractiveActionType.BROADCAST_SHELTER_IN_PLACE, parameters={"target_zone_ids": ["zone-start"]})
        )
        step_result = simulation.step()

        self.assertIn("occ-1", step_result.replanned_occupant_ids)
        self.assertEqual(
            simulation._stepper.occupant_state("occ-1"), OccupantState.STATIONARY,
        )

    def test_broadcast_evacuate_reverses_shelter_in_place(self):

        simulation = make_simulation()

        simulation.queue_action(
            Action(InteractiveActionType.BROADCAST_SHELTER_IN_PLACE, parameters={"target_zone_ids": ["zone-start"]})
        )
        simulation.step()
        self.assertEqual(simulation._stepper.occupant_state("occ-1"), OccupantState.STATIONARY)

        simulation.queue_action(
            Action(InteractiveActionType.BROADCAST_EVACUATE, parameters={"target_zone_ids": ["zone-start"]})
        )
        simulation.step()
        simulation.run_to_completion()

        final_timeline = simulation._stepper.snapshot_result().occupants["occ-1"]
        self.assertEqual(final_timeline.state, OccupantState.ARRIVED)


class DeployStaffTests(unittest.TestCase):

    def test_deploy_staff_is_recorded_but_not_applied(self):

        simulation = make_simulation()

        simulation.queue_action(Action(InteractiveActionType.DEPLOY_STAFF, target_id="zone-start"))
        step_result = simulation.step()

        self.assertEqual(len(step_result.applied_actions), 1)
        result = step_result.applied_actions[0]
        self.assertIsInstance(result, ActionResult)
        self.assertFalse(result.applied)
        self.assertIsNotNone(result.reason)


class NoMidEdgeReroutingTests(unittest.TestCase):

    def test_closing_door_while_occupant_is_traversing_does_not_affect_them(self):

        simulation = make_simulation(dt=0.5)

        # One small step is enough for occ-1's t=0 TRY_ENTER_EDGE to be
        # processed and admit them onto door-a (duration is several
        # seconds, comfortably longer than a 0.5s step).
        simulation.step()
        self.assertEqual(simulation._stepper.occupant_state("occ-1"), OccupantState.TRAVERSING)

        simulation.queue_action(Action(InteractiveActionType.CLOSE_DOOR, target_id="door-a"))
        step_result = simulation.step()

        self.assertNotIn("occ-1", step_result.replanned_occupant_ids)

        simulation.run_to_completion()
        final_timeline = simulation._stepper.snapshot_result().occupants["occ-1"]

        self.assertEqual(final_timeline.state, OccupantState.ARRIVED)
        self.assertEqual(list(final_timeline.route.edge_ids), ["door-a", "exit-a"])


class StateSnapshotTests(unittest.TestCase):

    def test_snapshot_reflects_occupant_positions_and_engineering_state(self):

        simulation = make_simulation()

        step_result = simulation.step()
        snapshot = step_result.snapshot

        self.assertEqual(snapshot.time, step_result.time)
        self.assertEqual(len(snapshot.occupants), 1)
        self.assertEqual(snapshot.door_states["door-a"], "OPEN")
        self.assertEqual(snapshot.exit_states["exit-a"], "OPEN")

        simulation.queue_action(Action(InteractiveActionType.CLOSE_DOOR, target_id="door-a"))
        step_result = simulation.step()

        self.assertEqual(step_result.snapshot.door_states["door-a"], "LOCKED")

    def test_snapshot_reports_queue_length_at_capacity_limited_door(self):

        floor1 = Floor(
            name="Ground", id="floor-1",
            zones=[
                Zone(id="zone-start", name="Start", x=0.0, y=0.0, width=2.0, height=2.0),
                Zone(id="zone-a", name="A", x=10.0, y=0.0, width=2.0, height=2.0),
            ],
            doors=[
                Door(
                    id="door-a", normally_open=True, width=0.1,
                    zone_a_id="zone-start", zone_b_id="zone-a",
                ),
            ],
            exits=[Exit(id="exit-a", zone_id="zone-a")],
        )
        building = Building(name="Capacity Test", id="building-3", floors=[floor1])

        scenario = make_scenario(
            occupants=(
                make_occupant(occupant_id="occ-1"),
                make_occupant(occupant_id="occ-2"),
            ),
        )

        simulation = make_simulation(scenario=scenario, building=building, dt=0.5)
        step_result = simulation.step()

        total_queue_length = sum(step_result.snapshot.edge_queue_lengths.values())
        total_congestion = sum(step_result.snapshot.edge_congestion.values())

        self.assertEqual(total_congestion, 1)
        self.assertEqual(total_queue_length, 1)


class DeterministicReplayTests(unittest.TestCase):

    def _run_scripted_scenario(self):

        simulation = make_simulation(dt=1.0)

        simulation.queue_action(Action(InteractiveActionType.CLOSE_DOOR, target_id="door-a"))
        simulation.step()

        simulation.queue_action(
            Action(InteractiveActionType.RECOMMEND_EXIT, target_id="zone-start", parameters={"exit_id": "exit-b"})
        )
        simulation.step()

        simulation.run_to_completion()

        return simulation._stepper.snapshot_result()

    def test_identical_action_sequence_produces_identical_outcome(self):

        first = self._run_scripted_scenario()
        second = self._run_scripted_scenario()

        self.assertEqual(
            first.occupants["occ-1"].arrival_time,
            second.occupants["occ-1"].arrival_time,
        )
        self.assertEqual(
            list(first.occupants["occ-1"].route.edge_ids),
            list(second.occupants["occ-1"].route.edge_ids),
        )
        self.assertEqual(first.occupants["occ-1"].state, second.occupants["occ-1"].state)


class StrategyReuseTests(unittest.TestCase):

    def test_replan_uses_the_same_strategy_instances_the_registry_resolved(self):

        simulation = make_simulation()

        registration = simulation._route_manager._registrations["occ-1"]

        template = DEFAULT_PROFILE_REGISTRY["Staff_Default"]

        self.assertIs(registration.route_choice_strategy._base, template.route_choice_strategy)
        self.assertIs(registration.decision_strategy._base, template.decision_strategy)
        self.assertIs(registration.pre_movement_strategy, template.pre_movement_strategy)


class IndependenceTests(unittest.TestCase):

    def test_package_never_reaches_into_pathfinding_internals(self):

        package_dir = Path(__file__).resolve().parent.parent / "simulation_interactive"

        for path in package_dir.glob("*.py"):

            text = path.read_text(encoding="utf-8")

            self.assertNotIn("_search(", text, msg=f"{path} touches PathfindingEngine internals")
            self.assertNotIn("._relax(", text, msg=f"{path} touches PathfindingEngine internals")

    def test_package_never_mutates_coordinator_bookkeeping_directly(self):

        package_dir = Path(__file__).resolve().parent.parent / "simulation_interactive"
        forbidden_assignments = (
            "._occupants[", "._edge_occupancy[", "._edge_queues[",
            "._node_occupancy[", "._generation[", "._event_heap",
        )

        for path in package_dir.glob("*.py"):

            if path.name != "movement_stepper.py":
                text = path.read_text(encoding="utf-8")
                for forbidden in forbidden_assignments:
                    self.assertNotIn(
                        forbidden, text,
                        msg=f"{path} reaches into MultiAgentSimulation internals directly",
                    )


if __name__ == "__main__":
    unittest.main()
