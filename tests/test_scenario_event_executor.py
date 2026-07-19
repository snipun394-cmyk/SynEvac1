import unittest

from models.building import Building
from models.camera import Camera
from models.detector import Detector
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.obstacle import Obstacle
from models.staircase import Staircase
from models.zone import Zone

from navigation.edge import Edge

from scenario import Scenario, ScenarioEvent, ScenarioFire, ScenarioMetadata
from scenario_runner import run

from scenario_event_executor import (
    EVENT_HANDLERS,
    ScenarioEventExecutor,
    UnsupportedEventTargetTypeError,
    execute_event,
)


def make_building():

    floor1 = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-1", name="Lobby", x=0.0, y=0.0, width=10.0, height=8.0),
            Zone(id="zone-2", name="Office", x=20.0, y=0.0, width=6.0, height=6.0),
        ],
        doors=[Door(id="door-1", normally_open=True, zone_a_id="zone-1", zone_b_id="zone-2")],
        exits=[Exit(id="exit-1", zone_id="zone-1")],
        obstacles=[Obstacle(id="obs-1", active=False)],
        cameras=[Camera(id="cam-1", active=True)],
        detectors=[Detector(id="det-1", active=True)],
        stairs=[Staircase(id="stair-1", from_zone_id="zone-1", to_zone_id="zone-3", to_floor_id="floor-2")],
    )
    floor2 = Floor(name="Upper", id="floor-2", zones=[Zone(id="zone-3", name="Attic")])

    return Building(name="Test Building", id="building-1", floors=[floor1, floor2])


def make_metadata(**overrides):

    defaults = dict(
        scenario_id="scn-1", definition_id="def-1", definition_content_hash="hash-abc",
        generation_version="scenario_generator/1", seed=42, created_at="2026-07-13T00:00:00",
    )
    defaults.update(overrides)

    return ScenarioMetadata(**defaults)


def make_event(**overrides):

    defaults = dict(
        event_id="evt-1", target_type="door", target_id="door-1",
        event_type="close", time=10.0, parameters={"state": "LOCKED"},
    )
    defaults.update(overrides)

    return ScenarioEvent(**defaults)


def make_scenario(**overrides):

    defaults = dict(
        metadata=make_metadata(),
        fire=ScenarioFire(
            ignition_zone_id="zone-2", ignition_floor_id="floor-1",
            fire_profile="Electrical", growth_parameters={"growth_time": 200.0},
        ),
        events=(make_event(),),
    )
    defaults.update(overrides)

    return Scenario(**defaults)


def make_context(scenario=None, building=None):

    return run(scenario or make_scenario(), building or make_building())


class EventOrderingTests(unittest.TestCase):

    def test_executor_trusts_pre_sorted_input_rather_than_re_sorting(self):

        # scenario_validator's own Event Validation is what guarantees
        # scheduled_events arrives time-ordered upstream (§8 of the
        # architecture document: "the Executor performs no sorting and
        # no re-validation of order"). Feeding it deliberately
        # out-of-order data here proves that directly: the executor
        # walks the tuple exactly as given, never reordering by time,
        # so a caller bypassing the normal Validator-guaranteed path
        # gets exactly what it asked for, not a silently "corrected"
        # result.
        scenario = make_scenario(
            events=(
                make_event(event_id="evt-late", target_id="door-1", time=50.0),
                make_event(
                    event_id="evt-early", target_type="camera", target_id="cam-1",
                    time=10.0, parameters={"availability": "FAILED"},
                ),
            ),
        )
        context = make_context(scenario=scenario)

        executor = ScenarioEventExecutor(context)
        fired = executor.advance_to(100.0)

        self.assertEqual([e.event_id for e in fired], ["evt-late", "evt-early"])

    def test_pre_sorted_events_execute_in_list_order(self):

        scenario = make_scenario(
            events=(
                make_event(
                    event_id="evt-1", target_type="camera", target_id="cam-1",
                    time=10.0, parameters={"availability": "FAILED"},
                ),
                make_event(event_id="evt-2", target_id="door-1", time=20.0),
            ),
        )
        context = make_context(scenario=scenario)
        executor = ScenarioEventExecutor(context)

        fired = executor.advance_to(100.0)

        self.assertEqual([e.event_id for e in fired], ["evt-1", "evt-2"])


class ExecutionTimingTests(unittest.TestCase):

    def test_event_does_not_fire_before_its_time(self):

        context = make_context()
        executor = ScenarioEventExecutor(context)

        fired = executor.advance_to(5.0)

        self.assertEqual(fired, ())
        self.assertFalse(context.building.floors[0].doors[0].locked)

    def test_event_fires_exactly_at_its_time(self):

        context = make_context()
        executor = ScenarioEventExecutor(context)

        fired = executor.advance_to(10.0)

        self.assertEqual([e.event_id for e in fired], ["evt-1"])
        self.assertTrue(context.building.floors[0].doors[0].locked)

    def test_event_fires_when_target_time_passes_it(self):

        context = make_context()
        executor = ScenarioEventExecutor(context)

        fired = executor.advance_to(50.0)

        self.assertEqual([e.event_id for e in fired], ["evt-1"])

    def test_already_fired_event_does_not_fire_again_on_a_later_advance(self):

        context = make_context()
        executor = ScenarioEventExecutor(context)

        first = executor.advance_to(10.0)
        second = executor.advance_to(20.0)

        self.assertEqual([e.event_id for e in first], ["evt-1"])
        self.assertEqual(second, ())

    def test_advancing_backwards_raises(self):

        context = make_context()
        executor = ScenarioEventExecutor(context)

        executor.advance_to(50.0)

        with self.assertRaises(ValueError):
            executor.advance_to(10.0)

    def test_repeated_advance_to_the_same_time_is_a_harmless_no_op(self):

        context = make_context()
        executor = ScenarioEventExecutor(context)

        executor.advance_to(10.0)
        second = executor.advance_to(10.0)

        self.assertEqual(second, ())

    def test_is_complete_reflects_whether_every_event_has_fired(self):

        context = make_context()
        executor = ScenarioEventExecutor(context)

        self.assertFalse(executor.is_complete)

        executor.advance_to(10.0)

        self.assertTrue(executor.is_complete)

    def test_remaining_events_shrinks_as_events_fire(self):

        scenario = make_scenario(
            events=(make_event(event_id="evt-1", time=10.0), make_event(event_id="evt-2", time=20.0)),
        )
        context = make_context(scenario=scenario)
        executor = ScenarioEventExecutor(context)

        self.assertEqual(len(executor.remaining_events), 2)

        executor.advance_to(10.0)

        self.assertEqual(len(executor.remaining_events), 1)
        self.assertEqual(executor.remaining_events[0].event_id, "evt-2")


class SimultaneousEventTests(unittest.TestCase):

    def test_two_events_at_the_same_time_both_fire_on_one_advance(self):

        scenario = make_scenario(
            events=(
                make_event(event_id="evt-door", target_id="door-1", time=30.0),
                make_event(
                    event_id="evt-camera", target_type="camera", target_id="cam-1",
                    time=30.0, parameters={"availability": "FAILED"},
                ),
            ),
        )
        context = make_context(scenario=scenario)
        executor = ScenarioEventExecutor(context)

        fired = executor.advance_to(30.0)

        self.assertEqual({e.event_id for e in fired}, {"evt-door", "evt-camera"})
        self.assertTrue(context.building.floors[0].doors[0].locked)
        self.assertFalse(context.building.floors[0].cameras[0].active)

    def test_simultaneous_events_apply_in_list_order(self):

        scenario = make_scenario(
            events=(
                make_event(
                    event_id="evt-1", target_type="camera", target_id="cam-1",
                    time=15.0, parameters={"availability": "FAILED"},
                ),
                make_event(event_id="evt-2", target_id="door-1", time=15.0),
            ),
        )
        context = make_context(scenario=scenario)
        executor = ScenarioEventExecutor(context)

        fired = executor.advance_to(15.0)

        self.assertEqual([e.event_id for e in fired], ["evt-1", "evt-2"])


class EngineeringObjectUpdateTests(unittest.TestCase):

    def test_door_locked_event(self):

        context = make_context()
        ScenarioEventExecutor(context).advance_to(10.0)

        door = context.building.floors[0].doors[0]
        self.assertTrue(door.locked)
        self.assertFalse(door.normally_open)

    def test_door_open_event(self):

        scenario = make_scenario(events=(make_event(parameters={"state": "OPEN"}),))
        context = make_context(scenario=scenario)
        ScenarioEventExecutor(context).advance_to(10.0)

        door = context.building.floors[0].doors[0]
        self.assertTrue(door.normally_open)
        self.assertFalse(door.locked)

    def test_exit_close_event(self):

        scenario = make_scenario(
            events=(make_event(target_type="exit", target_id="exit-1", parameters={"is_open": False}),),
        )
        context = make_context(scenario=scenario)
        ScenarioEventExecutor(context).advance_to(10.0)

        self.assertTrue(context.building.floors[0].exits[0].is_blocked)

    def test_exit_open_event(self):

        scenario = make_scenario(
            events=(make_event(target_type="exit", target_id="exit-1", parameters={"is_open": True}),),
        )
        context = make_context(scenario=scenario)
        ScenarioEventExecutor(context).advance_to(10.0)

        self.assertFalse(context.building.floors[0].exits[0].is_blocked)

    def test_obstacle_appears_event(self):

        scenario = make_scenario(
            events=(
                make_event(
                    target_type="obstacle", target_id="obs-1", parameters={"presence": "ACTIVE"},
                ),
            ),
        )
        context = make_context(scenario=scenario)
        ScenarioEventExecutor(context).advance_to(10.0)

        self.assertTrue(context.building.floors[0].obstacles[0].active)

    def test_obstacle_disappears_event(self):

        building = make_building()
        building.floors[0].obstacles[0].active = True

        scenario = make_scenario(
            events=(
                make_event(
                    target_type="obstacle", target_id="obs-1", parameters={"presence": "INACTIVE"},
                ),
            ),
        )
        context = make_context(scenario=scenario, building=building)
        ScenarioEventExecutor(context).advance_to(10.0)

        self.assertFalse(context.building.floors[0].obstacles[0].active)

    def test_camera_failure_event(self):

        scenario = make_scenario(
            events=(
                make_event(
                    target_type="camera", target_id="cam-1", parameters={"availability": "FAILED"},
                ),
            ),
        )
        context = make_context(scenario=scenario)
        ScenarioEventExecutor(context).advance_to(10.0)

        self.assertFalse(context.building.floors[0].cameras[0].active)

    def test_camera_restored_event(self):

        building = make_building()
        building.floors[0].cameras[0].active = False

        scenario = make_scenario(
            events=(
                make_event(
                    target_type="camera", target_id="cam-1", parameters={"availability": "AVAILABLE"},
                ),
            ),
        )
        context = make_context(scenario=scenario, building=building)
        ScenarioEventExecutor(context).advance_to(10.0)

        self.assertTrue(context.building.floors[0].cameras[0].active)

    def test_detector_failure_event(self):

        scenario = make_scenario(
            events=(
                make_event(
                    target_type="detector", target_id="det-1", parameters={"availability": "FAILED"},
                ),
            ),
        )
        context = make_context(scenario=scenario)
        ScenarioEventExecutor(context).advance_to(10.0)

        self.assertFalse(context.building.floors[0].detectors[0].active)

    def test_unreferenced_target_id_is_skipped_not_raised(self):

        scenario = make_scenario(events=(make_event(target_id="door-ghost"),))
        context = make_context(scenario=scenario)

        # Should not raise.
        fired = ScenarioEventExecutor(context).advance_to(10.0)
        self.assertEqual([e.event_id for e in fired], ["evt-1"])

    def test_original_building_is_unaffected_by_event_execution(self):

        building = make_building()
        original_locked = building.floors[0].doors[0].locked

        context = make_context(building=building)
        ScenarioEventExecutor(context).advance_to(10.0)

        self.assertEqual(building.floors[0].doors[0].locked, original_locked)
        self.assertTrue(context.building.floors[0].doors[0].locked)


class StairAvailabilityEventTests(unittest.TestCase):

    def _stair_edge_ids(self, context):

        return [e.id for e in context.graph.edges if e.edge_type == Edge.STAIR]

    def test_stair_closure_removes_the_edge(self):

        scenario = make_scenario(
            events=(
                make_event(
                    target_type="stair", target_id="stair-1", time=10.0,
                    parameters={"availability": "CLOSED"},
                ),
            ),
        )
        context = make_context(scenario=scenario)

        self.assertEqual(self._stair_edge_ids(context), ["stair-1"])

        ScenarioEventExecutor(context).advance_to(10.0)

        self.assertEqual(self._stair_edge_ids(context), [])

    def test_stair_reopening_restores_the_edge(self):

        scenario = make_scenario(
            events=(
                make_event(
                    event_id="evt-close", target_type="stair", target_id="stair-1",
                    time=10.0, parameters={"availability": "CLOSED"},
                ),
                make_event(
                    event_id="evt-open", target_type="stair", target_id="stair-1",
                    time=20.0, parameters={"availability": "AVAILABLE"},
                ),
            ),
        )
        context = make_context(scenario=scenario)
        executor = ScenarioEventExecutor(context)

        executor.advance_to(10.0)
        self.assertEqual(self._stair_edge_ids(context), [])

        executor.advance_to(20.0)
        self.assertEqual(self._stair_edge_ids(context), ["stair-1"])

    def test_reopening_an_already_open_stair_is_a_harmless_no_op(self):

        scenario = make_scenario(
            events=(
                make_event(
                    target_type="stair", target_id="stair-1", time=10.0,
                    parameters={"availability": "AVAILABLE"},
                ),
            ),
        )
        context = make_context(scenario=scenario)

        ScenarioEventExecutor(context).advance_to(10.0)

        self.assertEqual(self._stair_edge_ids(context), ["stair-1"])

    def test_stair_reopening_produces_a_usable_edge(self):

        scenario = make_scenario(
            events=(
                make_event(
                    event_id="evt-close", target_type="stair", target_id="stair-1",
                    time=10.0, parameters={"availability": "CLOSED"},
                ),
                make_event(
                    event_id="evt-open", target_type="stair", target_id="stair-1",
                    time=20.0, parameters={"availability": "AVAILABLE"},
                ),
            ),
        )
        context = make_context(scenario=scenario)
        executor = ScenarioEventExecutor(context)

        executor.advance_to(20.0)

        restored_edge = next(e for e in context.graph.edges if e.id == "stair-1")

        self.assertEqual(restored_edge.from_node, "zone-1")
        self.assertEqual(restored_edge.to_node, "zone-3")

    def test_other_stairs_are_unaffected_by_one_stairs_closure(self):

        floor1 = make_building().floors[0]
        floor1.stairs.append(
            Staircase(id="stair-2", from_zone_id="zone-2", to_zone_id="zone-3", to_floor_id="floor-2"),
        )
        building = Building(name="B", id="b2", floors=[floor1, make_building().floors[1]])

        scenario = make_scenario(
            events=(
                make_event(
                    target_type="stair", target_id="stair-1", time=10.0,
                    parameters={"availability": "CLOSED"},
                ),
            ),
        )
        context = make_context(scenario=scenario, building=building)
        ScenarioEventExecutor(context).advance_to(10.0)

        stair_ids = self._stair_edge_ids(context)
        self.assertNotIn("stair-1", stair_ids)
        self.assertIn("stair-2", stair_ids)


class RouteStaticLimitationTests(unittest.TestCase):

    # Executes the honest boundary the architecture document leads
    # with (§2): a door/stair state change during simulation must not
    # alter routing behavior -- no dynamic replanning exists, and this
    # package must not introduce any.

    def test_executing_events_never_touches_simulation_or_engine(self):

        scenario = make_scenario(
            events=(
                make_event(time=10.0),
                make_event(
                    event_id="evt-2", target_type="stair", target_id="stair-1",
                    time=20.0, parameters={"availability": "CLOSED"},
                ),
            ),
        )
        context = make_context(scenario=scenario)

        simulation_before = context.simulation
        engine_before = context.engine

        ScenarioEventExecutor(context).advance_to(20.0)

        self.assertIs(context.simulation, simulation_before)
        self.assertIs(context.engine, engine_before)
        self.assertEqual(context.simulation._occupants, {})


class DispatchTests(unittest.TestCase):

    def test_execute_event_dispatches_by_target_type(self):

        context = make_context()
        execute_event(context, make_event())

        self.assertTrue(context.building.floors[0].doors[0].locked)

    def test_every_frozen_event_type_has_a_registered_handler(self):

        for target_type in ("door", "exit", "obstacle", "camera", "detector", "stair"):
            self.assertIn(target_type, EVENT_HANDLERS)

    def test_unsupported_target_type_raises(self):

        context = make_context()
        event = make_event(target_type="ai_action", target_id="agent-1", parameters={})

        with self.assertRaises(UnsupportedEventTargetTypeError):
            execute_event(context, event)

    def test_unsupported_target_type_error_names_the_type_and_event(self):

        context = make_context()
        event = make_event(
            event_id="evt-mystery", target_type="firefighter_action", parameters={},
        )

        with self.assertRaises(UnsupportedEventTargetTypeError) as ctx:
            execute_event(context, event)

        self.assertIn("firefighter_action", str(ctx.exception))
        self.assertIn("evt-mystery", str(ctx.exception))

    def test_advance_to_propagates_unsupported_target_type_errors(self):

        scenario = make_scenario(
            events=(make_event(target_type="ai_action", target_id="agent-1", parameters={}),),
        )
        context = make_context(scenario=scenario)

        with self.assertRaises(UnsupportedEventTargetTypeError):
            ScenarioEventExecutor(context).advance_to(10.0)


class DeterminismAndReplayTests(unittest.TestCase):

    def test_two_independent_executors_over_the_same_context_shape_produce_identical_state(self):

        scenario = make_scenario(
            events=(
                make_event(event_id="evt-1", time=10.0),
                make_event(
                    event_id="evt-2", target_type="camera", target_id="cam-1",
                    time=20.0, parameters={"availability": "FAILED"},
                ),
            ),
        )
        building = make_building()

        first_context = make_context(scenario=scenario, building=building)
        second_context = make_context(scenario=scenario, building=building)

        ScenarioEventExecutor(first_context).advance_to(100.0)
        ScenarioEventExecutor(second_context).advance_to(100.0)

        self.assertEqual(
            first_context.building.floors[0].doors[0].locked,
            second_context.building.floors[0].doors[0].locked,
        )
        self.assertEqual(
            first_context.building.floors[0].cameras[0].active,
            second_context.building.floors[0].cameras[0].active,
        )

    def test_stepping_in_smaller_increments_produces_the_same_end_state_as_one_big_step(self):

        scenario = make_scenario(
            events=(
                make_event(event_id="evt-1", time=10.0),
                make_event(
                    event_id="evt-2", target_type="camera", target_id="cam-1",
                    time=20.0, parameters={"availability": "FAILED"},
                ),
            ),
        )
        building = make_building()

        stepped_context = make_context(scenario=scenario, building=building)
        stepped_executor = ScenarioEventExecutor(stepped_context)
        for t in (5.0, 10.0, 15.0, 20.0, 25.0):
            stepped_executor.advance_to(t)

        one_shot_context = make_context(scenario=scenario, building=building)
        ScenarioEventExecutor(one_shot_context).advance_to(25.0)

        self.assertEqual(
            stepped_context.building.floors[0].doors[0].locked,
            one_shot_context.building.floors[0].doors[0].locked,
        )
        self.assertEqual(
            stepped_context.building.floors[0].cameras[0].active,
            one_shot_context.building.floors[0].cameras[0].active,
        )


class ScenarioEventExecutorPackageDependencyDirectionTests(unittest.TestCase):

    def test_package_never_imports_forbidden_packages(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "scenario_event_executor"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(scenario_generator|scenario_validator|scenario_pipeline|behavior\b|"
            r"behavior_library|hazard_evolution|fire_growth|ai_decision|perception|"
            r"sensors|occupancy|rl|sandbox|designer|random)\b"
        )

        for path in sorted(package_dir.glob("*.py")):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"scenario_event_executor/{path.name} imports a package the "
                f"architecture (docs/architecture/scenario_event_execution.md §15) "
                f"forbids",
            )

    @staticmethod
    def _non_comment_code(text):

        # Strips comments so this structural check asserts against
        # actual code, not the explanatory prose that deliberately
        # names the very things it forbids (e.g. "never touches
        # MultiAgentSimulation") to document *why* this module stays
        # within its boundary.
        return "\n".join(line.split("#", 1)[0] for line in text.splitlines())

    def test_package_never_touches_simulator_or_behavior_apis(self):

        import pathlib

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "scenario_event_executor"

        for path in sorted(package_dir.glob("*.py")):

            code = self._non_comment_code(path.read_text())

            self.assertNotIn("MultiAgentSimulation", code, f"scenario_event_executor/{path.name}")
            self.assertNotIn("HazardEvolutionEngine", code, f"scenario_event_executor/{path.name}")
            self.assertNotIn("HumanBehaviorLayer", code, f"scenario_event_executor/{path.name}")
            self.assertNotIn(".submit_decision", code, f"scenario_event_executor/{path.name}")
            self.assertNotIn(".add_occupant", code, f"scenario_event_executor/{path.name}")

    def test_package_performs_no_file_io(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "scenario_event_executor"

        forbidden = r"\bopen\s*\(|\.write\s*\(|\.read\s*\(|\bjson\.(load|dump)\s*\("

        for path in sorted(package_dir.glob("*.py")):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text),
                f"scenario_event_executor/{path.name} appears to perform file I/O",
            )


if __name__ == "__main__":
    unittest.main()
