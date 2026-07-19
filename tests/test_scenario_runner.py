import unittest
from dataclasses import FrozenInstanceError

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
from navigation.graph import NavigationGraph
from pathfinding.engine import PathfindingEngine

from simulator.coordinator import MultiAgentSimulation

from hazard.snapshot import HazardSnapshot
from hazard_evolution.engine import HazardEvolutionEngine

from fire_growth.model import FireGrowthModel
from fire_growth.spread import FireSpreadModel

from smoke_propagation.model import SmokePropagationModel

from scenario import (
    DeviceAvailability,
    DoorState,
    PresenceState,
    Scenario,
    ScenarioCameraState,
    ScenarioDetectorState,
    ScenarioDoorState,
    ScenarioEvent,
    ScenarioExitState,
    ScenarioFire,
    ScenarioMetadata,
    ScenarioObstacleState,
    ScenarioOccupant,
    ScenarioStairState,
    StairAvailability,
)

from scenario_runner import SimulationContext, run


def make_building():

    floor1 = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-1", name="Lobby", x=0.0, y=0.0, width=10.0, height=8.0),
            Zone(id="zone-2", name="Office", x=20.0, y=0.0, width=6.0, height=6.0),
        ],
        doors=[Door(id="door-1", normally_open=True, zone_a_id="zone-1", zone_b_id="zone-2")],
        exits=[Exit(id="exit-1", zone_id="zone-1")],
        obstacles=[Obstacle(id="obs-1", active=True), Obstacle(id="obs-2", active=False)],
        cameras=[Camera(id="cam-1", active=True), Camera(id="cam-2", active=False)],
        detectors=[Detector(id="det-1", active=True), Detector(id="det-2", active=False)],
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


def make_occupant(**overrides):

    defaults = dict(
        occupant_id="occ-1", zone_id="zone-1", floor_id="floor-1",
        position=(3.5, 4.25), behaviour_profile_id="Adult_Default",
    )
    defaults.update(overrides)

    return ScenarioOccupant(**defaults)


def make_scenario(**overrides):

    defaults = dict(
        metadata=make_metadata(),
        occupants=(make_occupant(), make_occupant(occupant_id="occ-2", position=(1.0, 1.0))),
        fire=ScenarioFire(
            ignition_zone_id="zone-2", ignition_floor_id="floor-1",
            fire_profile="Electrical", growth_parameters={"growth_time": 250.0},
        ),
        door_states=(ScenarioDoorState(door_id="door-1", state=DoorState.LOCKED),),
        exit_states=(ScenarioExitState(exit_id="exit-1", is_open=False),),
        obstacle_states=(
            ScenarioObstacleState(obstacle_id="obs-1", presence=PresenceState.INACTIVE),
            ScenarioObstacleState(obstacle_id="obs-2", presence=PresenceState.ACTIVE),
        ),
        camera_states=(
            ScenarioCameraState(camera_id="cam-1", availability=DeviceAvailability.FAILED),
            ScenarioCameraState(camera_id="cam-2", availability=DeviceAvailability.AVAILABLE),
        ),
        detector_states=(
            ScenarioDetectorState(detector_id="det-1", availability=DeviceAvailability.FAILED),
            ScenarioDetectorState(detector_id="det-2", availability=DeviceAvailability.AVAILABLE),
        ),
        stair_states=(ScenarioStairState(stair_id="stair-1", availability=StairAvailability.CLOSED),),
        events=(
            ScenarioEvent(
                event_id="evt-1", target_type="door", target_id="door-1",
                event_type="close", time=90.0, parameters={"reason": "drill"},
            ),
        ),
    )
    defaults.update(overrides)

    return Scenario(**defaults)


class BuildingDeepCopyTests(unittest.TestCase):

    def test_returned_building_is_not_the_original_object(self):

        building = make_building()
        context = run(make_scenario(), building)

        self.assertIsNot(context.building, building)

    def test_original_building_door_state_is_unchanged(self):

        building = make_building()
        original_normally_open = building.floors[0].doors[0].normally_open
        original_locked = building.floors[0].doors[0].locked

        run(make_scenario(), building)

        self.assertEqual(building.floors[0].doors[0].normally_open, original_normally_open)
        self.assertEqual(building.floors[0].doors[0].locked, original_locked)

    def test_original_building_exit_state_is_unchanged(self):

        building = make_building()
        original_is_blocked = building.floors[0].exits[0].is_blocked

        run(make_scenario(), building)

        self.assertEqual(building.floors[0].exits[0].is_blocked, original_is_blocked)

    def test_original_building_obstacle_camera_detector_state_is_unchanged(self):

        building = make_building()
        original_obstacle_active = building.floors[0].obstacles[0].active
        original_camera_active = building.floors[0].cameras[0].active
        original_detector_active = building.floors[0].detectors[0].active

        run(make_scenario(), building)

        self.assertEqual(building.floors[0].obstacles[0].active, original_obstacle_active)
        self.assertEqual(building.floors[0].cameras[0].active, original_camera_active)
        self.assertEqual(building.floors[0].detectors[0].active, original_detector_active)

    def test_copy_has_the_same_structural_shape_as_the_original(self):

        building = make_building()
        context = run(make_scenario(), building)

        self.assertEqual(len(context.building.floors), len(building.floors))
        self.assertEqual(
            {z.id for z in context.building.floors[0].zones},
            {z.id for z in building.floors[0].zones},
        )

    def test_running_twice_never_lets_the_second_run_see_the_first_runs_mutations(self):

        building = make_building()

        first_context = run(make_scenario(), building)
        second_context = run(
            make_scenario(door_states=(ScenarioDoorState(door_id="door-1", state=DoorState.OPEN),)),
            building,
        )

        self.assertTrue(first_context.building.floors[0].doors[0].locked)
        self.assertTrue(second_context.building.floors[0].doors[0].normally_open)
        self.assertIsNot(first_context.building, second_context.building)


class EngineeringStateApplicationTests(unittest.TestCase):

    def test_door_locked_state_is_applied(self):

        context = run(make_scenario(), make_building())
        door = context.building.floors[0].doors[0]

        self.assertTrue(door.locked)
        self.assertFalse(door.normally_open)

    def test_door_open_state_is_applied(self):

        scenario = make_scenario(
            door_states=(ScenarioDoorState(door_id="door-1", state=DoorState.OPEN),),
        )
        context = run(scenario, make_building())
        door = context.building.floors[0].doors[0]

        self.assertFalse(door.locked)
        self.assertTrue(door.normally_open)

    def test_door_closed_state_is_applied(self):

        scenario = make_scenario(
            door_states=(ScenarioDoorState(door_id="door-1", state=DoorState.CLOSED),),
        )
        context = run(scenario, make_building())
        door = context.building.floors[0].doors[0]

        self.assertFalse(door.locked)
        self.assertFalse(door.normally_open)

    def test_exit_is_open_maps_to_is_blocked_false(self):

        scenario = make_scenario(exit_states=(ScenarioExitState(exit_id="exit-1", is_open=True),))
        context = run(scenario, make_building())

        self.assertFalse(context.building.floors[0].exits[0].is_blocked)

    def test_exit_not_open_maps_to_is_blocked_true(self):

        context = run(make_scenario(), make_building())

        self.assertTrue(context.building.floors[0].exits[0].is_blocked)

    def test_obstacle_states_are_applied(self):

        context = run(make_scenario(), make_building())
        obstacles = {o.id: o.active for o in context.building.floors[0].obstacles}

        self.assertFalse(obstacles["obs-1"])
        self.assertTrue(obstacles["obs-2"])

    def test_camera_states_are_applied(self):

        context = run(make_scenario(), make_building())
        cameras = {c.id: c.active for c in context.building.floors[0].cameras}

        self.assertFalse(cameras["cam-1"])
        self.assertTrue(cameras["cam-2"])

    def test_detector_states_are_applied(self):

        context = run(make_scenario(), make_building())
        detectors = {d.id: d.active for d in context.building.floors[0].detectors}

        self.assertFalse(detectors["det-1"])
        self.assertTrue(detectors["det-2"])

    def test_unreferenced_ids_in_scenario_state_are_skipped_not_crashed(self):

        scenario = make_scenario(
            door_states=(ScenarioDoorState(door_id="door-ghost", state=DoorState.OPEN),),
        )
        # Should not raise.
        run(scenario, make_building())


class StairAvailabilityTests(unittest.TestCase):

    def test_closed_stair_removes_the_edge_from_the_graph(self):

        context = run(make_scenario(), make_building())
        stair_edges = [e for e in context.graph.edges if e.edge_type == Edge.STAIR]

        self.assertEqual(stair_edges, [])

    def test_available_stair_keeps_the_edge_in_the_graph(self):

        scenario = make_scenario(
            stair_states=(ScenarioStairState(stair_id="stair-1", availability=StairAvailability.AVAILABLE),),
        )
        context = run(scenario, make_building())
        stair_edges = [e for e in context.graph.edges if e.edge_type == Edge.STAIR]

        self.assertEqual(len(stair_edges), 1)
        self.assertEqual(stair_edges[0].id, "stair-1")

    def test_no_stair_state_declared_leaves_the_edge_in_place(self):

        scenario = make_scenario(stair_states=())
        context = run(scenario, make_building())
        stair_edges = [e for e in context.graph.edges if e.edge_type == Edge.STAIR]

        self.assertEqual(len(stair_edges), 1)

    def test_stair_exclusion_does_not_remove_door_or_exit_edges(self):

        context = run(make_scenario(), make_building())
        edge_types = {e.edge_type for e in context.graph.edges}

        self.assertIn(Edge.DOOR, edge_types)
        self.assertIn(Edge.EXIT, edge_types)


class NavigationGraphTests(unittest.TestCase):

    def test_graph_is_a_navigation_graph_built_from_the_copy(self):

        context = run(make_scenario(), make_building())

        self.assertIsInstance(context.graph, NavigationGraph)
        self.assertIn("zone-1", context.graph.nodes)
        self.assertIn("zone-2", context.graph.nodes)

    def test_engine_wraps_the_same_graph(self):

        context = run(make_scenario(), make_building())

        self.assertIsInstance(context.engine, PathfindingEngine)
        self.assertIs(context.engine.graph, context.graph)

    def test_navigation_reflects_applied_door_state_not_original_building(self):

        # door-1 is normally_open=True on the *original* Building but
        # pinned LOCKED by the default Scenario fixture -- the graph
        # must reflect the Scenario's state, not the Building's.
        context = run(make_scenario(), make_building())

        door_edge = next(e for e in context.graph.edges if e.edge_type == Edge.DOOR)

        self.assertFalse(door_edge.traversable)


class FireInitializationTests(unittest.TestCase):

    def test_hazard_engine_has_fire_growth_fire_spread_and_smoke_sources(self):

        context = run(make_scenario(), make_building())

        self.assertEqual(len(context.hazard_engine.sources), 3)
        self.assertIsInstance(context.hazard_engine.sources[0], FireGrowthModel)
        self.assertIsInstance(context.hazard_engine.sources[1], FireSpreadModel)
        self.assertIsInstance(context.hazard_engine.sources[2], SmokePropagationModel)

    def test_fire_growth_model_uses_the_scenario_ignition_zone(self):

        context = run(make_scenario(), make_building())
        fire_source = context.hazard_engine.sources[0]

        self.assertEqual(fire_source.ignition_node_id, "zone-2")

    def test_fire_growth_model_ignition_time_is_zero(self):

        context = run(make_scenario(), make_building())
        fire_source = context.hazard_engine.sources[0]

        self.assertEqual(fire_source.ignition_time, 0.0)

    def test_fire_growth_curve_uses_the_scenario_growth_time(self):

        context = run(make_scenario(), make_building())
        fire_source = context.hazard_engine.sources[0]

        self.assertEqual(fire_source.growth_curve.growth_time, 250.0)

    def test_initial_hazard_snapshot_is_empty_and_at_time_zero(self):

        context = run(make_scenario(), make_building())

        self.assertIsInstance(context.initial_hazard_snapshot, HazardSnapshot)
        self.assertEqual(context.initial_hazard_snapshot.timestamp, 0.0)
        self.assertEqual(len(context.initial_hazard_snapshot.node_states), 0)
        self.assertEqual(len(context.initial_hazard_snapshot.edge_states), 0)

    def test_hazard_engine_evolve_is_never_called_by_the_runner(self):

        # No step has been taken -- evolving from the initial snapshot
        # at time=0 with dt=0 must still reflect an un-ignited fire
        # (elapsed_time exactly at ignition boundary), proving no prior
        # evolve() call already advanced it.
        context = run(make_scenario(), make_building())

        result = context.hazard_engine.evolve(context.initial_hazard_snapshot, time=0.0, dt=0.0)

        self.assertEqual(result.timestamp, 0.0)


class OccupantInitializationTests(unittest.TestCase):

    def test_every_scenario_occupant_is_carried_into_the_context(self):

        scenario = make_scenario()
        context = run(scenario, make_building())

        self.assertEqual(context.occupants, scenario.occupants)

    def test_occupant_positions_are_not_modified(self):

        scenario = make_scenario()
        context = run(scenario, make_building())

        for original, carried in zip(scenario.occupants, context.occupants):
            self.assertEqual(original.position, carried.position)

    def test_behaviour_profile_ids_are_preserved_verbatim(self):

        scenario = make_scenario(
            occupants=(make_occupant(behaviour_profile_id="Totally_Unrecognized_Profile"),),
        )
        context = run(scenario, make_building())

        self.assertEqual(context.occupants[0].behaviour_profile_id, "Totally_Unrecognized_Profile")

    def test_simulation_is_constructed_but_has_no_registered_occupants(self):

        context = run(make_scenario(), make_building())

        self.assertIsInstance(context.simulation, MultiAgentSimulation)
        self.assertEqual(context.simulation._occupants, {})

    def test_simulation_wraps_the_same_engine(self):

        context = run(make_scenario(), make_building())

        self.assertIs(context.simulation.engine, context.engine)

    def test_no_occupants_is_handled_without_error(self):

        context = run(make_scenario(occupants=()), make_building())

        self.assertEqual(context.occupants, ())


class EventInitializationTests(unittest.TestCase):

    def test_events_are_carried_through_verbatim(self):

        scenario = make_scenario()
        context = run(scenario, make_building())

        self.assertEqual(context.scheduled_events, scenario.events)

    def test_event_parameters_are_unchanged(self):

        scenario = make_scenario()
        context = run(scenario, make_building())

        self.assertEqual(dict(context.scheduled_events[0].parameters), {"reason": "drill"})

    def test_no_events_is_handled_without_error(self):

        context = run(make_scenario(events=()), make_building())

        self.assertEqual(context.scheduled_events, ())


class SimulationContextConstructionTests(unittest.TestCase):

    def test_run_returns_a_simulation_context(self):

        context = run(make_scenario(), make_building())

        self.assertIsInstance(context, SimulationContext)

    def test_context_is_frozen(self):

        context = run(make_scenario(), make_building())

        with self.assertRaises(FrozenInstanceError):
            context.building = None

    def test_context_metadata_matches_the_scenario(self):

        scenario = make_scenario()
        context = run(scenario, make_building())

        self.assertEqual(context.metadata, scenario.metadata)

    def test_context_has_every_required_field(self):

        context = run(make_scenario(), make_building())

        for field_name in (
            "building", "graph", "engine", "simulation", "hazard_engine",
            "initial_hazard_snapshot", "occupants", "scheduled_events", "metadata",
        ):
            self.assertTrue(hasattr(context, field_name))

    def test_two_runs_with_the_same_inputs_produce_independent_contexts(self):

        scenario = make_scenario()
        building = make_building()

        first = run(scenario, building)
        second = run(scenario, building)

        self.assertIsNot(first, second)
        self.assertIsNot(first.building, second.building)
        self.assertIsNot(first.graph, second.graph)
        self.assertIsNot(first.simulation, second.simulation)
        self.assertEqual(first.occupants, second.occupants)
        self.assertEqual(first.scheduled_events, second.scheduled_events)


class ScenarioRunnerPackageDependencyDirectionTests(unittest.TestCase):

    # scenario_runner/ must never import scenario_generator,
    # scenario_validator, scenario_pipeline, behavior, behavior_library,
    # ai_decision, perception, sensors, occupancy, rl, sandbox,
    # designer, or random (architecture doc §12).

    def test_package_never_imports_forbidden_packages(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "scenario_runner"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(scenario_generator|scenario_validator|scenario_pipeline|behavior\b|"
            r"behavior_library|ai_decision|perception|sensors|occupancy|rl|sandbox|"
            r"designer|random)\b"
        )

        for path in sorted(package_dir.glob("*.py")):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"scenario_runner/{path.name} imports a package the architecture "
                f"(docs/architecture/scenario_runner.md §12) forbids",
            )

    @staticmethod
    def _non_comment_code(text):

        # Strips full-line and trailing comments so these structural
        # checks assert against actual code, not the explanatory prose
        # (which deliberately names the very things it forbids, e.g.
        # "must never call submit_decision()") documenting *why* each
        # module stays within its boundary.
        lines = []

        for line in text.splitlines():

            code_part = line.split("#", 1)[0]
            lines.append(code_part)

        return "\n".join(lines)

    def test_package_never_imports_behavior_under_any_spelling(self):

        import pathlib

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "scenario_runner"

        for path in sorted(package_dir.glob("*.py")):

            code = self._non_comment_code(path.read_text())

            self.assertNotIn("behavior_library", code, f"scenario_runner/{path.name}")
            self.assertNotIn("BehaviorProfile", code, f"scenario_runner/{path.name}")
            self.assertNotIn("DecisionStrategy", code, f"scenario_runner/{path.name}")
            self.assertNotIn("HumanBehaviorLayer", code, f"scenario_runner/{path.name}")

    def test_package_never_calls_register_or_submit_decision(self):

        import pathlib

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "scenario_runner"

        for path in sorted(package_dir.glob("*.py")):

            code = self._non_comment_code(path.read_text())

            self.assertNotIn(".register(", code, f"scenario_runner/{path.name}")
            self.assertNotIn("submit_decision", code, f"scenario_runner/{path.name}")

    def test_package_never_advances_simulation_time(self):

        import pathlib

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "scenario_runner"

        for path in sorted(package_dir.glob("*.py")):

            code = self._non_comment_code(path.read_text())

            self.assertNotIn(".evolve(", code, f"scenario_runner/{path.name}")
            self.assertNotIn(".run()", code, f"scenario_runner/{path.name}")

    def test_package_performs_no_file_io(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "scenario_runner"

        forbidden = r"\bopen\s*\(|\.write\s*\(|\.read\s*\(|\bjson\.(load|dump)\s*\("

        for path in sorted(package_dir.glob("*.py")):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text),
                f"scenario_runner/{path.name} appears to perform file I/O",
            )


if __name__ == "__main__":
    unittest.main()
