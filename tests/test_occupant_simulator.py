import unittest

from models.assembly_point import AssemblyPoint
from models.building import Building
from models.door import Door
from models.exit import Exit
from models.staircase import Staircase
from models.zone import Zone

from navigation.edge import Edge
from navigation.graph_builder import NavigationGraphGenerator

from pathfinding.engine import PathfindingEngine

from simulator.engine import OccupantSimulator


def make_zone(name, **kwargs):

    fields = dict(x=0.0, y=0.0, width=2.0, height=2.0)
    fields.update(kwargs)

    return Zone(name=name, **fields)


class LinearCorridorSimulationTests(unittest.TestCase):

    # A -- door --> B -- door --> C -- exit --> Outside

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone_a = make_zone("A", x=0.0, y=0.0)
        self.zone_b = make_zone("B", x=5.0, y=0.0)
        self.zone_c = make_zone("C", x=10.0, y=0.0)

        for zone in (self.zone_a, self.zone_b, self.zone_c):
            self.floor.add_zone(zone)

        self.door_ab = Door(
            name="AB", zone_a_id=self.zone_a.id, zone_b_id=self.zone_b.id,
            floor_id=self.floor.id,
        )
        self.door_bc = Door(
            name="BC", zone_a_id=self.zone_b.id, zone_b_id=self.zone_c.id,
            floor_id=self.floor.id,
        )
        self.floor.add_door(self.door_ab)
        self.floor.add_door(self.door_bc)

        self.exit_obj = Exit(name="Ex1", zone_id=self.zone_c.id, floor_id=self.floor.id)
        self.floor.add_exit(self.exit_obj)

        self.graph = NavigationGraphGenerator().build(self.building)
        self.engine = PathfindingEngine(self.graph)
        self.simulator = OccupantSimulator(self.engine)

    def test_simulate_to_goal_walks_every_hop_in_order(self):

        result = self.simulator.simulate_to_goal(self.zone_a.id, self.zone_c.id)

        self.assertTrue(result.reached_goal)
        self.assertEqual(
            result.visited_node_ids,
            [self.zone_a.id, self.zone_b.id, self.zone_c.id],
        )
        self.assertEqual(
            result.traversed_edge_ids,
            [self.door_ab.id, self.door_bc.id],
        )
        self.assertEqual(len(result.steps), 2)

    def test_step_indices_and_endpoints_match_the_route(self):

        result = self.simulator.simulate_to_goal(self.zone_a.id, self.zone_c.id)

        self.assertEqual(result.steps[0].index, 0)
        self.assertEqual(result.steps[0].from_node.id, self.zone_a.id)
        self.assertEqual(result.steps[0].to_node.id, self.zone_b.id)
        self.assertEqual(result.steps[0].edge.id, self.door_ab.id)

        self.assertEqual(result.steps[1].index, 1)
        self.assertEqual(result.steps[1].from_node.id, self.zone_b.id)
        self.assertEqual(result.steps[1].to_node.id, self.zone_c.id)
        self.assertEqual(result.steps[1].edge.id, self.door_bc.id)

    def test_elapsed_time_accumulates_monotonically_and_matches_total(self):

        result = self.simulator.simulate_to_goal(self.zone_a.id, self.zone_c.id)

        self.assertEqual(result.steps[0].start_time, 0.0)
        self.assertEqual(result.steps[0].end_time, result.steps[1].start_time)
        self.assertEqual(result.steps[1].end_time, result.total_elapsed_time)

        self.assertGreater(result.total_elapsed_time, 0.0)

    def test_total_distance_matches_route_distance(self):

        route = self.engine.dijkstra(self.zone_a.id, self.zone_c.id)
        result = self.simulator.simulate_to_goal(self.zone_a.id, self.zone_c.id)

        self.assertAlmostEqual(result.total_distance, route.total_distance)

    def test_step_distance_matches_edge_walking_distance(self):

        result = self.simulator.simulate_to_goal(self.zone_a.id, self.zone_c.id)

        for step in result.steps:
            self.assertEqual(step.distance, step.edge.walking_distance)

    def test_evacuate_matches_nearest_exit_route(self):

        route = self.engine.nearest_exit(self.zone_a.id)
        result = self.simulator.evacuate(self.zone_a.id)

        self.assertTrue(result.reached_goal)
        self.assertEqual(result.goal_node_id, route.goal.id)
        self.assertEqual(result.traversed_edge_ids, route.edge_ids)
        self.assertEqual(result.traversed_edge_ids[-1], self.exit_obj.id)

    def test_unreachable_goal_produces_a_failed_result_not_an_exception(self):

        lonely = make_zone("Lonely", x=200.0, y=200.0)
        self.floor.add_zone(lonely)

        graph = NavigationGraphGenerator().build(self.building)
        simulator = OccupantSimulator(PathfindingEngine(graph))

        result = simulator.simulate_to_goal(self.zone_a.id, lonely.id)

        self.assertFalse(result.reached_goal)
        self.assertIsNone(result.route)
        self.assertEqual(result.steps, [])
        self.assertIsNone(result.total_elapsed_time)
        self.assertIsNone(result.total_distance)
        self.assertEqual(result.visited_node_ids, [])
        self.assertEqual(result.traversed_edge_ids, [])
        self.assertEqual(result.start_node_id, self.zone_a.id)
        self.assertEqual(result.goal_node_id, lonely.id)

    def test_unknown_node_id_produces_a_failed_result(self):

        result = self.simulator.simulate_to_goal("not-a-real-id", self.zone_c.id)

        self.assertFalse(result.reached_goal)
        self.assertIsNone(result.route)

    def test_trivial_route_when_start_equals_goal(self):

        result = self.simulator.simulate_to_goal(self.zone_a.id, self.zone_a.id)

        self.assertTrue(result.reached_goal)
        self.assertEqual(result.steps, [])
        self.assertEqual(result.visited_node_ids, [self.zone_a.id])
        self.assertEqual(result.traversed_edge_ids, [])
        self.assertEqual(result.total_elapsed_time, 0.0)
        self.assertEqual(result.total_distance, 0.0)

    def test_occupant_id_defaults_and_is_configurable(self):

        default_result = self.simulator.simulate_to_goal(self.zone_a.id, self.zone_c.id)
        self.assertEqual(default_result.occupant_id, "occupant-1")

        named_result = self.simulator.simulate_to_goal(
            self.zone_a.id, self.zone_c.id, occupant_id="alice",
        )
        self.assertEqual(named_result.occupant_id, "alice")

    def test_no_floor_transitions_on_a_single_floor_route(self):

        result = self.simulator.evacuate(self.zone_a.id)

        self.assertEqual(result.floor_transitions, [])

    def test_exiting_through_an_exit_is_not_a_floor_transition(self):

        result = self.simulator.evacuate(self.zone_c.id)

        exit_step = result.steps[-1]
        self.assertEqual(exit_step.edge.edge_type, Edge.EXIT)
        self.assertFalse(exit_step.is_floor_transition)


class MultiFloorSimulationTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.ground = self.building.create_floor(name="Ground Floor")
        self.floor1 = self.building.create_floor(name="Floor 1", height=3.0)

        self.lobby = make_zone("Lobby", x=0.0, y=0.0)
        self.upstairs = make_zone("Upstairs", x=0.0, y=0.0, floor_id=self.floor1.id)

        self.ground.add_zone(self.lobby)
        self.floor1.add_zone(self.upstairs)

        self.exit_obj = Exit(name="Front Exit", zone_id=self.lobby.id, floor_id=self.ground.id)
        self.ground.add_exit(self.exit_obj)

        self.stair = Staircase(
            name="Main Stair",
            from_floor_id=self.ground.id, to_floor_id=self.floor1.id,
            from_zone_id=self.lobby.id, to_zone_id=self.upstairs.id,
        )
        self.ground.add_stair(self.stair)

        self.graph = NavigationGraphGenerator().build(self.building)
        self.simulator = OccupantSimulator(PathfindingEngine(self.graph))

    def test_stair_crossing_is_flagged_as_a_floor_transition(self):

        result = self.simulator.simulate_to_goal(self.upstairs.id, self.lobby.id)

        self.assertEqual(len(result.steps), 1)

        step = result.steps[0]
        self.assertEqual(step.edge.id, self.stair.id)
        self.assertTrue(step.is_floor_transition)
        self.assertEqual(result.floor_transitions, [step])

    def test_evacuate_from_upper_floor_crosses_stair_then_exit(self):

        result = self.simulator.evacuate(self.upstairs.id)

        self.assertTrue(result.reached_goal)
        self.assertEqual(len(result.steps), 2)

        self.assertEqual(result.steps[0].edge.id, self.stair.id)
        self.assertTrue(result.steps[0].is_floor_transition)

        self.assertEqual(result.steps[1].edge.id, self.exit_obj.id)
        self.assertFalse(result.steps[1].is_floor_transition)

        self.assertEqual(len(result.floor_transitions), 1)


class SimulateRouteTests(unittest.TestCase):

    # simulate_route() must make no routing decisions of its own --
    # it should faithfully execute whatever Route it is handed,
    # including a nearest_assembly_point() or alternative_paths()
    # result, not just the primary shortest path.

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone = make_zone("Zone", x=0.0, y=0.0)
        self.assembly_point = AssemblyPoint(
            name="Muster", position=(20.0, 0.0), length=5.0, width=2.0,
        )

        self.floor.add_zone(self.zone)
        self.floor.add_assembly_point(self.assembly_point)

        self.door = Door(
            name="D1", zone_a_id=self.zone.id, zone_b_id=self.assembly_point.id,
            floor_id=self.floor.id,
        )
        self.floor.add_door(self.door)

        self.graph = NavigationGraphGenerator().build(self.building)
        self.engine = PathfindingEngine(self.graph)
        self.simulator = OccupantSimulator(self.engine)

    def test_executes_a_nearest_assembly_point_route(self):

        route = self.engine.nearest_assembly_point(self.zone.id)
        result = self.simulator.simulate_route(route)

        self.assertTrue(result.reached_goal)
        self.assertEqual(result.goal_node_id, self.assembly_point.id)
        self.assertEqual(result.traversed_edge_ids, [self.door.id])

    def test_none_route_produces_a_failed_result(self):

        result = self.simulator.simulate_route(None)

        self.assertFalse(result.reached_goal)
        self.assertIsNone(result.start_node_id)
        self.assertIsNone(result.goal_node_id)
        self.assertEqual(result.steps, [])


class SimulatorIndependenceTests(unittest.TestCase):

    def test_simulator_package_never_touches_reference_or_engineering_models(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "simulator"

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertNotIn(
                ".reference", text, f"{path} touches .reference directly"
            )
            self.assertIsNone(
                re.search(r"^\s*(from|import)\s+(models|designer)\b", text, re.MULTILINE),
                f"{path} imports models/designer directly",
            )

    def test_simulator_never_imports_pathfinding_search_internals(self):

        # Simulation must obtain routes only through PathfindingEngine's
        # public query methods -- never reach into its private _search/
        # _relax machinery, which would amount to computing routes
        # itself.
        import pathlib

        engine_text = (
            pathlib.Path(__file__).resolve().parent.parent
            / "simulator" / "engine.py"
        ).read_text()

        self.assertNotIn("_search", engine_text)
        self.assertNotIn("_relax", engine_text)
        self.assertNotIn("heapq", engine_text)


if __name__ == "__main__":
    unittest.main()
