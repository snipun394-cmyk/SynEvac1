import unittest

from models.assembly_point import AssemblyPoint
from models.building import Building
from models.door import Door
from models.exit import Exit
from models.staircase import Staircase
from models.zone import Zone

from navigation.cost import CostModel, DefaultCostModel
from navigation.graph_builder import NavigationGraphGenerator
from navigation.node import Node

from pathfinding.engine import PathfindingEngine
from pathfinding.heuristics import ZeroHeuristic


def make_zone(name, **kwargs):

    fields = dict(x=0.0, y=0.0, width=2.0, height=2.0)
    fields.update(kwargs)

    return Zone(name=name, **fields)


def make_assembly_point(name, **kwargs):

    fields = dict(position=(0.0, 0.0), length=5.0, width=2.0)
    fields.update(kwargs)

    return AssemblyPoint(name=name, **fields)


class LinearCorridorTests(unittest.TestCase):

    # A -- door --> B -- door --> C -- exit --> Outside

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone_a = make_zone("A", x=0.0, y=0.0)
        self.zone_b = make_zone("B", x=5.0, y=0.0)
        self.zone_c = make_zone("C", x=10.0, y=0.0)

        self.floor.add_zone(self.zone_a)
        self.floor.add_zone(self.zone_b)
        self.floor.add_zone(self.zone_c)

        self.door_ab = Door(
            name="AB",
            zone_a_id=self.zone_a.id,
            zone_b_id=self.zone_b.id,
            floor_id=self.floor.id,
        )
        self.door_bc = Door(
            name="BC",
            zone_a_id=self.zone_b.id,
            zone_b_id=self.zone_c.id,
            floor_id=self.floor.id,
        )
        self.floor.add_door(self.door_ab)
        self.floor.add_door(self.door_bc)

        self.exit_obj = Exit(
            name="Ex1",
            zone_id=self.zone_c.id,
            floor_id=self.floor.id,
        )
        self.floor.add_exit(self.exit_obj)

        self.graph = NavigationGraphGenerator().build(self.building)
        self.engine = PathfindingEngine(self.graph)

    def test_dijkstra_finds_the_only_path(self):

        route = self.engine.dijkstra(self.zone_a.id, self.zone_c.id)

        self.assertIsNotNone(route)
        self.assertEqual(route.node_ids, [self.zone_a.id, self.zone_b.id, self.zone_c.id])
        self.assertEqual(route.edge_ids, [self.door_ab.id, self.door_bc.id])
        self.assertAlmostEqual(route.total_cost, route.total_distance)

    def test_a_star_matches_dijkstra_with_zero_heuristic(self):

        dijkstra_route = self.engine.dijkstra(self.zone_a.id, self.zone_c.id)
        a_star_route = self.engine.a_star(self.zone_a.id, self.zone_c.id)

        self.assertEqual(dijkstra_route.node_ids, a_star_route.node_ids)
        self.assertEqual(dijkstra_route.edge_ids, a_star_route.edge_ids)
        self.assertAlmostEqual(dijkstra_route.total_cost, a_star_route.total_cost)

    def test_shortest_path_dispatches_by_algorithm_name(self):

        via_default = self.engine.shortest_path(self.zone_a.id, self.zone_c.id)
        via_dijkstra = self.engine.shortest_path(
            self.zone_a.id, self.zone_c.id, algorithm="dijkstra"
        )
        via_a_star = self.engine.shortest_path(
            self.zone_a.id, self.zone_c.id, algorithm="a_star"
        )

        self.assertEqual(via_default.edge_ids, via_dijkstra.edge_ids)
        self.assertEqual(via_default.edge_ids, via_a_star.edge_ids)

    def test_unknown_algorithm_raises(self):

        with self.assertRaises(ValueError):
            self.engine.shortest_path(self.zone_a.id, self.zone_c.id, algorithm="bfs")

    def test_route_reconstruction_edges_match_nodes(self):

        route = self.engine.dijkstra(self.zone_a.id, self.zone_c.id)

        self.assertEqual(len(route.edges), len(route.nodes) - 1)

        for i, edge in enumerate(route.edges):

            self.assertIn(route.nodes[i].id, (edge.from_node, edge.to_node))
            self.assertIn(route.nodes[i + 1].id, (edge.from_node, edge.to_node))

    def test_start_equals_goal_is_a_trivial_route(self):

        route = self.engine.dijkstra(self.zone_a.id, self.zone_a.id)

        self.assertEqual(route.node_ids, [self.zone_a.id])
        self.assertEqual(route.edges, [])
        self.assertEqual(route.total_cost, 0.0)

    def test_unknown_node_id_returns_none(self):

        self.assertIsNone(self.engine.dijkstra("not-a-real-id", self.zone_c.id))
        self.assertIsNone(self.engine.dijkstra(self.zone_a.id, "not-a-real-id"))

    def test_nearest_exit_reaches_outside_via_the_exit_edge(self):

        route = self.engine.nearest_exit(self.zone_a.id)

        self.assertIsNotNone(route)
        self.assertEqual(route.goal.id, Node.OUTSIDE_NODE_ID)
        self.assertEqual(route.edges[-1].id, self.exit_obj.id)

    def test_locked_door_forces_a_detour_or_failure(self):

        self.door_ab.locked = True

        graph = NavigationGraphGenerator().build(self.building)
        engine = PathfindingEngine(graph)

        # A -> B is the only connection between them; locking it
        # makes C (and therefore Outside) unreachable from A.
        self.assertIsNone(engine.dijkstra(self.zone_a.id, self.zone_c.id))
        self.assertIsNone(engine.nearest_exit(self.zone_a.id))

        # B and C are still reachable from each other and can still
        # reach Outside.
        self.assertIsNotNone(engine.dijkstra(self.zone_b.id, self.zone_c.id))
        self.assertIsNotNone(engine.nearest_exit(self.zone_b.id))

    def test_blocked_exit_is_not_traversable(self):

        self.exit_obj.is_blocked = True

        graph = NavigationGraphGenerator().build(self.building)
        engine = PathfindingEngine(graph)

        self.assertIsNone(engine.nearest_exit(self.zone_c.id))


class MultiFloorRoutingTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.ground = self.building.create_floor(name="Ground Floor")
        self.floor1 = self.building.create_floor(name="Floor 1", height=3.0)

        self.lobby = make_zone("Lobby", x=0.0, y=0.0)
        self.upstairs = make_zone("Upstairs", x=0.0, y=0.0, floor_id=self.floor1.id)

        self.ground.add_zone(self.lobby)
        self.floor1.add_zone(self.upstairs)

        self.exit_obj = Exit(
            name="Front Exit",
            zone_id=self.lobby.id,
            floor_id=self.ground.id,
        )
        self.ground.add_exit(self.exit_obj)

        self.stair = Staircase(
            name="Main Stair",
            from_floor_id=self.ground.id,
            to_floor_id=self.floor1.id,
            from_zone_id=self.lobby.id,
            to_zone_id=self.upstairs.id,
        )
        self.ground.add_stair(self.stair)

        self.graph = NavigationGraphGenerator().build(self.building)
        self.engine = PathfindingEngine(self.graph)

    def test_route_crosses_floors_via_stair(self):

        route = self.engine.dijkstra(self.upstairs.id, self.lobby.id)

        self.assertEqual(route.node_ids, [self.upstairs.id, self.lobby.id])
        self.assertEqual(route.edges[0].id, self.stair.id)
        self.assertNotEqual(route.nodes[0].floor_id, route.nodes[1].floor_id)

    def test_nearest_exit_from_upper_floor_crosses_the_stair(self):

        route = self.engine.nearest_exit(self.upstairs.id)

        self.assertIsNotNone(route)
        self.assertEqual(route.goal.id, Node.OUTSIDE_NODE_ID)
        self.assertIn(self.stair.id, route.edge_ids)
        self.assertEqual(route.edges[-1].id, self.exit_obj.id)


class NearestAssemblyPointTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.start_zone = make_zone("Start", x=0.0, y=0.0)
        self.mid_zone = make_zone("Mid", x=5.0, y=0.0)

        self.floor.add_zone(self.start_zone)
        self.floor.add_zone(self.mid_zone)

        self.near_ap = make_assembly_point("Near AP", position=(1.0, 0.0))
        self.far_ap = make_assembly_point("Far AP", position=(50.0, 0.0))

        self.floor.add_assembly_point(self.near_ap)
        self.floor.add_assembly_point(self.far_ap)

        door_start_near = Door(
            name="Start-Near",
            zone_a_id=self.start_zone.id,
            zone_b_id=self.near_ap.id,
            floor_id=self.floor.id,
        )
        door_start_mid = Door(
            name="Start-Mid",
            zone_a_id=self.start_zone.id,
            zone_b_id=self.mid_zone.id,
            floor_id=self.floor.id,
        )
        door_mid_far = Door(
            name="Mid-Far",
            zone_a_id=self.mid_zone.id,
            zone_b_id=self.far_ap.id,
            floor_id=self.floor.id,
        )

        self.floor.add_door(door_start_near)
        self.floor.add_door(door_start_mid)
        self.floor.add_door(door_mid_far)

        self.graph = NavigationGraphGenerator().build(self.building)
        self.engine = PathfindingEngine(self.graph)

    def test_nearest_assembly_point_picks_the_closer_one(self):

        route = self.engine.nearest_assembly_point(self.start_zone.id)

        self.assertIsNotNone(route)
        self.assertEqual(route.goal.id, self.near_ap.id)

    def test_assembly_point_start_returns_itself(self):

        route = self.engine.nearest_assembly_point(self.near_ap.id)

        self.assertEqual(route.node_ids, [self.near_ap.id])
        self.assertEqual(route.total_cost, 0.0)


class CostModelIntegrationTests(unittest.TestCase):

    # Two parallel routes from A to D of different geometric length;
    # a custom CostModel that heavily penalises the shorter one's
    # door should flip which route the engine picks -- proving the
    # engine only ever reads cost through CostModel, never raw
    # geometry or engineering objects.

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone_a = make_zone("A", x=0.0, y=0.0)
        self.zone_b = make_zone("B", x=2.0, y=0.0)
        self.zone_c = make_zone("C", x=0.0, y=20.0)
        self.zone_d = make_zone("D", x=2.0, y=20.0)

        for zone in (self.zone_a, self.zone_b, self.zone_c, self.zone_d):
            self.floor.add_zone(zone)

        # Short path: A -> B -> D
        self.door_short_1 = Door(
            name="A-B", zone_a_id=self.zone_a.id, zone_b_id=self.zone_b.id,
            floor_id=self.floor.id,
        )
        self.door_short_2 = Door(
            name="B-D", zone_a_id=self.zone_b.id, zone_b_id=self.zone_d.id,
            floor_id=self.floor.id,
        )

        # Long path: A -> C -> D
        self.door_long_1 = Door(
            name="A-C", zone_a_id=self.zone_a.id, zone_b_id=self.zone_c.id,
            floor_id=self.floor.id,
        )
        self.door_long_2 = Door(
            name="C-D", zone_a_id=self.zone_c.id, zone_b_id=self.zone_d.id,
            floor_id=self.floor.id,
        )

        for door in (
            self.door_short_1,
            self.door_short_2,
            self.door_long_1,
            self.door_long_2,
        ):
            self.floor.add_door(door)

        self.graph = NavigationGraphGenerator().build(self.building)

    def test_default_cost_model_prefers_the_geometrically_shorter_route(self):

        engine = PathfindingEngine(self.graph)
        route = engine.dijkstra(self.zone_a.id, self.zone_d.id)

        self.assertEqual(
            route.edge_ids,
            [self.door_short_1.id, self.door_short_2.id],
        )

    def test_custom_cost_model_can_flip_the_preferred_route(self):

        door_short_1_id = self.door_short_1.id

        class HazardCostModel(CostModel):

            def cost(self, edge):

                if edge.id == door_short_1_id:
                    return 1000.0

                return edge.traversal_cost

        engine = PathfindingEngine(self.graph, cost_model=HazardCostModel())
        route = engine.dijkstra(self.zone_a.id, self.zone_d.id)

        self.assertEqual(
            route.edge_ids,
            [self.door_long_1.id, self.door_long_2.id],
        )

    def test_hazard_aware_wrapper_pattern_from_cost_module_works_end_to_end(self):

        # Mirrors the documented composition pattern in
        # navigation/cost.py (DefaultCostModel wrapped by a
        # hazard-aware model reading external state keyed by
        # edge.id), exercised through the actual engine.
        hazard_state = {self.door_short_1.id: 1000.0}

        class HazardAwareCostModel(CostModel):

            def __init__(self, base_model, hazard_state):
                self.base_model = base_model
                self.hazard_state = hazard_state

            def cost(self, edge):
                penalty = self.hazard_state.get(edge.id, 0.0)
                return self.base_model.cost(edge) + penalty

        engine = PathfindingEngine(
            self.graph,
            cost_model=HazardAwareCostModel(DefaultCostModel(), hazard_state),
        )
        route = engine.dijkstra(self.zone_a.id, self.zone_d.id)

        self.assertEqual(
            route.edge_ids,
            [self.door_long_1.id, self.door_long_2.id],
        )


class AlternativePathsTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone_a = make_zone("A", x=0.0, y=0.0)
        self.zone_b = make_zone("B", x=2.0, y=0.0)
        self.zone_c = make_zone("C", x=0.0, y=20.0)
        self.zone_d = make_zone("D", x=2.0, y=20.0)

        for zone in (self.zone_a, self.zone_b, self.zone_c, self.zone_d):
            self.floor.add_zone(zone)

        self.door_a_b = Door(
            name="A-B", zone_a_id=self.zone_a.id, zone_b_id=self.zone_b.id,
            floor_id=self.floor.id,
        )
        self.door_b_d = Door(
            name="B-D", zone_a_id=self.zone_b.id, zone_b_id=self.zone_d.id,
            floor_id=self.floor.id,
        )
        self.door_a_c = Door(
            name="A-C", zone_a_id=self.zone_a.id, zone_b_id=self.zone_c.id,
            floor_id=self.floor.id,
        )
        self.door_c_d = Door(
            name="C-D", zone_a_id=self.zone_c.id, zone_b_id=self.zone_d.id,
            floor_id=self.floor.id,
        )

        for door in (self.door_a_b, self.door_b_d, self.door_a_c, self.door_c_d):
            self.floor.add_door(door)

        self.graph = NavigationGraphGenerator().build(self.building)
        self.engine = PathfindingEngine(self.graph)

    def test_returns_two_distinct_loopless_routes_in_increasing_cost_order(self):

        routes = self.engine.alternative_paths(self.zone_a.id, self.zone_d.id, k=2)

        self.assertEqual(len(routes), 2)

        self.assertEqual(
            routes[0].edge_ids,
            [self.door_a_b.id, self.door_b_d.id],
        )
        self.assertEqual(
            routes[1].edge_ids,
            [self.door_a_c.id, self.door_c_d.id],
        )

        self.assertLessEqual(routes[0].total_cost, routes[1].total_cost)

        self.assertNotEqual(set(routes[0].edge_ids), set(routes[1].edge_ids))

    def test_first_alternative_matches_plain_dijkstra(self):

        dijkstra_route = self.engine.dijkstra(self.zone_a.id, self.zone_d.id)
        alternatives = self.engine.alternative_paths(self.zone_a.id, self.zone_d.id, k=1)

        self.assertEqual(len(alternatives), 1)
        self.assertEqual(alternatives[0].edge_ids, dijkstra_route.edge_ids)

    def test_requesting_more_than_exist_returns_only_what_exists(self):

        routes = self.engine.alternative_paths(self.zone_a.id, self.zone_d.id, k=10)

        self.assertEqual(len(routes), 2)

    def test_k_below_one_returns_empty_list(self):

        self.assertEqual(
            self.engine.alternative_paths(self.zone_a.id, self.zone_d.id, k=0),
            [],
        )

    def test_unreachable_goal_returns_empty_list(self):

        lonely = make_zone("Lonely", x=100.0, y=100.0)
        self.floor.add_zone(lonely)

        graph = NavigationGraphGenerator().build(self.building)
        engine = PathfindingEngine(graph)

        self.assertEqual(
            engine.alternative_paths(self.zone_a.id, lonely.id, k=3),
            [],
        )


class HeuristicWiringTests(unittest.TestCase):

    def test_engine_defaults_to_zero_heuristic(self):

        building = Building(name="B")
        graph = NavigationGraphGenerator().build(building)

        engine = PathfindingEngine(graph)

        self.assertIsInstance(engine.heuristic, ZeroHeuristic)

    def test_engine_accepts_a_custom_heuristic(self):

        building = Building(name="B")
        graph = NavigationGraphGenerator().build(building)

        custom = ZeroHeuristic()
        engine = PathfindingEngine(graph, heuristic=custom)

        self.assertIs(engine.heuristic, custom)


if __name__ == "__main__":
    unittest.main()
