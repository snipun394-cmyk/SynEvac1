import unittest

from dataclasses import fields

from models.assembly_point import AssemblyPoint
from models.building import Building
from models.door import Door
from models.exit import Exit
from models.staircase import Staircase
from models.zone import Zone

from navigation.cost import CostModel, DefaultCostModel
from navigation.edge import Edge
from navigation.graph_builder import NavigationGraphGenerator
from navigation.node import Node
from navigation.pathfinder import Pathfinder


def make_zone(name, **kwargs):

    fields = dict(
        x=0.0,
        y=0.0,
        width=2.0,
        height=4.0,
        zone_type="Room",
        max_occupancy=12,
    )
    fields.update(kwargs)

    return Zone(name=name, **fields)


def make_assembly_point(name, **kwargs):

    return AssemblyPoint(
        name=name,
        position=(10.0, 10.0),
        length=5.0,
        width=2.0,
        capacity=100,
        **kwargs,
    )


class NodeEngineeringPropertyTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone = make_zone("Lobby")
        self.assembly_point = make_assembly_point("Muster Area")

        self.floor.add_zone(self.zone)
        self.floor.add_assembly_point(self.assembly_point)

        self.graph = NavigationGraphGenerator().build(self.building)

    def test_zone_node_exposes_space_type_area_and_capacity(self):

        node = self.graph.find_node(self.zone.id)

        self.assertEqual(node.space_type, "Room")
        self.assertEqual(node.area, self.zone.area)
        self.assertEqual(node.capacity, 12)

    def test_assembly_point_node_exposes_area_and_capacity_but_no_space_type(self):

        node = self.graph.find_node(self.assembly_point.id)

        self.assertIsNone(node.space_type)
        self.assertEqual(node.area, self.assembly_point.area)
        self.assertEqual(node.capacity, 100)

    def test_outside_node_has_no_engineering_values(self):

        outside = self.graph.find_node(Node.OUTSIDE_NODE_ID)

        self.assertIsNone(outside.space_type)
        self.assertIsNone(outside.area)
        self.assertIsNone(outside.capacity)

    def test_node_carries_no_dynamic_state_fields(self):

        # Occupancy/smoke/temperature/visibility must never live on
        # Node -- rebuilding the graph replaces every Node instance,
        # so anything stored here would be lost on the next rebuild.
        # Future Simulation/Hazard layers own that state themselves.
        field_names = {f.name for f in fields(Node)}

        self.assertNotIn("dynamic_state", field_names)
        self.assertEqual(
            field_names,
            {"id", "name", "floor_id", "node_type", "reference"},
        )


class EdgeEngineeringPropertyTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone_a = make_zone("A")
        self.zone_b = make_zone("B", x=10.0, y=0.0)

        self.floor.add_zone(self.zone_a)
        self.floor.add_zone(self.zone_b)

    def _find_edge(self, graph, edge_type):

        return next(
            edge
            for edge in graph.edges
            if edge.edge_type == edge_type
        )

    def test_door_edge_exposes_width_traversable_and_walking_distance(self):

        door = Door(
            name="D1",
            zone_a_id=self.zone_a.id,
            zone_b_id=self.zone_b.id,
            floor_id=self.floor.id,
            width=1.1,
            start_point=(2.0, 2.0),
            end_point=(2.0, 2.0),
        )
        self.floor.add_door(door)

        graph = NavigationGraphGenerator().build(self.building)
        edge = self._find_edge(graph, Edge.DOOR)

        self.assertEqual(edge.width, 1.1)
        self.assertIsNone(edge.capacity)
        self.assertTrue(edge.traversable)
        self.assertIsNotNone(edge.walking_distance)
        self.assertEqual(edge.traversal_cost, edge.walking_distance)
        self.assertEqual(
            edge.traversal_time,
            edge.walking_distance / Edge.ASSUMED_WALK_SPEED_M_PER_S,
        )

    def test_locked_door_is_not_traversable(self):

        door = Door(
            name="D1",
            zone_a_id=self.zone_a.id,
            zone_b_id=self.zone_b.id,
            floor_id=self.floor.id,
            locked=True,
        )
        self.floor.add_door(door)

        graph = NavigationGraphGenerator().build(self.building)
        edge = self._find_edge(graph, Edge.DOOR)

        self.assertFalse(edge.traversable)

    def test_inactive_door_is_not_traversable(self):

        door = Door(
            name="D1",
            zone_a_id=self.zone_a.id,
            zone_b_id=self.zone_b.id,
            floor_id=self.floor.id,
            active=False,
        )
        self.floor.add_door(door)

        graph = NavigationGraphGenerator().build(self.building)
        edge = self._find_edge(graph, Edge.DOOR)

        self.assertFalse(edge.traversable)

    def test_exit_edge_exposes_capacity_and_blocked_state(self):

        exit_obj = Exit(
            name="Ex1",
            zone_id=self.zone_a.id,
            floor_id=self.floor.id,
            capacity=75,
            is_blocked=True,
            start_point=(1.0, 0.0),
            end_point=(1.0, 0.0),
        )
        self.floor.add_exit(exit_obj)

        graph = NavigationGraphGenerator().build(self.building)
        edge = self._find_edge(graph, Edge.EXIT)

        self.assertEqual(edge.capacity, 75)
        self.assertFalse(edge.traversable)
        self.assertIsNotNone(edge.walking_distance)

    def test_stair_walking_distance_uses_travel_distance(self):

        floor1 = self.building.create_floor(name="Floor 1", height=3.0)
        zone_up = make_zone("Up", floor_id=floor1.id)
        floor1.add_zone(zone_up)

        stair = Staircase(
            name="S1",
            from_floor_id=self.floor.id,
            to_floor_id=floor1.id,
            from_zone_id=self.zone_a.id,
            to_zone_id=zone_up.id,
            width=1.5,
        )
        self.floor.add_stair(stair)

        graph = NavigationGraphGenerator().build(self.building)
        edge = self._find_edge(graph, Edge.STAIR)

        self.assertEqual(edge.width, 1.5)
        self.assertTrue(edge.traversable)
        self.assertEqual(
            edge.walking_distance,
            stair.travel_distance(self.building),
        )
        self.assertGreater(edge.walking_distance, 0.0)

    def test_edge_without_derivable_distance_falls_back_to_default_cost(self):

        edge = Edge(
            id="e1",
            edge_type=Edge.DOOR,
            from_node="a",
            to_node="b",
        )

        self.assertIsNone(edge.walking_distance)
        self.assertEqual(edge.traversal_cost, Edge.DEFAULT_TRAVERSAL_COST)
        self.assertIsNone(edge.traversal_time)

    def test_edge_carries_no_dynamic_state_fields(self):

        # Blocked/smoke/fire/congestion must never live on Edge, for
        # the same reason as Node: rebuilding the graph replaces every
        # Edge instance. Those penalties belong to a future CostModel
        # that wraps DefaultCostModel and consults its own external
        # state (see navigation/cost.py), not to fields on Edge.
        #
        # Obstacle -> Navigation & Evacuation Connectivity milestone --
        # `blocking_obstacles` is a deliberate, narrow exception, not a
        # violation of this rule: it holds LIVE references to the same
        # Obstacle objects Floor.obstacles already owns (exactly like
        # `reference` itself holds a live Door/Exit/Stair reference,
        # never a cached snapshot of its state) -- Edge.traversable
        # reads an obstacle's CURRENT active/traversability/position
        # fresh on every access, never a value baked in at build time.
        # See navigation/edge.py's own field comment for the full
        # reasoning.
        field_names = {f.name for f in fields(Edge)}

        self.assertNotIn("dynamic_state", field_names)
        self.assertEqual(
            field_names,
            {
                "id",
                "edge_type",
                "from_node",
                "to_node",
                "reference",
                "walking_distance",
                "blocking_obstacles",
            },
        )


class CostModelTests(unittest.TestCase):

    def test_default_cost_model_returns_edge_traversal_cost(self):

        edge = Edge(
            id="e1",
            edge_type=Edge.DOOR,
            from_node="a",
            to_node="b",
            walking_distance=4.2,
        )

        self.assertEqual(DefaultCostModel().cost(edge), 4.2)

    def test_cost_model_base_class_is_not_directly_usable(self):

        edge = Edge(id="e1", edge_type=Edge.DOOR, from_node="a", to_node="b")

        with self.assertRaises(NotImplementedError):
            CostModel().cost(edge)

    def test_hazard_aware_cost_model_can_wrap_the_default_without_touching_edge(self):

        # Proves out the extension point documented in
        # navigation/cost.py: a future Simulation/Hazard layer owns
        # its own state (here, a plain dict keyed by edge.id) and
        # supplies penalised costs by wrapping DefaultCostModel --
        # Edge itself never has to change or carry that state.
        edge = Edge(
            id="e1",
            edge_type=Edge.DOOR,
            from_node="a",
            to_node="b",
            walking_distance=4.0,
        )

        class HazardAwareCostModel(CostModel):

            def __init__(self, base_model, hazard_state):
                self.base_model = base_model
                self.hazard_state = hazard_state

            def cost(self, edge):
                penalty = self.hazard_state.get(edge.id, 0.0)
                return self.base_model.cost(edge) + penalty

        hazard_state = {"e1": 10.0}
        model = HazardAwareCostModel(DefaultCostModel(), hazard_state)

        self.assertEqual(model.cost(edge), 14.0)
        self.assertEqual(edge.walking_distance, 4.0)  # Edge itself is untouched


class PathfinderCostModelWiringTests(unittest.TestCase):

    def test_pathfinder_defaults_to_default_cost_model(self):

        building = Building(name="B")
        graph = NavigationGraphGenerator().build(building)

        pathfinder = Pathfinder(graph)

        self.assertIsInstance(pathfinder.cost_model, DefaultCostModel)

    def test_pathfinder_accepts_a_custom_cost_model(self):

        building = Building(name="B")
        graph = NavigationGraphGenerator().build(building)

        class FixedCostModel(CostModel):

            def cost(self, edge):
                return 1.0

        custom_model = FixedCostModel()
        pathfinder = Pathfinder(graph, cost_model=custom_model)

        self.assertIs(pathfinder.cost_model, custom_model)

    def test_shortest_path_still_unimplemented(self):

        building = Building(name="B")
        graph = NavigationGraphGenerator().build(building)

        pathfinder = Pathfinder(graph)

        self.assertEqual(pathfinder.shortest_path("a", "b"), [])


if __name__ == "__main__":
    unittest.main()
