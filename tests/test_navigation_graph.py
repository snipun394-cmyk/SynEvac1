import unittest

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.project import Project
from models.staircase import Staircase
from models.zone import Zone

from navigation.edge import Edge
from navigation.graph_builder import NavigationGraphGenerator
from navigation.node import Node

from serialization.serializer import Serializer


def make_zone(name, **kwargs):

    return Zone(
        name=name,
        x=0.0,
        y=0.0,
        width=2.0,
        height=2.0,
        **kwargs,
    )


class SingleFloorGraphTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.ground = self.building.create_floor(name="Ground Floor")

        self.zone_a = make_zone("A")
        self.zone_b = make_zone("B")

        self.ground.add_zone(self.zone_a)
        self.ground.add_zone(self.zone_b)

        self.graph = NavigationGraphGenerator().build(self.building)

    def test_zone_nodes_are_created(self):

        self.assertEqual(
            self.graph.find_node(self.zone_a.id).node_type,
            Node.ZONE,
        )

        self.assertEqual(
            self.graph.find_node(self.zone_b.id).node_type,
            Node.ZONE,
        )

    def test_outside_node_always_exists(self):

        outside = self.graph.find_node(Node.OUTSIDE_NODE_ID)

        self.assertIsNotNone(outside)
        self.assertEqual(outside.node_type, Node.OUTSIDE)

    def test_node_reference_is_the_original_object_not_a_copy(self):

        node = self.graph.find_node(self.zone_a.id)

        self.assertIs(node.reference, self.zone_a)

    def test_disconnected_floor_not_flagged_for_single_floor_building(self):

        report = self.graph.validate()

        self.assertEqual(report.by_code("disconnected_floor"), [])


class DoorConnectivityTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone_a = make_zone("A")
        self.zone_b = make_zone("B")

        self.floor.add_zone(self.zone_a)
        self.floor.add_zone(self.zone_b)

    def build_door(self, **overrides):

        kwargs = dict(
            name="D1",
            zone_a_id=self.zone_a.id,
            zone_b_id=self.zone_b.id,
            floor_id=self.floor.id,
        )

        kwargs.update(overrides)

        door = Door(**kwargs)
        self.floor.add_door(door)

        return door

    def test_valid_door_produces_zone_to_zone_edge(self):

        door = self.build_door()

        graph = NavigationGraphGenerator().build(self.building)

        door_edges = [
            edge
            for edge in graph.edges
            if edge.edge_type == Edge.DOOR
        ]

        self.assertEqual(len(door_edges), 1)

        edge = door_edges[0]

        self.assertEqual(edge.id, door.id)
        self.assertIs(edge.reference, door)

        self.assertEqual(
            {edge.from_node, edge.to_node},
            {self.zone_a.id, self.zone_b.id},
        )

    def test_door_edge_is_traversable_both_ways(self):

        self.build_door()

        graph = NavigationGraphGenerator().build(self.building)

        node_a = graph.find_node(self.zone_a.id)
        node_b = graph.find_node(self.zone_b.id)

        neighbors_of_a = [
            node.id
            for node, _ in graph.find_neighbors(node_a)
        ]

        neighbors_of_b = [
            node.id
            for node, _ in graph.find_neighbors(node_b)
        ]

        self.assertIn(self.zone_b.id, neighbors_of_a)
        self.assertIn(self.zone_a.id, neighbors_of_b)

    def test_door_missing_zone_a_is_reported_and_no_edge_created(self):

        door = self.build_door(zone_a_id="")

        graph = NavigationGraphGenerator().build(self.building)

        self.assertEqual(
            [
                edge
                for edge in graph.edges
                if edge.edge_type == Edge.DOOR
            ],
            [],
        )

        report = graph.validate()
        issues = report.by_code("door_missing_zone_a")

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].object_id, door.id)
        self.assertTrue(report.is_valid)  # warning, not error

    def test_door_missing_zone_b_is_reported_and_no_edge_created(self):

        door = self.build_door(zone_b_id="")

        graph = NavigationGraphGenerator().build(self.building)

        self.assertEqual(
            [
                edge
                for edge in graph.edges
                if edge.edge_type == Edge.DOOR
            ],
            [],
        )

        report = graph.validate()
        issues = report.by_code("door_missing_zone_b")

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].object_id, door.id)

    def test_door_with_dangling_zone_reference_is_invalid_reference(self):

        self.build_door(zone_a_id="not-a-real-zone-id")

        graph = NavigationGraphGenerator().build(self.building)

        report = graph.validate()

        self.assertFalse(report.is_valid)
        self.assertEqual(len(report.by_code("invalid_reference")), 1)

    def test_door_referencing_same_zone_twice_is_invalid(self):

        self.build_door(zone_b_id=self.zone_a.id)

        graph = NavigationGraphGenerator().build(self.building)

        report = graph.validate()

        self.assertFalse(report.is_valid)
        self.assertEqual(
            [
                edge
                for edge in graph.edges
                if edge.edge_type == Edge.DOOR
            ],
            [],
        )


class ExitConnectivityTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone = make_zone("A")
        self.floor.add_zone(self.zone)

    def test_valid_exit_produces_zone_to_outside_edge(self):

        exit_obj = Exit(
            name="Ex1",
            zone_id=self.zone.id,
            floor_id=self.floor.id,
        )

        self.floor.add_exit(exit_obj)

        graph = NavigationGraphGenerator().build(self.building)

        exit_edges = [
            edge
            for edge in graph.edges
            if edge.edge_type == Edge.EXIT
        ]

        self.assertEqual(len(exit_edges), 1)

        edge = exit_edges[0]

        self.assertEqual(
            {edge.from_node, edge.to_node},
            {self.zone.id, Node.OUTSIDE_NODE_ID},
        )

        self.assertIs(edge.reference, exit_obj)

    def test_exit_not_connected_to_a_zone_is_reported(self):

        exit_obj = Exit(name="Ex1", floor_id=self.floor.id)

        self.floor.add_exit(exit_obj)

        graph = NavigationGraphGenerator().build(self.building)

        self.assertEqual(
            [
                edge
                for edge in graph.edges
                if edge.edge_type == Edge.EXIT
            ],
            [],
        )

        report = graph.validate()
        issues = report.by_code("exit_missing_zone")

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].object_id, exit_obj.id)


class StairConnectivityTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.ground = self.building.create_floor(name="Ground Floor")
        self.floor1 = self.building.create_floor(name="Floor 1")

        self.zone_ground = make_zone("Ground Zone")
        self.zone_floor1 = make_zone("Floor 1 Zone")

        self.ground.add_zone(self.zone_ground)
        self.floor1.add_zone(self.zone_floor1)

    def build_stair(self, **overrides):

        kwargs = dict(
            name="S1",
            from_floor_id=self.ground.id,
            to_floor_id=self.floor1.id,
            from_zone_id=self.zone_ground.id,
            to_zone_id=self.zone_floor1.id,
        )

        kwargs.update(overrides)

        stair = Staircase(**kwargs)
        self.ground.add_stair(stair)

        return stair

    def test_valid_stair_produces_cross_floor_zone_edge(self):

        stair = self.build_stair()

        graph = NavigationGraphGenerator().build(self.building)

        stair_edges = [
            edge
            for edge in graph.edges
            if edge.edge_type == Edge.STAIR
        ]

        self.assertEqual(len(stair_edges), 1)

        edge = stair_edges[0]

        self.assertEqual(edge.id, stair.id)

        self.assertEqual(
            {edge.from_node, edge.to_node},
            {self.zone_ground.id, self.zone_floor1.id},
        )

    def test_stair_connects_two_different_floors(self):

        self.build_stair()

        graph = NavigationGraphGenerator().build(self.building)

        from_node = graph.find_node(self.zone_ground.id)
        to_node = graph.find_node(self.zone_floor1.id)

        self.assertNotEqual(from_node.floor_id, to_node.floor_id)

    def test_stair_missing_destination_floor_is_reported(self):

        stair = self.build_stair(to_floor_id="")

        graph = NavigationGraphGenerator().build(self.building)

        self.assertEqual(
            [
                edge
                for edge in graph.edges
                if edge.edge_type == Edge.STAIR
            ],
            [],
        )

        report = graph.validate()
        issues = report.by_code("stair_missing_destination_floor")

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].object_id, stair.id)

    def test_stair_with_dangling_destination_floor_is_invalid_reference(self):

        self.build_stair(to_floor_id="not-a-real-floor-id")

        graph = NavigationGraphGenerator().build(self.building)

        report = graph.validate()

        self.assertFalse(report.is_valid)
        self.assertGreaterEqual(
            len(report.by_code("invalid_reference")),
            1,
        )

    def test_stair_missing_origin_or_destination_zone_is_reported(self):

        self.build_stair(from_zone_id="", to_zone_id="")

        graph = NavigationGraphGenerator().build(self.building)

        report = graph.validate()

        self.assertEqual(len(report.by_code("stair_missing_origin_zone")), 1)
        self.assertEqual(
            len(report.by_code("stair_missing_destination_zone")),
            1,
        )

    def test_stair_zone_on_wrong_floor_is_invalid_reference(self):

        # from_zone_id points at a Zone that actually lives on
        # to_floor's floor, not the Stair's own from_floor -- a real
        # data-entry mistake, not merely "unset".
        self.build_stair(from_zone_id=self.zone_floor1.id)

        graph = NavigationGraphGenerator().build(self.building)

        report = graph.validate()

        self.assertFalse(report.is_valid)
        self.assertEqual(
            [
                edge
                for edge in graph.edges
                if edge.edge_type == Edge.STAIR
            ],
            [],
        )


class MultiFloorGraphTests(unittest.TestCase):

    def test_full_building_connectivity_end_to_end(self):

        building = Building(name="B")
        ground = building.create_floor(name="Ground Floor")
        floor1 = building.create_floor(name="Floor 1")

        lobby = make_zone("Lobby")
        office = make_zone("Office")

        ground.add_zone(lobby)
        ground.add_zone(office)

        upstairs_office = make_zone("Upstairs Office")
        floor1.add_zone(upstairs_office)

        door = Door(
            name="Lobby-Office Door",
            zone_a_id=lobby.id,
            zone_b_id=office.id,
            floor_id=ground.id,
        )
        ground.add_door(door)

        exit_obj = Exit(
            name="Front Exit",
            zone_id=lobby.id,
            floor_id=ground.id,
        )
        ground.add_exit(exit_obj)

        stair = Staircase(
            name="Main Stair",
            from_floor_id=ground.id,
            to_floor_id=floor1.id,
            from_zone_id=lobby.id,
            to_zone_id=upstairs_office.id,
        )
        ground.add_stair(stair)

        graph = NavigationGraphGenerator().build(building)

        report = graph.validate()

        self.assertTrue(report.is_valid)
        self.assertEqual(report.by_code("disconnected_floor"), [])
        self.assertEqual(report.by_code("isolated_zone"), [])
        self.assertEqual(report.by_code("zone_without_connections"), [])

        # Upstairs Office can reach Outside only via: itself -> Stair
        # -> Lobby -> Exit -> Outside.
        upstairs_node = graph.find_node(upstairs_office.id)
        neighbor_ids = {
            node.id
            for node, _ in graph.find_neighbors(upstairs_node)
        }

        self.assertEqual(neighbor_ids, {lobby.id})

    def test_isolated_zone_flagged_when_it_cannot_reach_outside(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        reachable = make_zone("Reachable")
        stuck = make_zone("Stuck")

        floor.add_zone(reachable)
        floor.add_zone(stuck)

        # Doors connect Reachable <-> Stuck, but nothing leads
        # Outside -- both zones should be flagged: Reachable because
        # it isn't part of the "outside" component either, Stuck for
        # the same reason (connected to something, just never to an
        # exit).
        door = Door(
            name="Internal Door",
            zone_a_id=reachable.id,
            zone_b_id=stuck.id,
            floor_id=floor.id,
        )
        floor.add_door(door)

        graph = NavigationGraphGenerator().build(building)
        report = graph.validate()

        isolated_ids = {
            issue.object_id
            for issue in report.by_code("isolated_zone")
        }

        self.assertEqual(isolated_ids, {reachable.id, stuck.id})

    def test_zone_without_any_connections_flagged(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        lonely = make_zone("Lonely")
        floor.add_zone(lonely)

        graph = NavigationGraphGenerator().build(building)
        report = graph.validate()

        issues = report.by_code("zone_without_connections")

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].object_id, lonely.id)

        # A zone with zero edges is trivially unreachable too, but it
        # should only be reported once (as "no connections"), not
        # again as "isolated".
        self.assertEqual(report.by_code("isolated_zone"), [])

    def test_disconnected_floor_flagged_when_no_stair_links_it(self):

        building = Building(name="B")
        ground = building.create_floor(name="Ground Floor")
        floor1 = building.create_floor(name="Floor 1")

        ground.add_zone(make_zone("Ground Zone"))
        floor1.add_zone(make_zone("Floor 1 Zone"))

        graph = NavigationGraphGenerator().build(building)
        report = graph.validate()

        disconnected_floor_ids = {
            issue.floor_id
            for issue in report.by_code("disconnected_floor")
        }

        self.assertEqual(
            disconnected_floor_ids,
            {ground.id, floor1.id},
        )


class DuplicateIdTests(unittest.TestCase):

    def test_duplicate_node_id_is_reported_and_first_wins(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        zone_1 = make_zone("First")
        zone_2 = make_zone("Second")
        zone_2.id = zone_1.id  # force a collision

        floor.add_zone(zone_1)
        floor.add_zone(zone_2)

        graph = NavigationGraphGenerator().build(building)
        report = graph.validate()

        self.assertFalse(report.is_valid)
        self.assertEqual(len(report.by_code("duplicate_node_id")), 1)

        self.assertIs(graph.find_node(zone_1.id).reference, zone_1)

    def test_duplicate_edge_id_is_reported(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        zone_a = make_zone("A")
        zone_b = make_zone("B")
        zone_c = make_zone("C")

        floor.add_zone(zone_a)
        floor.add_zone(zone_b)
        floor.add_zone(zone_c)

        door_1 = Door(
            name="D1",
            zone_a_id=zone_a.id,
            zone_b_id=zone_b.id,
            floor_id=floor.id,
        )

        door_2 = Door(
            name="D2",
            zone_a_id=zone_b.id,
            zone_b_id=zone_c.id,
            floor_id=floor.id,
        )
        door_2.id = door_1.id  # force a collision

        floor.add_door(door_1)
        floor.add_door(door_2)

        graph = NavigationGraphGenerator().build(building)
        report = graph.validate()

        self.assertFalse(report.is_valid)
        self.assertEqual(len(report.by_code("duplicate_edge_id")), 1)


class SaveLoadCompatibilityTests(unittest.TestCase):

    def test_graph_from_reloaded_project_matches_graph_from_original(self):

        project = Project.new_default()
        building = project.building

        ground = building.ordered_floors()[0]
        floor1 = building.create_floor(name="Floor 1", height=3.0)

        lobby = make_zone("Lobby")
        upstairs = make_zone("Upstairs")

        ground.add_zone(lobby)
        floor1.add_zone(upstairs)

        exit_obj = Exit(
            name="Front Exit",
            zone_id=lobby.id,
            floor_id=ground.id,
        )
        ground.add_exit(exit_obj)

        stair = Staircase(
            name="Main Stair",
            from_floor_id=ground.id,
            to_floor_id=floor1.id,
            from_zone_id=lobby.id,
            to_zone_id=upstairs.id,
        )
        ground.add_stair(stair)

        original_graph = NavigationGraphGenerator().build(building)
        original_report = original_graph.validate()

        # Round-trip through the exact on-disk representation
        # (Serializer writes/reads real JSON files) rather than only
        # exercising to_dict()/from_dict() in memory.
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:

            path = os.path.join(tmp_dir, "roundtrip.syn")

            Serializer.save(project, path)
            reloaded_project = Serializer.load(path)

        reloaded_graph = NavigationGraphGenerator().build(
            reloaded_project.building
        )
        reloaded_report = reloaded_graph.validate()

        self.assertEqual(
            sorted(original_graph.nodes.keys()),
            sorted(reloaded_graph.nodes.keys()),
        )

        self.assertEqual(
            sorted(
                (edge.edge_type, edge.from_node, edge.to_node)
                for edge in original_graph.edges
            ),
            sorted(
                (edge.edge_type, edge.from_node, edge.to_node)
                for edge in reloaded_graph.edges
            ),
        )

        self.assertEqual(
            original_report.is_valid,
            reloaded_report.is_valid,
        )

        self.assertEqual(
            len(original_report.issues),
            len(reloaded_report.issues),
        )


if __name__ == "__main__":
    unittest.main()
