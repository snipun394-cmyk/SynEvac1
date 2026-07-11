import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import unittest

from PyQt6.QtWidgets import QApplication

from models.assembly_point import AssemblyPoint
from models.building import Building
from models.door import Door
from models.project import Project
from models.zone import Zone

from navigation.edge import Edge
from navigation.graph_builder import NavigationGraphGenerator
from navigation.node import Node

from serialization.serializer import Serializer

_app = QApplication.instance() or QApplication([])

from designer.items.door_item import DoorItem
from designer.widgets.property_panel import PropertyPanel


def make_zone(name, **kwargs):

    return Zone(
        name=name,
        x=0.0,
        y=0.0,
        width=2.0,
        height=2.0,
        **kwargs,
    )


def make_assembly_point(name, **kwargs):

    return AssemblyPoint(
        name=name,
        position=(0.0, 0.0),
        **kwargs,
    )


class ZoneToZoneRegressionTests(unittest.TestCase):

    # Locks in that generalizing Door connectivity did not change
    # anything about the pre-existing Zone <-> Zone case.

    def test_zone_to_zone_door_still_produces_the_same_edge_shape(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        zone_a = make_zone("Lobby")
        zone_b = make_zone("Corridor")
        floor.add_zone(zone_a)
        floor.add_zone(zone_b)

        door = Door(
            name="D1",
            zone_a_id=zone_a.id,
            zone_b_id=zone_b.id,
            floor_id=floor.id,
        )
        floor.add_door(door)

        graph = NavigationGraphGenerator().build(building)

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
            {zone_a.id, zone_b.id},
        )

        # Both zones have a connection (the door itself), so neither
        # is "without connections" -- whether they can also reach
        # Outside is a separate concern this test isn't about (there
        # is deliberately no Exit here).
        report = graph.validate()
        self.assertTrue(report.is_valid)
        self.assertEqual(report.by_code("zone_without_connections"), [])

    def test_door_missing_zone_a_still_uses_the_same_code_and_message(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        zone_b = make_zone("Corridor")
        floor.add_zone(zone_b)

        door = Door(name="D1", zone_b_id=zone_b.id, floor_id=floor.id)
        floor.add_door(door)

        graph = NavigationGraphGenerator().build(building)
        report = graph.validate()

        issues = report.by_code("door_missing_zone_a")
        self.assertEqual(len(issues), 1)
        self.assertEqual(
            issues[0].message,
            "Door 'D1' has no Zone A assigned.",
        )


class ZoneToAssemblyPointTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone = make_zone("Lobby")
        self.assembly_point = make_assembly_point("Muster Area")

        self.floor.add_zone(self.zone)
        self.floor.add_assembly_point(self.assembly_point)

    def test_assembly_point_node_is_created(self):

        graph = NavigationGraphGenerator().build(self.building)

        node = graph.find_node(self.assembly_point.id)

        self.assertIsNotNone(node)
        self.assertEqual(node.node_type, Node.ASSEMBLY_POINT)
        self.assertIs(node.reference, self.assembly_point)

    def test_door_connects_zone_to_assembly_point(self):

        door = Door(
            name="Gate",
            zone_a_id=self.zone.id,
            zone_b_id=self.assembly_point.id,
            floor_id=self.floor.id,
        )
        self.floor.add_door(door)

        graph = NavigationGraphGenerator().build(self.building)

        door_edges = [
            edge
            for edge in graph.edges
            if edge.edge_type == Edge.DOOR
        ]

        self.assertEqual(len(door_edges), 1)

        edge = door_edges[0]
        self.assertEqual(
            {edge.from_node, edge.to_node},
            {self.zone.id, self.assembly_point.id},
        )

        report = graph.validate()
        self.assertTrue(report.is_valid)

    def test_assembly_point_traversable_as_a_neighbor(self):

        door = Door(
            name="Gate",
            zone_a_id=self.zone.id,
            zone_b_id=self.assembly_point.id,
            floor_id=self.floor.id,
        )
        self.floor.add_door(door)

        graph = NavigationGraphGenerator().build(self.building)

        zone_node = graph.find_node(self.zone.id)
        neighbor_ids = {
            node.id
            for node, _ in graph.find_neighbors(zone_node)
        }

        self.assertIn(self.assembly_point.id, neighbor_ids)

    def test_assembly_point_missing_from_door_uses_zone_slot_code(self):

        door = Door(
            name="Gate",
            zone_a_id=self.zone.id,
            floor_id=self.floor.id,
        )
        self.floor.add_door(door)

        graph = NavigationGraphGenerator().build(self.building)
        report = graph.validate()

        self.assertEqual(len(report.by_code("door_missing_zone_b")), 1)

    def test_dangling_assembly_point_reference_is_invalid(self):

        door = Door(
            name="Gate",
            zone_a_id=self.zone.id,
            zone_b_id="not-a-real-id",
            floor_id=self.floor.id,
        )
        self.floor.add_door(door)

        graph = NavigationGraphGenerator().build(self.building)
        report = graph.validate()

        self.assertFalse(report.is_valid)
        self.assertEqual(len(report.by_code("invalid_reference")), 1)


class AssemblyPointToAssemblyPointTests(unittest.TestCase):

    def test_two_assembly_points_can_be_connected_by_a_door(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        ap_1 = make_assembly_point("AP-1")
        ap_2 = make_assembly_point("Muster Area")

        floor.add_assembly_point(ap_1)
        floor.add_assembly_point(ap_2)

        door = Door(
            name="Connector",
            zone_a_id=ap_1.id,
            zone_b_id=ap_2.id,
            floor_id=floor.id,
        )
        floor.add_door(door)

        graph = NavigationGraphGenerator().build(building)

        door_edges = [
            edge
            for edge in graph.edges
            if edge.edge_type == Edge.DOOR
        ]

        self.assertEqual(len(door_edges), 1)
        self.assertEqual(
            {door_edges[0].from_node, door_edges[0].to_node},
            {ap_1.id, ap_2.id},
        )

        report = graph.validate()
        self.assertTrue(report.is_valid)


class ExitAndStairRemainZoneOnlyTests(unittest.TestCase):

    # Generalization was scoped to Door only -- Exit/Stair must keep
    # rejecting an Assembly Point exactly like any other invalid
    # reference.

    def test_exit_cannot_resolve_to_an_assembly_point(self):

        from models.exit import Exit

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        ap = make_assembly_point("AP-1")
        floor.add_assembly_point(ap)

        exit_obj = Exit(name="Ex1", zone_id=ap.id, floor_id=floor.id)
        floor.add_exit(exit_obj)

        graph = NavigationGraphGenerator().build(building)
        report = graph.validate()

        self.assertFalse(report.is_valid)
        self.assertEqual(len(report.by_code("invalid_reference")), 1)
        self.assertEqual(
            [
                edge
                for edge in graph.edges
                if edge.edge_type == Edge.EXIT
            ],
            [],
        )


class AssemblyPointConnectivityValidationTests(unittest.TestCase):

    def test_assembly_point_without_connections_is_flagged(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        ap = make_assembly_point("Lonely AP")
        floor.add_assembly_point(ap)

        graph = NavigationGraphGenerator().build(building)
        report = graph.validate()

        issues = report.by_code("zone_without_connections")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].object_id, ap.id)
        self.assertIn("Assembly Point", issues[0].message)

    def test_assembly_point_unreachable_from_outside_is_isolated(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        zone = make_zone("Sealed Room")
        ap = make_assembly_point("Sealed AP")

        floor.add_zone(zone)
        floor.add_assembly_point(ap)

        door = Door(
            name="D1",
            zone_a_id=zone.id,
            zone_b_id=ap.id,
            floor_id=floor.id,
        )
        floor.add_door(door)

        graph = NavigationGraphGenerator().build(building)
        report = graph.validate()

        isolated_ids = {
            issue.object_id
            for issue in report.by_code("isolated_zone")
        }

        self.assertEqual(isolated_ids, {zone.id, ap.id})


class PropertyPanelAssemblyPointComboTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone = make_zone("Lobby")
        self.assembly_point = make_assembly_point("AP-1")

        self.floor.add_zone(self.zone)
        self.floor.add_assembly_point(self.assembly_point)

        self.door = Door(name="D1", floor_id=self.floor.id)
        self.floor.add_door(self.door)

        self.door_item = DoorItem(0, 0, 100, 0, model=self.door)

        self.panel = PropertyPanel()
        self.panel.set_building(self.building)

    def combo_entries(self, combo):

        return [
            (combo.itemText(i), combo.itemData(i))
            for i in range(combo.count())
        ]

    def test_combo_lists_both_zone_and_assembly_point_with_type_labels(self):

        self.panel.show_door(self.door_item)

        texts = [
            text
            for text, _ in self.combo_entries(self.panel.door_zone_a)
        ]

        self.assertIn("Zone: Lobby", texts)
        self.assertIn("Assembly Point: AP-1", texts)

        for text in texts:
            self.assertNotEqual(text, self.zone.id)
            self.assertNotEqual(text, self.assembly_point.id)

    def test_selecting_assembly_point_writes_back_plain_id(self):

        self.panel.show_door(self.door_item)

        index = self.panel.door_zone_b.findData(self.assembly_point.id)
        self.assertNotEqual(index, -1)

        self.panel.door_zone_b.setCurrentIndex(index)
        self.panel.update_door_zone_b(index)

        self.assertEqual(self.door.zone_b_id, self.assembly_point.id)

    def test_current_assembly_point_selection_is_preselected_on_refresh(self):

        self.door.zone_a_id = self.assembly_point.id

        self.panel.show_door(self.door_item)

        current_data = self.panel.door_zone_a.itemData(
            self.panel.door_zone_a.currentIndex()
        )

        self.assertEqual(current_data, self.assembly_point.id)


class SaveLoadCompatibilityTests(unittest.TestCase):

    def test_zone_to_assembly_point_door_round_trips_through_disk(self):

        project = Project.new_default()
        floor = project.building.ordered_floors()[0]

        zone = make_zone("Lobby")
        ap = make_assembly_point("AP-1")

        floor.add_zone(zone)
        floor.add_assembly_point(ap)

        door = Door(
            name="Gate",
            zone_a_id=zone.id,
            zone_b_id=ap.id,
            floor_id=floor.id,
        )
        floor.add_door(door)

        original_graph = NavigationGraphGenerator().build(project.building)
        original_report = original_graph.validate()

        with tempfile.TemporaryDirectory() as tmp_dir:

            path = os.path.join(tmp_dir, "roundtrip.syn")

            Serializer.save(project, path)
            reloaded_project = Serializer.load(path)

        reloaded_graph = NavigationGraphGenerator().build(
            reloaded_project.building
        )
        reloaded_report = reloaded_graph.validate()

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

        self.assertEqual(original_report.is_valid, reloaded_report.is_valid)
        self.assertEqual(
            len(original_report.issues),
            len(reloaded_report.issues),
        )

        reloaded_ap_node = reloaded_graph.find_node(ap.id)
        self.assertIsNotNone(reloaded_ap_node)
        self.assertEqual(reloaded_ap_node.node_type, Node.ASSEMBLY_POINT)
        self.assertEqual(reloaded_ap_node.name, "AP-1")


if __name__ == "__main__":
    unittest.main()
