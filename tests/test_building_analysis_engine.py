import unittest

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.staircase import Staircase
from models.zone import Zone

from navigation.edge import Edge
from navigation.graph_builder import NavigationGraphGenerator

from pathfinding.engine import PathfindingEngine

from analysis.engine import BuildingAnalysisEngine


def make_zone(name, **kwargs):

    fields = dict(x=0.0, y=0.0, width=2.0, height=2.0)
    fields.update(kwargs)

    return Zone(name=name, **fields)


class ExitServiceAreaTests(unittest.TestCase):

    # Two independent wings, each with its own exit: A near Exit1,
    # B/C near Exit2 (via a corridor). Nothing connects the wings
    # directly, so service areas must not cross.

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone_a = make_zone("A", x=0.0, y=0.0)
        self.zone_b = make_zone("B", x=50.0, y=0.0)
        self.zone_c = make_zone("C", x=55.0, y=0.0)

        for zone in (self.zone_a, self.zone_b, self.zone_c):
            self.floor.add_zone(zone)

        self.door_bc = Door(
            name="BC", zone_a_id=self.zone_b.id, zone_b_id=self.zone_c.id,
            floor_id=self.floor.id,
        )
        self.floor.add_door(self.door_bc)

        self.exit_1 = Exit(name="Exit1", zone_id=self.zone_a.id, floor_id=self.floor.id)
        self.exit_2 = Exit(name="Exit2", zone_id=self.zone_b.id, floor_id=self.floor.id)
        self.floor.add_exit(self.exit_1)
        self.floor.add_exit(self.exit_2)

        self.graph = NavigationGraphGenerator().build(self.building)
        self.analysis = BuildingAnalysisEngine(PathfindingEngine(self.graph))

    def test_zones_are_partitioned_by_nearest_exit(self):

        report = self.analysis.exit_service_areas()

        self.assertEqual(
            set(report.by_exit[self.exit_1.id].served_node_ids),
            {self.zone_a.id},
        )
        self.assertEqual(
            set(report.by_exit[self.exit_2.id].served_node_ids),
            {self.zone_b.id, self.zone_c.id},
        )

    def test_distances_are_reported_per_zone(self):

        report = self.analysis.exit_service_areas()

        area = report.by_exit[self.exit_2.id]

        self.assertLess(
            area.distances[self.zone_b.id],
            area.distances[self.zone_c.id],
        )

    def test_unreachable_zone_is_reported_as_unserved(self):

        lonely = make_zone("Lonely", x=200.0, y=200.0)
        self.floor.add_zone(lonely)

        graph = NavigationGraphGenerator().build(self.building)
        analysis = BuildingAnalysisEngine(PathfindingEngine(graph))

        report = analysis.exit_service_areas()

        self.assertIn(lonely.id, report.unserved_node_ids)


class MaxTravelDistanceTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.near = make_zone("Near", x=0.0, y=0.0)
        self.far = make_zone("Far", x=100.0, y=0.0)

        self.floor.add_zone(self.near)
        self.floor.add_zone(self.far)

        self.door = Door(
            name="D1", zone_a_id=self.near.id, zone_b_id=self.far.id,
            floor_id=self.floor.id,
        )
        self.floor.add_door(self.door)

        self.exit_obj = Exit(name="Ex1", zone_id=self.near.id, floor_id=self.floor.id)
        self.floor.add_exit(self.exit_obj)

        self.graph = NavigationGraphGenerator().build(self.building)
        self.analysis = BuildingAnalysisEngine(PathfindingEngine(self.graph))

    def test_worst_case_zone_is_identified(self):

        report = self.analysis.max_travel_distance()

        self.assertEqual(report.max_distance_node_id, self.far.id)
        self.assertEqual(report.max_distance, report.distances[self.far.id])

    def test_threshold_filters_exceeding_zones(self):

        report = self.analysis.max_travel_distance()
        far_distance = report.distances[self.far.id]

        below = self.analysis.max_travel_distance(threshold=far_distance + 1)
        self.assertEqual(below.exceeding_threshold_node_ids, [])

        above = self.analysis.max_travel_distance(threshold=far_distance - 1)
        self.assertIn(self.far.id, above.exceeding_threshold_node_ids)
        self.assertNotIn(self.near.id, above.exceeding_threshold_node_ids)

    def test_no_threshold_means_no_exceeding_list(self):

        report = self.analysis.max_travel_distance()

        self.assertEqual(report.exceeding_threshold_node_ids, [])


class CriticalConnectorTests(unittest.TestCase):

    def test_bridge_door_strands_a_whole_wing(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        lobby = make_zone("Lobby", x=0.0, y=0.0)
        bridge_zone = make_zone("Bridge", x=10.0, y=0.0)
        wing_zone = make_zone("Wing", x=20.0, y=0.0)

        floor.add_zone(lobby)
        floor.add_zone(bridge_zone)
        floor.add_zone(wing_zone)

        door_1 = Door(
            name="Lobby-Bridge", zone_a_id=lobby.id, zone_b_id=bridge_zone.id,
            floor_id=floor.id,
        )
        bridge_door = Door(
            name="Bridge-Wing", zone_a_id=bridge_zone.id, zone_b_id=wing_zone.id,
            floor_id=floor.id,
        )
        floor.add_door(door_1)
        floor.add_door(bridge_door)

        exit_obj = Exit(name="Ex1", zone_id=lobby.id, floor_id=floor.id)
        floor.add_exit(exit_obj)

        graph = NavigationGraphGenerator().build(building)
        analysis = BuildingAnalysisEngine(PathfindingEngine(graph))

        findings = analysis.critical_connectors()
        finding_by_edge = {f.edge_id: f for f in findings}

        self.assertIn(bridge_door.id, finding_by_edge)
        self.assertEqual(
            finding_by_edge[bridge_door.id].stranded_node_ids,
            [wing_zone.id],
        )

    def test_redundant_loop_has_no_critical_connectors(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        lobby = make_zone("Lobby", x=0.0, y=0.0)
        room = make_zone("Room", x=10.0, y=0.0)

        floor.add_zone(lobby)
        floor.add_zone(room)

        # Two independent doors between the same two zones -- a loop,
        # so removing either one alone strands nothing.
        door_1 = Door(
            name="D1", zone_a_id=lobby.id, zone_b_id=room.id, floor_id=floor.id,
        )
        door_2 = Door(
            name="D2", zone_a_id=lobby.id, zone_b_id=room.id, floor_id=floor.id,
        )
        floor.add_door(door_1)
        floor.add_door(door_2)

        exit_obj = Exit(name="Ex1", zone_id=lobby.id, floor_id=floor.id)
        floor.add_exit(exit_obj)

        graph = NavigationGraphGenerator().build(building)
        analysis = BuildingAnalysisEngine(PathfindingEngine(graph))

        findings = analysis.critical_connectors()
        finding_edge_ids = {f.edge_id for f in findings}

        self.assertNotIn(door_1.id, finding_edge_ids)
        self.assertNotIn(door_2.id, finding_edge_ids)

    def test_findings_are_sorted_by_impact_descending(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        lobby = make_zone("Lobby", x=0.0, y=0.0)
        small_branch = make_zone("SmallBranch", x=10.0, y=0.0)
        big_branch_1 = make_zone("BigBranch1", x=20.0, y=0.0)
        big_branch_2 = make_zone("BigBranch2", x=30.0, y=0.0)

        for zone in (lobby, small_branch, big_branch_1, big_branch_2):
            floor.add_zone(zone)

        small_bridge = Door(
            name="Small", zone_a_id=lobby.id, zone_b_id=small_branch.id,
            floor_id=floor.id,
        )
        big_bridge = Door(
            name="Big", zone_a_id=lobby.id, zone_b_id=big_branch_1.id,
            floor_id=floor.id,
        )
        chain = Door(
            name="Chain", zone_a_id=big_branch_1.id, zone_b_id=big_branch_2.id,
            floor_id=floor.id,
        )
        floor.add_door(small_bridge)
        floor.add_door(big_bridge)
        floor.add_door(chain)

        exit_obj = Exit(name="Ex1", zone_id=lobby.id, floor_id=floor.id)
        floor.add_exit(exit_obj)

        graph = NavigationGraphGenerator().build(building)
        analysis = BuildingAnalysisEngine(PathfindingEngine(graph))

        findings = analysis.critical_connectors()

        impacts = [len(f.stranded_node_ids) for f in findings]
        self.assertEqual(impacts, sorted(impacts, reverse=True))

        # The sole Exit is the single most critical connector of all
        # (removing it strands the entire building, lobby included) --
        # ranked above big_bridge, which only strands its own branch.
        self.assertEqual(findings[0].edge_id, exit_obj.id)
        self.assertEqual(len(findings[0].stranded_node_ids), 4)

        self.assertEqual(findings[1].edge_id, big_bridge.id)
        self.assertEqual(len(findings[1].stranded_node_ids), 2)


class StairDependencyTests(unittest.TestCase):

    def test_sole_stair_is_flagged_as_a_dependency(self):

        building = Building(name="B")
        ground = building.create_floor(name="Ground Floor")
        floor1 = building.create_floor(name="Floor 1", height=3.0)

        lobby = make_zone("Lobby", x=0.0, y=0.0)
        upstairs = make_zone("Upstairs", x=0.0, y=0.0, floor_id=floor1.id)

        ground.add_zone(lobby)
        floor1.add_zone(upstairs)

        exit_obj = Exit(name="Ex1", zone_id=lobby.id, floor_id=ground.id)
        ground.add_exit(exit_obj)

        stair = Staircase(
            name="S1", from_floor_id=ground.id, to_floor_id=floor1.id,
            from_zone_id=lobby.id, to_zone_id=upstairs.id,
        )
        ground.add_stair(stair)

        graph = NavigationGraphGenerator().build(building)
        analysis = BuildingAnalysisEngine(PathfindingEngine(graph))

        findings = analysis.stair_dependency()

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].edge_id, stair.id)
        self.assertEqual(findings[0].stranded_node_ids, [upstairs.id])

    def test_second_stair_removes_the_dependency(self):

        building = Building(name="B")
        ground = building.create_floor(name="Ground Floor")
        floor1 = building.create_floor(name="Floor 1", height=3.0)

        lobby = make_zone("Lobby", x=0.0, y=0.0)
        upstairs = make_zone("Upstairs", x=0.0, y=0.0, floor_id=floor1.id)

        ground.add_zone(lobby)
        floor1.add_zone(upstairs)

        exit_obj = Exit(name="Ex1", zone_id=lobby.id, floor_id=ground.id)
        ground.add_exit(exit_obj)

        stair_1 = Staircase(
            name="S1", from_floor_id=ground.id, to_floor_id=floor1.id,
            from_zone_id=lobby.id, to_zone_id=upstairs.id,
        )
        stair_2 = Staircase(
            name="S2", from_floor_id=ground.id, to_floor_id=floor1.id,
            from_zone_id=lobby.id, to_zone_id=upstairs.id,
        )
        ground.add_stair(stair_1)
        ground.add_stair(stair_2)

        graph = NavigationGraphGenerator().build(building)
        analysis = BuildingAnalysisEngine(PathfindingEngine(graph))

        findings = analysis.stair_dependency()

        self.assertEqual(findings, [])

    def test_stair_dependency_only_includes_stair_edges(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        lobby = make_zone("Lobby", x=0.0, y=0.0)
        room = make_zone("Room", x=10.0, y=0.0)

        floor.add_zone(lobby)
        floor.add_zone(room)

        bridge_door = Door(
            name="D1", zone_a_id=lobby.id, zone_b_id=room.id, floor_id=floor.id,
        )
        floor.add_door(bridge_door)

        exit_obj = Exit(name="Ex1", zone_id=lobby.id, floor_id=floor.id)
        floor.add_exit(exit_obj)

        graph = NavigationGraphGenerator().build(building)
        analysis = BuildingAnalysisEngine(PathfindingEngine(graph))

        # bridge_door IS a critical connector, but not a Stair, so it
        # must not show up in stair_dependency().
        self.assertTrue(
            any(f.edge_id == bridge_door.id for f in analysis.critical_connectors())
        )
        self.assertEqual(analysis.stair_dependency(), [])


class ConnectivityReportTests(unittest.TestCase):

    def test_report_composes_validation_and_analyses_without_reimplementing_them(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        lonely = make_zone("Lonely", x=0.0, y=0.0)
        floor.add_zone(lonely)

        graph = NavigationGraphGenerator().build(building)
        analysis = BuildingAnalysisEngine(PathfindingEngine(graph))

        report = analysis.connectivity_report()

        # graph.validate() is surfaced verbatim.
        direct_validation = graph.validate()
        self.assertEqual(
            [i.code for i in report.validation.issues],
            [i.code for i in direct_validation.issues],
        )

        self.assertIsNotNone(report.travel_distance)
        self.assertEqual(report.critical_connectors, [])
        self.assertEqual(report.stair_dependencies, [])

    def test_stair_dependencies_are_a_subset_of_critical_connectors(self):

        building = Building(name="B")
        ground = building.create_floor(name="Ground Floor")
        floor1 = building.create_floor(name="Floor 1", height=3.0)

        lobby = make_zone("Lobby", x=0.0, y=0.0)
        upstairs = make_zone("Upstairs", x=0.0, y=0.0, floor_id=floor1.id)

        ground.add_zone(lobby)
        floor1.add_zone(upstairs)

        exit_obj = Exit(name="Ex1", zone_id=lobby.id, floor_id=ground.id)
        ground.add_exit(exit_obj)

        stair = Staircase(
            name="S1", from_floor_id=ground.id, to_floor_id=floor1.id,
            from_zone_id=lobby.id, to_zone_id=upstairs.id,
        )
        ground.add_stair(stair)

        graph = NavigationGraphGenerator().build(building)
        analysis = BuildingAnalysisEngine(PathfindingEngine(graph))

        report = analysis.connectivity_report()

        critical_ids = {f.edge_id for f in report.critical_connectors}
        stair_ids = {f.edge_id for f in report.stair_dependencies}

        self.assertTrue(stair_ids.issubset(critical_ids))
        self.assertEqual(stair_ids, {stair.id})


class EngineeringModelIndependenceTests(unittest.TestCase):

    def test_analysis_package_never_touches_reference_or_engineering_models(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "analysis"

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertNotIn(
                ".reference", text, f"{path} touches .reference directly"
            )
            self.assertIsNone(
                re.search(r"^\s*(from|import)\s+(models|designer)\b", text, re.MULTILINE),
                f"{path} imports models/designer directly",
            )


if __name__ == "__main__":
    unittest.main()
