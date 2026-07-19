import csv
import tempfile
import unittest
from pathlib import Path

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
from navigation.node import Node

from pathfinding.route import Route

from scenario.engineering_state import DoorState, ScenarioDoorState, ScenarioExitState
from scenario.fire import ScenarioFire
from scenario.metadata import ScenarioMetadata
from scenario.occupant import ScenarioOccupant
from scenario.scenario import Scenario

from simulator.multi_agent_result import (
    MultiAgentSimulationResult,
    OccupantTimeline,
    OccupantTimelineStep,
)
from simulator.occupant import OccupantState

from dataset_builder import (
    DatasetBuilder,
    SimulationRun,
    export_csv,
    extract_scenario_features,
    extract_simulation_outcome,
    extract_zone_results,
)
from dataset_builder.schema import (
    camera_state_columns,
    door_state_columns,
    detector_state_columns,
    exit_state_columns,
    obstacle_state_columns,
    stair_state_columns,
    zone_occupancy_columns,
)


# =====================================================
# Fixtures -- plain module-level functions with **overrides, matching
# the rest of tests/*.py. Scenario/Building/MultiAgentSimulationResult
# are all constructed directly rather than run through the real
# generator/simulator pipelines, since the Dataset Builder only ever
# needs to read their finished shape.
# =====================================================


def make_building(zone_count=2, door_count=1, exit_count=1, stair_count=1,
                   obstacle_count=1, detector_count=1, camera_count=1):

    zones = [
        Zone(id=f"zone-{i}", name=f"Zone {i}", x=0.0, y=0.0, width=4.0, height=5.0)
        for i in range(zone_count)
    ]

    doors = [
        Door(id=f"door-{i}", floor_id="floor-1", active=True, locked=False)
        for i in range(door_count)
    ]

    exits = [
        Exit(id=f"exit-{i}", floor_id="floor-1", zone_id=zones[-1].id if zones else "",
             is_blocked=False)
        for i in range(exit_count)
    ]

    stairs = [
        Staircase(id=f"stair-{i}", from_floor_id="floor-1", to_floor_id="floor-1")
        for i in range(stair_count)
    ]

    obstacles = [
        Obstacle(id=f"obstacle-{i}", floor_id="floor-1", active=True)
        for i in range(obstacle_count)
    ]

    detectors = [
        Detector(id=f"detector-{i}", floor_id="floor-1", active=True)
        for i in range(detector_count)
    ]

    cameras = [
        Camera(id=f"camera-{i}", floor_id="floor-1", active=True)
        for i in range(camera_count)
    ]

    floor = Floor(
        name="Ground", id="floor-1", display_order=0,
        zones=zones, doors=doors, exits=exits, stairs=stairs,
        obstacles=obstacles, detectors=detectors, cameras=cameras,
    )

    return Building(name="Test Building", id="building-1", floors=[floor])


def make_metadata(**overrides):

    defaults = dict(
        scenario_id="scn-1", definition_id="def-1",
        definition_content_hash="hash-1", generation_version="v1",
        seed=42, created_at="2026-01-01T00:00:00",
    )
    defaults.update(overrides)

    return ScenarioMetadata(**defaults)


def make_occupant(occupant_id, zone_id, **overrides):

    defaults = dict(
        occupant_id=occupant_id, zone_id=zone_id, floor_id="floor-1",
        position=(0.0, 0.0), behaviour_profile_id="Adult_Default",
    )
    defaults.update(overrides)

    return ScenarioOccupant(**defaults)


def make_scenario(building, occupants=None, fire=None, **state_overrides):

    if occupants is None:
        occupants = (
            make_occupant("occ-1", building.floors[0].zones[0].id),
            make_occupant("occ-2", building.floors[0].zones[0].id),
        )

    if fire is None and building.floors[0].zones:
        fire = ScenarioFire(
            ignition_zone_id=building.floors[0].zones[0].id,
            ignition_floor_id="floor-1", fire_profile="Electrical",
            growth_parameters={"growth_time": 250.0},
        )

    return Scenario(
        metadata=make_metadata(), occupants=occupants, fire=fire,
        door_states=state_overrides.get("door_states", ()),
        exit_states=state_overrides.get("exit_states", ()),
        stair_states=state_overrides.get("stair_states", ()),
        obstacle_states=state_overrides.get("obstacle_states", ()),
        camera_states=state_overrides.get("camera_states", ()),
        detector_states=state_overrides.get("detector_states", ()),
    )


def make_zone_node(zone):

    return Node(id=zone.id, name=zone.name, floor_id=zone.floor_id, node_type=Node.ZONE)


def make_outside_node():

    return Node(id=Node.OUTSIDE_NODE_ID, name="Outside", floor_id="", node_type=Node.OUTSIDE)


def make_evacuation_timeline(occupant_id, from_zone, exit_obj, depart_time=0.0,
                             queue_wait_time=0.0, travel_time=10.0):

    # A single-hop route straight from the occupant's zone out through
    # `exit_obj` -- enough to exercise route/edge-based extraction
    # (exit used, route length) without needing a real pathfinder.

    from_node = make_zone_node(from_zone)
    to_node = make_outside_node()

    edge = Edge(id=exit_obj.id, edge_type=Edge.EXIT, from_node=from_node.id,
                to_node=to_node.id, walking_distance=8.0)

    route = Route(nodes=[from_node, to_node], edges=[edge], total_cost=8.0, total_distance=8.0)

    step = OccupantTimelineStep(
        index=0, from_node=from_node, to_node=to_node, edge=edge,
        queue_wait_time=queue_wait_time, start_time=depart_time,
        end_time=depart_time + travel_time,
    )

    return OccupantTimeline(
        occupant_id=occupant_id, route=route, steps=[step],
        state=OccupantState.ARRIVED, depart_time=depart_time,
        arrival_time=depart_time + travel_time,
    )


def make_unreachable_timeline(occupant_id):

    return OccupantTimeline(
        occupant_id=occupant_id, route=None, steps=[],
        state=OccupantState.UNREACHABLE, depart_time=0.0, arrival_time=None,
    )


def make_stationary_timeline(occupant_id, depart_time=0.0):

    return OccupantTimeline(
        occupant_id=occupant_id, route=None, steps=[],
        state=OccupantState.STATIONARY, depart_time=depart_time, arrival_time=None,
    )


# =====================================================


class FeatureExtractionTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()
        self.zone_a, self.zone_b = self.building.floors[0].zones
        self.exit_obj = self.building.floors[0].exits[0]

        self.scenario = make_scenario(
            self.building,
            occupants=(
                make_occupant("occ-1", self.zone_a.id),
                make_occupant("occ-2", self.zone_a.id),
                make_occupant("occ-3", self.zone_b.id),
            ),
        )

        self.run = SimulationRun(
            scenario=self.scenario, building=self.building,
            movement_result=MultiAgentSimulationResult(occupants={}, total_evacuation_time=None),
        )

    def test_metadata_fields(self):

        row = extract_scenario_features(self.run)

        self.assertEqual(row["scenario_id"], "scn-1")
        self.assertEqual(row["definition_id"], "def-1")
        self.assertEqual(row["seed"], 42)

    def test_fire_fields(self):

        row = extract_scenario_features(self.run)

        self.assertEqual(row["ignition_zone"], self.zone_a.id)
        self.assertEqual(row["ignition_floor"], "floor-1")
        self.assertEqual(row["fire_profile"], "Electrical")
        self.assertEqual(row["growth_time"], 250.0)

    def test_total_and_per_zone_occupancy(self):

        row = extract_scenario_features(self.run)

        self.assertEqual(row["total_occupants"], 3)
        self.assertEqual(row["Zone_1_Occupancy"], 2)
        self.assertEqual(row["Zone_2_Occupancy"], 1)

    def test_engineering_state_falls_back_to_building_baseline(self):

        # No Scenario override for door-0/exit-0/... -- values must
        # come from the Building model's own baseline fields.
        row = extract_scenario_features(self.run)

        self.assertEqual(row["Door_1_State"], "OPEN")
        self.assertEqual(row["Exit_1_State"], "OPEN")
        self.assertEqual(row["Stair_1_State"], "AVAILABLE")
        self.assertEqual(row["Obstacle_1_State"], "ACTIVE")
        self.assertEqual(row["Detector_1_State"], "AVAILABLE")
        self.assertEqual(row["Camera_1_State"], "AVAILABLE")

    def test_engineering_state_prefers_scenario_override(self):

        door = self.building.floors[0].doors[0]
        exit_obj = self.building.floors[0].exits[0]

        scenario = make_scenario(
            self.building,
            door_states=(ScenarioDoorState(door_id=door.id, state=DoorState.LOCKED),),
            exit_states=(ScenarioExitState(exit_id=exit_obj.id, is_open=False),),
        )

        run = SimulationRun(
            scenario=scenario, building=self.building,
            movement_result=MultiAgentSimulationResult(occupants={}, total_evacuation_time=None),
        )

        row = extract_scenario_features(run)

        self.assertEqual(row["Door_1_State"], "LOCKED")
        self.assertEqual(row["Exit_1_State"], "CLOSED")

    def test_baseline_reflects_a_closed_or_failed_building_model(self):

        building = make_building()
        building.floors[0].doors[0].active = False
        building.floors[0].detectors[0].active = False
        building.floors[0].cameras[0].active = False
        building.floors[0].obstacles[0].active = False

        scenario = make_scenario(building)
        run = SimulationRun(
            scenario=scenario, building=building,
            movement_result=MultiAgentSimulationResult(occupants={}, total_evacuation_time=None),
        )

        row = extract_scenario_features(run)

        self.assertEqual(row["Door_1_State"], "CLOSED")
        self.assertEqual(row["Detector_1_State"], "FAILED")
        self.assertEqual(row["Camera_1_State"], "FAILED")
        self.assertEqual(row["Obstacle_1_State"], "INACTIVE")


# =====================================================


class VariableCountTests(unittest.TestCase):

    def test_zero_and_multiple_zones_produce_matching_columns(self):

        building = make_building(zone_count=3, door_count=0, exit_count=2,
                                  stair_count=0, obstacle_count=2,
                                  detector_count=0, camera_count=2)

        self.assertEqual(
            zone_occupancy_columns(building),
            ["Zone_1_Occupancy", "Zone_2_Occupancy", "Zone_3_Occupancy"],
        )
        self.assertEqual(door_state_columns(building), [])
        self.assertEqual(
            exit_state_columns(building), ["Exit_1_State", "Exit_2_State"],
        )
        self.assertEqual(stair_state_columns(building), [])
        self.assertEqual(
            obstacle_state_columns(building),
            ["Obstacle_1_State", "Obstacle_2_State"],
        )
        self.assertEqual(detector_state_columns(building), [])
        self.assertEqual(
            camera_state_columns(building), ["Camera_1_State", "Camera_2_State"],
        )

    def test_extraction_works_with_no_engineering_objects_at_all(self):

        building = make_building(zone_count=1, door_count=0, exit_count=0,
                                  stair_count=0, obstacle_count=0,
                                  detector_count=0, camera_count=0)

        scenario = make_scenario(
            building, occupants=(make_occupant("occ-1", building.floors[0].zones[0].id),),
        )

        run = SimulationRun(
            scenario=scenario, building=building,
            movement_result=MultiAgentSimulationResult(occupants={}, total_evacuation_time=None),
        )

        row = extract_scenario_features(run)

        self.assertEqual(row["Zone_1_Occupancy"], 1)
        self.assertNotIn("Door_1_State", row)
        self.assertNotIn("Exit_1_State", row)


# =====================================================


class SimulationOutcomeTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building(zone_count=2, exit_count=1)
        self.zone_a, self.zone_b = self.building.floors[0].zones
        self.exit_obj = self.building.floors[0].exits[0]

        self.scenario = make_scenario(
            self.building,
            occupants=(
                make_occupant("evacuated-1", self.zone_a.id),
                make_occupant("evacuated-2", self.zone_a.id),
                make_occupant("unreachable-1", self.zone_b.id),
            ),
        )

        timelines = {
            "evacuated-1": make_evacuation_timeline(
                "evacuated-1", self.zone_a, self.exit_obj, depart_time=0.0, travel_time=10.0,
            ),
            "evacuated-2": make_evacuation_timeline(
                "evacuated-2", self.zone_a, self.exit_obj, depart_time=0.0,
                queue_wait_time=2.0, travel_time=20.0,
            ),
            "unreachable-1": make_unreachable_timeline("unreachable-1"),
        }

        self.movement_result = MultiAgentSimulationResult(
            occupants=timelines, total_evacuation_time=20.0,
            unreachable_occupant_ids=["unreachable-1"],
            peak_edge_occupancy={self.exit_obj.id: 2},
            peak_node_occupancy={self.zone_a.id: 2, self.zone_b.id: 1},
        )

        self.run = SimulationRun(
            scenario=self.scenario, building=self.building,
            movement_result=self.movement_result, simulation_finished=True,
        )

    def test_evacuation_counts(self):

        row = extract_simulation_outcome(self.run)

        self.assertEqual(row["people_evacuated"], 2)
        self.assertEqual(row["people_trapped"], 1)
        self.assertEqual(row["reachable_occupants"], 2)
        self.assertEqual(row["unreachable_occupants"], 1)
        self.assertFalse(row["building_cleared"])
        self.assertTrue(row["simulation_finished"])

    def test_evacuation_times(self):

        row = extract_simulation_outcome(self.run)

        self.assertEqual(row["total_evacuation_time"], 20.0)
        self.assertEqual(row["last_occupant_exit_time"], 20.0)
        self.assertEqual(row["average_evacuation_time"], 15.0)

    def test_building_cleared_when_nobody_trapped(self):

        scenario = make_scenario(
            self.building, occupants=(make_occupant("evacuated-1", self.zone_a.id),),
        )
        movement_result = MultiAgentSimulationResult(
            occupants={
                "evacuated-1": make_evacuation_timeline(
                    "evacuated-1", self.zone_a, self.exit_obj,
                ),
            },
            total_evacuation_time=10.0,
        )

        row = extract_simulation_outcome(
            SimulationRun(scenario=scenario, building=self.building, movement_result=movement_result),
        )

        self.assertTrue(row["building_cleared"])

    def test_congestion_and_queue_metrics(self):

        row = extract_simulation_outcome(self.run)

        self.assertEqual(row["maximum_congestion"], 2)
        self.assertEqual(row["most_congested_exit"], self.exit_obj.id)
        self.assertIsNone(row["most_congested_stair"])
        self.assertEqual(row["maximum_queue_length"], 1)

    def test_maximum_density_derived_from_peak_occupancy_and_zone_area(self):

        row = extract_simulation_outcome(self.run)

        expected = 2 / (self.zone_a.width * self.zone_a.height)
        self.assertAlmostEqual(row["maximum_density"], expected)


# =====================================================


class ZoneResultTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building(zone_count=2, exit_count=1)
        self.zone_a, self.zone_b = self.building.floors[0].zones
        self.exit_obj = self.building.floors[0].exits[0]

        self.scenario = make_scenario(
            self.building,
            occupants=(
                make_occupant("evacuated-1", self.zone_a.id),
                make_occupant("stuck-1", self.zone_a.id),
                make_occupant("unreachable-1", self.zone_b.id),
            ),
        )

        self.timelines = {
            "evacuated-1": make_evacuation_timeline(
                "evacuated-1", self.zone_a, self.exit_obj, depart_time=1.0, travel_time=9.0,
            ),
            "stuck-1": make_stationary_timeline("stuck-1", depart_time=0.0),
            "unreachable-1": make_unreachable_timeline("unreachable-1"),
        }

        self.movement_result = MultiAgentSimulationResult(
            occupants=self.timelines, total_evacuation_time=10.0,
            unreachable_occupant_ids=["unreachable-1"],
        )

        self.run = SimulationRun(
            scenario=self.scenario, building=self.building, movement_result=self.movement_result,
        )

    def test_one_row_per_zone(self):

        rows = extract_zone_results(self.run)

        self.assertEqual(len(rows), 2)
        self.assertEqual({row["zone_id"] for row in rows}, {self.zone_a.id, self.zone_b.id})

    def test_zone_a_counts(self):

        rows = {row["zone_id"]: row for row in extract_zone_results(self.run)}
        zone_a_row = rows[self.zone_a.id]

        self.assertEqual(zone_a_row["initial_occupants"], 2)
        self.assertEqual(zone_a_row["evacuated"], 1)
        self.assertEqual(zone_a_row["trapped"], 1)

        # "stuck-1" never departed and has no steps -- still physically
        # in zone_a, so it counts toward final_occupants there.
        self.assertEqual(zone_a_row["final_occupants"], 1)

    def test_zone_b_counts_unreachable_occupant_as_still_present(self):

        rows = {row["zone_id"]: row for row in extract_zone_results(self.run)}
        zone_b_row = rows[self.zone_b.id]

        self.assertEqual(zone_b_row["initial_occupants"], 1)
        self.assertEqual(zone_b_row["evacuated"], 0)
        self.assertEqual(zone_b_row["trapped"], 1)
        self.assertEqual(zone_b_row["final_occupants"], 1)

    def test_exit_used_and_route_length_from_evacuated_occupant(self):

        rows = {row["zone_id"]: row for row in extract_zone_results(self.run)}
        zone_a_row = rows[self.zone_a.id]

        self.assertEqual(zone_a_row["exit_used"], self.exit_obj.id)
        self.assertEqual(zone_a_row["route_length"], 8.0)

    def test_average_delay_and_travel_time(self):

        rows = {row["zone_id"]: row for row in extract_zone_results(self.run)}
        zone_a_row = rows[self.zone_a.id]

        # depart_time average across both zone_a occupants: (1.0 + 0.0) / 2
        self.assertEqual(zone_a_row["average_delay"], 0.5)

        # travel time only over the ARRIVED occupant: 9.0
        self.assertEqual(zone_a_row["average_travel_time"], 9.0)


# =====================================================


class DeterministicOutputTests(unittest.TestCase):

    def test_repeated_extraction_is_identical(self):

        building = make_building()
        scenario = make_scenario(building)
        run = SimulationRun(
            scenario=scenario, building=building,
            movement_result=MultiAgentSimulationResult(occupants={}, total_evacuation_time=None),
        )

        first = extract_scenario_features(run)
        second = extract_scenario_features(run)

        self.assertEqual(first, second)

    def test_dataset_builder_build_all_is_stable_across_calls(self):

        building = make_building()
        scenario = make_scenario(building)
        run = SimulationRun(
            scenario=scenario, building=building,
            movement_result=MultiAgentSimulationResult(occupants={}, total_evacuation_time=None),
        )

        builder = DatasetBuilder([run])

        self.assertEqual(builder.build_all(), builder.build_all())


# =====================================================


class CsvExportTests(unittest.TestCase):

    def test_export_csv_writes_header_and_rows(self):

        rows = [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:

            path = str(Path(tmp_dir) / "out.csv")
            export_csv(rows, path)

            with open(path, newline="", encoding="utf-8") as csv_file:
                reader = list(csv.DictReader(csv_file))

            self.assertEqual(reader, [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}])

    def test_export_csv_fills_missing_columns_with_empty_string(self):

        rows = [{"a": 1}, {"a": 2, "b": 5}]

        with tempfile.TemporaryDirectory() as tmp_dir:

            path = str(Path(tmp_dir) / "out.csv")
            export_csv(rows, path)

            with open(path, newline="", encoding="utf-8") as csv_file:
                reader = list(csv.DictReader(csv_file))

            self.assertEqual(reader[0]["b"], "")
            self.assertEqual(reader[1]["b"], "5")

    def test_dataset_builder_export_all_writes_three_csv_files(self):

        building = make_building()
        scenario = make_scenario(building)
        run = SimulationRun(
            scenario=scenario, building=building,
            movement_result=MultiAgentSimulationResult(occupants={}, total_evacuation_time=None),
        )

        builder = DatasetBuilder([run])

        with tempfile.TemporaryDirectory() as tmp_dir:

            paths = builder.export_all(tmp_dir)

            for path in paths.values():
                self.assertTrue(Path(path).exists())

            with open(paths["scenario_features"], newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["scenario_id"], "scn-1")


if __name__ == "__main__":
    unittest.main()
