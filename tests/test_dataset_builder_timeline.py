import csv
import tempfile
import unittest
from pathlib import Path

from ai_decision.recommendation import DecisionRecommendation

from hazard.node_state import HazardNodeState
from hazard.snapshot import HazardSnapshot

from models.building import Building
from models.camera import Camera
from models.detector import Detector
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.staircase import Staircase
from models.zone import Zone

from navigation.edge import Edge
from navigation.node import Node

from occupancy.observation import OccupancyObservation
from occupancy.snapshot import OccupancySnapshot

from pathfinding.route import Route

from scenario.event import ScenarioEvent
from scenario.fire import ScenarioFire
from scenario.metadata import ScenarioMetadata
from scenario.occupant import ScenarioOccupant
from scenario.scenario import Scenario

from simulation_runtime.result import TickResult

from simulator.multi_agent_result import (
    MultiAgentSimulationResult,
    OccupantTimeline,
    OccupantTimelineStep,
)
from simulator.occupant import OccupantState

from dataset_builder.schema import (
    camera_state_columns,
    detector_state_columns,
    door_state_columns,
)
from dataset_builder.timeline import TimelineRun, extract_timeline_rows
from dataset_builder.timeline_exporter import TimelineDatasetBuilder, export_timeline_csv
from dataset_builder.timeline_schema import (
    zone_smoke_columns,
    zone_temperature_columns,
    zone_visibility_columns,
)


# =====================================================
# Fixtures
# =====================================================


def make_building(zone_count=2, door_count=1, exit_count=1, stair_count=0,
                   detector_count=1, camera_count=1, exit_capacity=2):

    zones = [
        Zone(id=f"zone-{i}", name=f"Zone {i}", x=0.0, y=0.0, width=4.0, height=5.0)
        for i in range(zone_count)
    ]

    doors = [
        Door(id=f"door-{i}", floor_id="floor-1", active=True, locked=False)
        for i in range(door_count)
    ]

    exits = [
        Exit(id=f"exit-{i}", floor_id="floor-1",
             zone_id=zones[-1].id if zones else "", is_blocked=False, capacity=exit_capacity)
        for i in range(exit_count)
    ]

    stairs = [
        Staircase(id=f"stair-{i}", from_floor_id="floor-1", to_floor_id="floor-1")
        for i in range(stair_count)
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
        detectors=detectors, cameras=cameras,
    )

    return Building(name="Test Building", id="building-1", floors=[floor])


def make_scenario(building, occupants):

    metadata = ScenarioMetadata(
        scenario_id="scn-1", definition_id="def-1", definition_content_hash="hash-1",
        generation_version="v1", seed=1, created_at="2026-01-01T00:00:00",
    )

    fire = None
    if building.floors[0].zones:
        fire = ScenarioFire(
            ignition_zone_id=building.floors[0].zones[0].id,
            ignition_floor_id="floor-1", fire_profile="Electrical",
            growth_parameters={"growth_time": 200.0},
        )

    return Scenario(metadata=metadata, occupants=occupants, fire=fire)


def make_zone_node(zone):

    return Node(id=zone.id, name=zone.name, floor_id=zone.floor_id, node_type=Node.ZONE)


def make_outside_node():

    return Node(id=Node.OUTSIDE_NODE_ID, name="Outside", floor_id="", node_type=Node.OUTSIDE)


def make_exit_timeline(occupant_id, from_zone, exit_obj, start_time, end_time,
                       queue_wait_time=0.0):

    from_node = make_zone_node(from_zone)
    to_node = make_outside_node()

    edge = Edge(id=exit_obj.id, edge_type=Edge.EXIT, from_node=from_node.id,
                to_node=to_node.id, walking_distance=8.0, reference=exit_obj)

    route = Route(nodes=[from_node, to_node], edges=[edge], total_cost=8.0, total_distance=8.0)

    step = OccupantTimelineStep(
        index=0, from_node=from_node, to_node=to_node, edge=edge,
        queue_wait_time=queue_wait_time, start_time=start_time, end_time=end_time,
    )

    return OccupantTimeline(
        occupant_id=occupant_id, route=route, steps=[step],
        state=OccupantState.ARRIVED, depart_time=start_time - queue_wait_time,
        arrival_time=end_time,
    )


def make_unreachable_timeline(occupant_id):

    return OccupantTimeline(
        occupant_id=occupant_id, route=None, steps=[],
        state=OccupantState.UNREACHABLE, depart_time=0.0, arrival_time=None,
    )


def make_tick(time, node_states=None, observations=None, fired_events=()):

    return TickResult(
        time=time,
        fired_events=tuple(fired_events),
        hazard_snapshot=HazardSnapshot(node_states=node_states or {}),
        occupancy_snapshot=OccupancySnapshot(observations=observations or {}),
        decision=DecisionRecommendation(),
    )


# =====================================================


class TimestepOrderingTests(unittest.TestCase):

    def test_rows_are_in_the_same_order_as_tick_results(self):

        building = make_building()
        scenario = make_scenario(building, occupants=())
        tick_results = (make_tick(5.0), make_tick(10.0), make_tick(15.0))

        run = TimelineRun(
            scenario=scenario, building=building,
            movement_result=MultiAgentSimulationResult(occupants={}, total_evacuation_time=None),
            tick_results=tick_results,
        )

        rows = extract_timeline_rows(run)

        self.assertEqual([row["simulation_time"] for row in rows], [5.0, 10.0, 15.0])

    def test_one_row_per_tick(self):

        building = make_building()
        scenario = make_scenario(building, occupants=())
        tick_results = tuple(make_tick(float(t)) for t in range(7))

        run = TimelineRun(
            scenario=scenario, building=building,
            movement_result=MultiAgentSimulationResult(occupants={}, total_evacuation_time=None),
            tick_results=tick_results,
        )

        self.assertEqual(len(extract_timeline_rows(run)), 7)


# =====================================================


class VariableSimulationLengthTests(unittest.TestCase):

    def test_zero_ticks_produces_zero_rows(self):

        building = make_building()
        scenario = make_scenario(building, occupants=())

        run = TimelineRun(
            scenario=scenario, building=building,
            movement_result=MultiAgentSimulationResult(occupants={}, total_evacuation_time=None),
            tick_results=(),
        )

        self.assertEqual(extract_timeline_rows(run), [])

    def test_many_ticks_produces_matching_row_count(self):

        building = make_building()
        scenario = make_scenario(building, occupants=())
        tick_results = tuple(make_tick(float(t)) for t in range(50))

        run = TimelineRun(
            scenario=scenario, building=building,
            movement_result=MultiAgentSimulationResult(occupants={}, total_evacuation_time=None),
            tick_results=tick_results,
        )

        self.assertEqual(len(extract_timeline_rows(run)), 50)


# =====================================================


class VariableObjectCountTests(unittest.TestCase):

    def test_zone_hazard_columns_match_zone_count(self):

        building = make_building(zone_count=4)

        self.assertEqual(len(zone_smoke_columns(building)), 4)
        self.assertEqual(len(zone_temperature_columns(building)), 4)
        self.assertEqual(len(zone_visibility_columns(building)), 4)

    def test_detector_and_camera_columns_match_their_counts(self):

        building = make_building(detector_count=3, camera_count=0)

        self.assertEqual(len(detector_state_columns(building)), 3)
        self.assertEqual(len(camera_state_columns(building)), 0)

    def test_row_has_no_detector_or_camera_columns_when_building_has_none(self):

        building = make_building(zone_count=1, door_count=0, exit_count=0,
                                  detector_count=0, camera_count=0)
        scenario = make_scenario(building, occupants=())

        run = TimelineRun(
            scenario=scenario, building=building,
            movement_result=MultiAgentSimulationResult(occupants={}, total_evacuation_time=None),
            tick_results=(make_tick(1.0),),
        )

        row = extract_timeline_rows(run)[0]

        self.assertNotIn("Detector_1_State", row)
        self.assertNotIn("Camera_1_State", row)
        self.assertNotIn("Door_1_State", row)


# =====================================================


class HazardAndOccupancyColumnTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building(zone_count=2, door_count=1, exit_count=1,
                                       detector_count=1, camera_count=1)
        self.zone_a, self.zone_b = self.building.floors[0].zones

    def test_zone_hazard_readings_are_read_from_the_hazard_snapshot(self):

        scenario = make_scenario(self.building, occupants=())

        tick = make_tick(
            5.0,
            node_states={
                self.zone_a.id: HazardNodeState(
                    smoke_level=0.3, temperature=45.0, visibility=6.0, hazard_score=0.4,
                ),
            },
            observations={
                self.zone_a.id: OccupancyObservation(occupant_count=2),
                self.zone_b.id: OccupancyObservation(occupant_count=0),
            },
        )

        run = TimelineRun(
            scenario=scenario, building=self.building,
            movement_result=MultiAgentSimulationResult(occupants={}, total_evacuation_time=None),
            tick_results=(tick,),
        )

        row = extract_timeline_rows(run)[0]

        self.assertEqual(row["Zone_1_Smoke"], 0.3)
        self.assertEqual(row["Zone_1_Temperature"], 45.0)
        self.assertEqual(row["Zone_1_Visibility"], 6.0)
        self.assertEqual(row["Zone_1_Occupancy"], 2)
        self.assertEqual(row["Zone_2_Occupancy"], 0)

        # zone_b has no HazardNodeState entry -- HazardSnapshot's
        # default (hazard_score 0.0, smoke/temp/visibility None) is
        # used, never a fabricated reading.
        self.assertIsNone(row["Zone_2_Smoke"])

    def test_node_state_is_read_once_per_zone_per_tick_not_once_per_column(self):

        # Platform-refinement performance fix (docs/validation/
        # technical_report.md §7.3): extract_timeline_rows() used to
        # call hazard_snapshot.node_state(zone.id) separately for
        # current_hazard_score/current_fire_zones, Smoke, Temperature,
        # and Visibility -- four redundant lookups per zone per tick
        # for the exact same (hazard_snapshot, zone.id) pair. It must
        # now be read once per zone and shared.

        scenario = make_scenario(self.building, occupants=())

        tick = make_tick(
            5.0,
            node_states={
                self.zone_a.id: HazardNodeState(
                    smoke_level=0.3, temperature=45.0, visibility=6.0, hazard_score=0.4,
                ),
            },
        )

        call_count = {"n": 0}
        original_node_state = HazardSnapshot.node_state

        def counting_node_state(self_snapshot, node_id):
            call_count["n"] += 1
            return original_node_state(self_snapshot, node_id)

        HazardSnapshot.node_state = counting_node_state
        try:
            run = TimelineRun(
                scenario=scenario, building=self.building,
                movement_result=MultiAgentSimulationResult(occupants={}, total_evacuation_time=None),
                tick_results=(tick,),
            )
            extract_timeline_rows(run)
        finally:
            HazardSnapshot.node_state = original_node_state

        # 2 zones, 1 tick -- exactly one node_state() call per zone.
        self.assertEqual(call_count["n"], 2)

    def test_current_fire_zones_and_hazard_score(self):

        scenario = make_scenario(self.building, occupants=())

        tick = make_tick(
            5.0,
            node_states={
                self.zone_a.id: HazardNodeState(hazard_score=0.2),
                self.zone_b.id: HazardNodeState(hazard_score=0.0),
            },
        )

        run = TimelineRun(
            scenario=scenario, building=self.building,
            movement_result=MultiAgentSimulationResult(occupants={}, total_evacuation_time=None),
            tick_results=(tick,),
        )

        row = extract_timeline_rows(run)[0]

        self.assertEqual(row["current_fire_zones"], self.zone_a.id)
        self.assertEqual(row["current_hazard_score"], 0.2)
        self.assertEqual(row["ignition_zone"], self.zone_a.id)


# =====================================================


class EngineeringStateReplayTests(unittest.TestCase):

    def test_door_state_changes_after_its_event_fires_and_persists(self):

        building = make_building(zone_count=1, door_count=1, exit_count=0,
                                  detector_count=0, camera_count=0)
        scenario = make_scenario(building, occupants=())
        door = building.floors[0].doors[0]

        close_event = ScenarioEvent(
            event_id="evt-1", target_type="door", target_id=door.id,
            event_type="close", time=10.0, parameters={"state": "CLOSED"},
        )

        tick_results = (
            make_tick(5.0),
            make_tick(10.0, fired_events=(close_event,)),
            make_tick(15.0),
        )

        run = TimelineRun(
            scenario=scenario, building=building,
            movement_result=MultiAgentSimulationResult(occupants={}, total_evacuation_time=None),
            tick_results=tick_results,
        )

        rows = extract_timeline_rows(run)

        self.assertEqual(rows[0]["Door_1_State"], "OPEN")
        self.assertEqual(rows[1]["Door_1_State"], "CLOSED")
        self.assertEqual(rows[2]["Door_1_State"], "CLOSED")


# =====================================================


class SimulationMetricsTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building(zone_count=2, exit_count=1, exit_capacity=2)
        self.zone_a, self.zone_b = self.building.floors[0].zones
        self.exit_obj = self.building.floors[0].exits[0]

        self.scenario = make_scenario(
            self.building,
            occupants=(
                ScenarioOccupant(
                    occupant_id="occ-1", zone_id=self.zone_a.id, floor_id="floor-1",
                    position=(0.0, 0.0), behaviour_profile_id="Adult_Default",
                ),
                ScenarioOccupant(
                    occupant_id="occ-2", zone_id=self.zone_a.id, floor_id="floor-1",
                    position=(0.0, 0.0), behaviour_profile_id="Adult_Default",
                ),
                ScenarioOccupant(
                    occupant_id="occ-3", zone_id=self.zone_b.id, floor_id="floor-1",
                    position=(0.0, 0.0), behaviour_profile_id="Adult_Default",
                ),
            ),
        )

        timelines = {
            "occ-1": make_exit_timeline(
                "occ-1", self.zone_a, self.exit_obj, start_time=0.0, end_time=12.0,
            ),
            "occ-2": make_exit_timeline(
                "occ-2", self.zone_a, self.exit_obj, start_time=6.0, end_time=20.0,
                queue_wait_time=5.0,
            ),
            "occ-3": make_unreachable_timeline("occ-3"),
        }

        self.movement_result = MultiAgentSimulationResult(
            occupants=timelines, total_evacuation_time=20.0,
            unreachable_occupant_ids=["occ-3"],
        )

        self.tick_results = (
            make_tick(
                5.0,
                observations={
                    self.zone_a.id: OccupancyObservation(occupant_count=2),
                    self.zone_b.id: OccupancyObservation(occupant_count=1),
                },
            ),
            make_tick(
                10.0,
                observations={
                    self.zone_a.id: OccupancyObservation(occupant_count=1),
                    self.zone_b.id: OccupancyObservation(occupant_count=1),
                },
            ),
            make_tick(
                15.0,
                observations={
                    self.zone_a.id: OccupancyObservation(occupant_count=0),
                    self.zone_b.id: OccupancyObservation(occupant_count=1),
                },
            ),
        )

        self.run = TimelineRun(
            scenario=self.scenario, building=self.building,
            movement_result=self.movement_result, tick_results=self.tick_results,
        )

    def test_people_counts_change_only_once_someone_arrives(self):

        rows = extract_timeline_rows(self.run)

        self.assertEqual([row["people_evacuated"] for row in rows], [0, 0, 1])
        self.assertEqual([row["people_remaining"] for row in rows], [3, 3, 2])
        self.assertEqual([row["people_trapped"] for row in rows], [1, 1, 1])

    def test_current_evacuation_rate(self):

        rows = extract_timeline_rows(self.run)

        self.assertEqual(rows[0]["current_evacuation_rate"], 0.0)
        self.assertEqual(rows[1]["current_evacuation_rate"], 0.0)
        self.assertAlmostEqual(rows[2]["current_evacuation_rate"], 0.2)

    def test_current_queue_length_reflects_occupant_waiting_for_the_edge(self):

        rows = extract_timeline_rows(self.run)

        # occ-2 joins the queue at time 1.0 (start_time 6.0 - queue_wait_time 5.0)
        # and is admitted at 6.0 -- still queued at t=5, not queued by t=10.
        self.assertEqual(rows[0]["current_queue_length"], 1)
        self.assertEqual(rows[1]["current_queue_length"], 0)

    def test_current_congestion_and_running_maximum(self):

        rows = extract_timeline_rows(self.run)

        self.assertEqual(rows[0]["current_congestion"], 2)
        self.assertEqual(rows[1]["current_congestion"], 2)
        self.assertEqual(rows[2]["current_congestion"], 1)

        self.assertEqual(
            [row["maximum_congestion_so_far"] for row in rows], [2, 2, 2],
        )

    def test_current_route_utilization_and_bottleneck(self):

        rows = extract_timeline_rows(self.run)

        self.assertAlmostEqual(rows[0]["current_route_utilization"], 0.5)
        self.assertAlmostEqual(rows[1]["current_route_utilization"], 1.0)

        self.assertEqual(rows[0]["current_bottleneck"], self.exit_obj.id)
        self.assertEqual(rows[1]["current_bottleneck"], self.exit_obj.id)

    def test_hazard_frontier_is_always_left_blank(self):

        rows = extract_timeline_rows(self.run)

        self.assertTrue(all(row["current_hazard_frontier"] is None for row in rows))


# =====================================================


class DeterministicExportTests(unittest.TestCase):

    def _make_run(self):

        building = make_building()
        scenario = make_scenario(building, occupants=())
        tick_results = (make_tick(5.0), make_tick(10.0))

        return TimelineRun(
            scenario=scenario, building=building,
            movement_result=MultiAgentSimulationResult(occupants={}, total_evacuation_time=None),
            tick_results=tick_results,
        )

    def test_repeated_extraction_is_identical(self):

        run = self._make_run()

        self.assertEqual(extract_timeline_rows(run), extract_timeline_rows(run))

    def test_timeline_dataset_builder_build_is_stable(self):

        builder = TimelineDatasetBuilder([self._make_run()])

        self.assertEqual(builder.build(), builder.build())


# =====================================================


class CsvExportTests(unittest.TestCase):

    def test_export_timeline_csv_writes_one_row_per_timestep(self):

        building = make_building(zone_count=1, door_count=1, exit_count=0,
                                  detector_count=0, camera_count=0)
        scenario = make_scenario(building, occupants=())
        tick_results = (make_tick(5.0), make_tick(10.0), make_tick(15.0))

        run = TimelineRun(
            scenario=scenario, building=building,
            movement_result=MultiAgentSimulationResult(occupants={}, total_evacuation_time=None),
            tick_results=tick_results,
        )

        rows = extract_timeline_rows(run)

        with tempfile.TemporaryDirectory() as tmp_dir:

            path = str(Path(tmp_dir) / "timeline.csv")
            export_timeline_csv(rows, path)

            with open(path, newline="", encoding="utf-8") as csv_file:
                read_rows = list(csv.DictReader(csv_file))

            self.assertEqual(len(read_rows), 3)
            self.assertEqual(
                [row["simulation_time"] for row in read_rows], ["5.0", "10.0", "15.0"],
            )

            # current_hazard_frontier is always None -- must round-trip
            # through CSV as an empty string, never the text "None".
            self.assertEqual(read_rows[0]["current_hazard_frontier"], "")

    def test_timeline_dataset_builder_export_writes_timeline_csv(self):

        building = make_building()
        scenario = make_scenario(building, occupants=())
        run = TimelineRun(
            scenario=scenario, building=building,
            movement_result=MultiAgentSimulationResult(occupants={}, total_evacuation_time=None),
            tick_results=(make_tick(1.0),),
        )

        builder = TimelineDatasetBuilder([run])

        with tempfile.TemporaryDirectory() as tmp_dir:

            path = builder.export(tmp_dir)

            self.assertEqual(Path(path).name, "timeline.csv")
            self.assertTrue(Path(path).exists())


if __name__ == "__main__":
    unittest.main()
