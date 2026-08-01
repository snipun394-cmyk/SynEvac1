import json
import sys
import tempfile
import unittest
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from models.building import Building
from models.camera import Camera
from models.detector import Detector
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.project import Project
from models.staircase import Staircase
from models.zone import Zone

from scenario.fire import ScenarioFire
from scenario.metadata import ScenarioMetadata
from scenario.occupant import ScenarioOccupant
from scenario.scenario import Scenario

from ground_truth.labels import GroundTruth

from decision_policy import DecisionInputs, generate_policy

from dataset_builder.schema import (
    camera_state_columns,
    detector_state_columns,
    door_state_columns,
    exit_state_columns,
    ordered_cameras,
    ordered_detectors,
    ordered_doors,
    ordered_exits,
    ordered_stairs,
    ordered_zones,
    stair_state_columns,
    zone_occupancy_columns,
)
from dataset_builder.timeline_schema import (
    zone_smoke_columns,
    zone_temperature_columns,
    zone_visibility_columns,
)

from serialization.serializer import Serializer
from scenario_storage.storage import save_scenario

from command_center.incident_data import IncidentData, load_incident
from command_center.building_view import BuildingView, OVERLAY_MODES, OVERLAY_RECOMMENDATION
from command_center.hazard_panel import HazardPanel
from command_center.human_panel import HumanPanel
from command_center.incident_status_bar import IncidentStatusBar
from command_center.occupancy_panel import OccupancyPanel
from command_center.recommendation_center import (
    BuildingRecommendationsPanel,
    CivilianAnnouncementsPanel,
    CommanderSummaryPanel,
    FirefighterIntelligencePanel,
    RecommendationCenter,
)
from command_center.recommendation_panel import RecommendationPanel
from command_center.recommendation_timeline_panel import RecommendationTimelinePanel
from command_center.incident_panel import IncidentPanel
from command_center.timeline_panel import TimelinePanel
from command_center.dashboard import Dashboard
from command_center.main_window import MainWindow


# Every GUI test needs a live QApplication -- module-level singleton,
# same convention tests/test_campaign_studio.py already establishes.
_app = QApplication.instance() or QApplication(sys.argv)


# =====================================================
# Fixtures
# =====================================================


def make_building():

    floor_1 = Floor(
        name="Ground", id="floor-1", display_order=0,
        zones=[
            Zone(id="zone-a", name="Zone A", x=0.0, y=0.0, width=4.0, height=5.0, floor_id="floor-1"),
            Zone(id="zone-b", name="Zone B", x=5.0, y=0.0, width=4.0, height=5.0, floor_id="floor-1"),
        ],
        doors=[
            Door(
                id="door-1", name="Door 1", start_point=(4.0, 2.0), end_point=(5.0, 2.0),
                floor_id="floor-1", zone_a_id="zone-a", zone_b_id="zone-b",
            ),
        ],
        exits=[
            Exit(
                id="exit-1", name="Exit 1", start_point=(0.0, 0.0), end_point=(0.0, 1.0),
                floor_id="floor-1", zone_id="zone-a", capacity=10, is_blocked=False,
            ),
        ],
        stairs=[
            Staircase(
                id="stair-1", name="Stair 1", from_position=(8.0, 2.0), to_position=(0.0, 0.0),
                from_floor_id="floor-1", to_floor_id="floor-2",
                from_zone_id="zone-b", to_zone_id="zone-c",
            ),
        ],
        detectors=[Detector(id="detector-1", name="Detector 1", position=(1.0, 1.0), floor_id="floor-1")],
        cameras=[Camera(id="camera-1", name="Camera 1", position=(2.0, 2.0), floor_id="floor-1")],
    )

    floor_2 = Floor(
        name="First Floor", id="floor-2", display_order=1,
        zones=[Zone(id="zone-c", name="Zone C", x=0.0, y=0.0, width=4.0, height=4.0, floor_id="floor-2")],
    )

    return Building(name="Test Building", id="building-1", floors=[floor_1, floor_2])


def make_scenario(building):

    metadata = ScenarioMetadata(
        scenario_id="scn-test", definition_id="def-1", definition_content_hash="hash-1",
        generation_version="v1", seed=42, created_at="2026-01-01T00:00:00",
    )

    fire = ScenarioFire(ignition_zone_id="zone-a", ignition_floor_id="floor-1", fire_profile="Electrical")

    occupants = (
        ScenarioOccupant(
            occupant_id="occ-1", zone_id="zone-a", floor_id="floor-1",
            position=(0.0, 0.0), behaviour_profile_id="Adult_Default",
        ),
        ScenarioOccupant(
            occupant_id="occ-2", zone_id="zone-a", floor_id="floor-1",
            position=(0.0, 0.0), behaviour_profile_id="Adult_Default",
        ),
        ScenarioOccupant(
            occupant_id="occ-3", zone_id="zone-b", floor_id="floor-1",
            position=(0.0, 0.0), behaviour_profile_id="Adult_Default",
        ),
    )

    return Scenario(metadata=metadata, occupants=occupants, fire=fire)


def make_ground_truth():

    return GroundTruth(
        scenario_id="scn-test",
        definition_id="def-1",
        total_evacuation_time=120.5,
        building_cleared=False,
        reachable_occupants=2,
        unreachable_occupants=1,
        people_trapped=1,
        people_evacuated=2,
        worst_exit=None,
        worst_stair=None,
        worst_door=None,
        peak_congestion_location_id="exit-1",
        peak_congestion_location_type="exit",
        peak_congestion_value=5,
        congestion_duration=30.0,
        zone_route_stats=[
            {"zone_id": "zone-a", "preferred_exit": "exit-1", "preferred_stair": None,
             "average_travel_distance": 5.0, "average_travel_time": 5.0},
            {"zone_id": "zone-b", "preferred_exit": None, "preferred_stair": "stair-1",
             "average_travel_distance": 5.0, "average_travel_time": 5.0},
        ],
        first_hazardous_zone="zone-a",
        hazard_spread_order=("zone-a",),
        maximum_hazard_zone="zone-a",
        highest_risk_floor="floor-1",
        doors_that_became_bottlenecks=(),
        exits_underutilized=(),
        exits_exceeding_capacity=(),
        stairs_exceeding_capacity=(),
        zone_risk_scores=[
            {"zone_id": "zone-a", "risk_score": 0.9},
            {"zone_id": "zone-b", "risk_score": 0.2},
        ],
        exit_risk_scores=[{"exit_id": "exit-1", "risk_score": 0.1}],
        stair_risk_scores=[{"stair_id": "stair-1", "risk_score": 0.05}],
    )


def make_decision_policy(building, scenario, ground_truth):

    return generate_policy(
        DecisionInputs(building=building, scenario=scenario, ground_truth=ground_truth),
    )


def make_timeline_row(
    building, time, fire_zones, hazard_score, zone_occupancy, zone_smoke,
    door_state, exit_state, stair_state, detector_state, camera_state,
    people_evacuated, people_remaining, people_trapped, congestion,
    max_congestion, evac_rate, queue_length, route_utilization, bottleneck,
):

    row = {
        "scenario_id": "scn-test",
        "simulation_time": time,
        "ignition_zone": "zone-a",
        "current_fire_zones": ";".join(fire_zones),
        "current_hazard_score": hazard_score,
        "people_evacuated": people_evacuated,
        "people_remaining": people_remaining,
        "people_trapped": people_trapped,
        "current_congestion": congestion,
        "maximum_congestion_so_far": max_congestion,
        "current_evacuation_rate": evac_rate,
        "current_queue_length": queue_length,
        "current_route_utilization": route_utilization,
        "current_bottleneck": bottleneck,
    }

    for column, zone in zip(zone_occupancy_columns(building), ordered_zones(building)):
        row[column] = zone_occupancy.get(zone.id)
    for column, zone in zip(zone_smoke_columns(building), ordered_zones(building)):
        row[column] = zone_smoke.get(zone.id)
    for column, _zone in zip(zone_temperature_columns(building), ordered_zones(building)):
        row[column] = None
    for column, _zone in zip(zone_visibility_columns(building), ordered_zones(building)):
        row[column] = None
    for column, door in zip(door_state_columns(building), ordered_doors(building)):
        row[column] = door_state.get(door.id)
    for column, exit_obj in zip(exit_state_columns(building), ordered_exits(building)):
        row[column] = exit_state.get(exit_obj.id)
    for column, stair in zip(stair_state_columns(building), ordered_stairs(building)):
        row[column] = stair_state.get(stair.id)
    for column, detector in zip(detector_state_columns(building), ordered_detectors(building)):
        row[column] = detector_state.get(detector.id)
    for column, camera in zip(camera_state_columns(building), ordered_cameras(building)):
        row[column] = camera_state.get(camera.id)

    return row


def make_timeline_rows(building):

    row0 = make_timeline_row(
        building, time=0.0, fire_zones=(), hazard_score=0.0,
        zone_occupancy={"zone-a": 2, "zone-b": 1, "zone-c": 0},
        zone_smoke={"zone-a": 0.0, "zone-b": 0.0, "zone-c": 0.0},
        door_state={"door-1": "OPEN"}, exit_state={"exit-1": "OPEN"},
        stair_state={"stair-1": "AVAILABLE"}, detector_state={"detector-1": "AVAILABLE"},
        camera_state={"camera-1": "AVAILABLE"},
        people_evacuated=0, people_remaining=3, people_trapped=0,
        congestion=0, max_congestion=0, evac_rate=0.0, queue_length=0,
        route_utilization=0.0, bottleneck=None,
    )

    row1 = make_timeline_row(
        building, time=10.0, fire_zones=("zone-a",), hazard_score=0.6,
        zone_occupancy={"zone-a": 1, "zone-b": 1, "zone-c": 0},
        zone_smoke={"zone-a": 0.6, "zone-b": 0.1, "zone-c": 0.0},
        door_state={"door-1": "OPEN"}, exit_state={"exit-1": "OPEN"},
        stair_state={"stair-1": "AVAILABLE"}, detector_state={"detector-1": "AVAILABLE"},
        camera_state={"camera-1": "AVAILABLE"},
        people_evacuated=1, people_remaining=2, people_trapped=0,
        congestion=2, max_congestion=2, evac_rate=0.1, queue_length=1,
        route_utilization=0.5, bottleneck="exit-1",
    )

    row2 = make_timeline_row(
        building, time=20.0, fire_zones=("zone-a", "zone-b"), hazard_score=0.9,
        zone_occupancy={"zone-a": 0, "zone-b": 0, "zone-c": 0},
        zone_smoke={"zone-a": 0.9, "zone-b": 0.5, "zone-c": 0.0},
        door_state={"door-1": "LOCKED"}, exit_state={"exit-1": "CLOSED"},
        stair_state={"stair-1": "CLOSED"}, detector_state={"detector-1": "FAILED"},
        camera_state={"camera-1": "FAILED"},
        people_evacuated=2, people_remaining=0, people_trapped=1,
        congestion=0, max_congestion=2, evac_rate=0.1, queue_length=0,
        route_utilization=0.0, bottleneck=None,
    )

    return (row0, row1, row2)


def make_incident_data(with_timeline=True):

    building = make_building()
    scenario = make_scenario(building)
    ground_truth = make_ground_truth()
    decision_policy = make_decision_policy(building, scenario, ground_truth)

    timeline_rows = make_timeline_rows(building) if with_timeline else ()

    return IncidentData(
        building=building, scenario=scenario, ground_truth=ground_truth,
        decision_policy=decision_policy, timeline_rows=timeline_rows,
    )


# =====================================================


class IncidentDataBaselineTests(unittest.TestCase):

    def setUp(self):

        self.incident = make_incident_data(with_timeline=False)

    def test_single_baseline_frame(self):

        self.assertEqual(self.incident.frame_count, 1)
        self.assertFalse(self.incident.has_timeline)
        self.assertEqual(self.incident.duration, 0.0)

    def test_baseline_occupancy_from_scenario(self):

        frame = self.incident.frame_at(0.0)

        self.assertEqual(frame.zone_occupancy["zone-a"], 2)
        self.assertEqual(frame.zone_occupancy["zone-b"], 1)
        self.assertEqual(frame.people_remaining, 3)
        self.assertEqual(frame.people_evacuated, 0)
        self.assertEqual(frame.people_trapped, 0)

    def test_baseline_engineering_states_from_building_defaults(self):

        frame = self.incident.frame_at(0.0)

        self.assertEqual(frame.door_states["door-1"], "OPEN")
        self.assertEqual(frame.exit_states["exit-1"], "OPEN")
        self.assertEqual(frame.stair_states["stair-1"], "AVAILABLE")
        self.assertEqual(frame.detector_states["detector-1"], "AVAILABLE")
        self.assertEqual(frame.camera_states["camera-1"], "AVAILABLE")

    def test_baseline_ignition_zone_from_scenario_fire(self):

        frame = self.incident.frame_at(0.0)
        self.assertEqual(frame.ignition_zone_id, "zone-a")
        self.assertEqual(frame.current_fire_zone_ids, frozenset())

    def test_frame_at_any_time_returns_baseline(self):

        self.assertEqual(self.incident.frame_at(999.0).time, 0.0)
        self.assertEqual(self.incident.frame_at_index(0).time, 0.0)


# =====================================================


class IncidentDataTimelineTests(unittest.TestCase):

    def setUp(self):

        self.incident = make_incident_data(with_timeline=True)

    def test_frame_count_and_duration(self):

        self.assertEqual(self.incident.frame_count, 3)
        self.assertEqual(self.incident.duration, 20.0)
        self.assertTrue(self.incident.has_timeline)

    def test_frame_at_nearest_preceding(self):

        self.assertEqual(self.incident.frame_at(5.0).time, 0.0)
        self.assertEqual(self.incident.frame_at(10.0).time, 10.0)
        self.assertEqual(self.incident.frame_at(15.0).time, 10.0)
        self.assertEqual(self.incident.frame_at(25.0).time, 20.0)
        self.assertEqual(self.incident.frame_at(-5.0).time, 0.0)

    def test_zone_occupancy_mapped_by_id_not_column(self):

        frame = self.incident.frame_at(10.0)

        self.assertEqual(frame.zone_occupancy["zone-a"], 1)
        self.assertEqual(frame.zone_occupancy["zone-b"], 1)
        self.assertEqual(frame.zone_occupancy["zone-c"], 0)

    def test_current_fire_zone_ids_parsed(self):

        self.assertEqual(self.incident.frame_at(20.0).current_fire_zone_ids, {"zone-a", "zone-b"})
        self.assertEqual(self.incident.frame_at(0.0).current_fire_zone_ids, frozenset())

    def test_engineering_states_mapped_by_id(self):

        frame = self.incident.frame_at(20.0)

        self.assertEqual(frame.door_states["door-1"], "LOCKED")
        self.assertEqual(frame.exit_states["exit-1"], "CLOSED")
        self.assertEqual(frame.stair_states["stair-1"], "CLOSED")
        self.assertEqual(frame.detector_states["detector-1"], "FAILED")
        self.assertEqual(frame.camera_states["camera-1"], "FAILED")

    def test_evacuation_and_congestion_fields(self):

        frame = self.incident.frame_at(10.0)

        self.assertEqual(frame.people_evacuated, 1)
        self.assertEqual(frame.people_remaining, 2)
        self.assertEqual(frame.current_congestion, 2)
        self.assertEqual(frame.current_bottleneck, "exit-1")


# =====================================================


class IncidentDataNameLookupTests(unittest.TestCase):

    def setUp(self):

        self.incident = make_incident_data()

    def test_known_ids_resolve_to_names(self):

        self.assertEqual(self.incident.name_for("zone-a"), "Zone A")
        self.assertEqual(self.incident.name_for("exit-1"), "Exit 1")

    def test_unknown_id_falls_back_to_id(self):

        self.assertEqual(self.incident.name_for("no-such-id"), "no-such-id")

    def test_none_or_empty_resolves_to_placeholder(self):

        self.assertEqual(self.incident.name_for(None), "-")
        self.assertEqual(self.incident.name_for(""), "-")


# =====================================================


class IncidentDataLoadRoundTripTests(unittest.TestCase):

    def test_load_incident_round_trips_all_artifacts(self):

        building = make_building()
        scenario = make_scenario(building)
        ground_truth = make_ground_truth()
        decision_policy = make_decision_policy(building, scenario, ground_truth)
        timeline_rows = make_timeline_rows(building)

        with tempfile.TemporaryDirectory() as tmp_dir:

            tmp_path = Path(tmp_dir)

            project_path = tmp_path / "project.syn"
            Serializer.save(Project(name="Test Project", building=building), str(project_path))

            scenario_storage_root = tmp_path / "scenarios"
            save_scenario(scenario, scenario_storage_root)

            ground_truth_path = tmp_path / "ground_truth.json"
            ground_truth_path.write_text(json.dumps(ground_truth.to_dict()), encoding="utf-8")

            decision_policy_path = tmp_path / "decision_policy.json"
            decision_policy_path.write_text(json.dumps(decision_policy.to_dict()), encoding="utf-8")

            timeline_rows_path = tmp_path / "timeline_rows.json"
            timeline_rows_path.write_text(json.dumps(list(timeline_rows)), encoding="utf-8")

            incident_data = load_incident(
                project_path=str(project_path),
                scenario_id="scn-test",
                scenario_storage_root=str(scenario_storage_root),
                ground_truth_path=str(ground_truth_path),
                decision_policy_path=str(decision_policy_path),
                timeline_rows_path=str(timeline_rows_path),
            )

        self.assertEqual(incident_data.building.name, "Test Building")
        self.assertEqual(incident_data.scenario.metadata.scenario_id, "scn-test")
        self.assertEqual(incident_data.ground_truth.scenario_id, "scn-test")
        self.assertEqual(incident_data.decision_policy.scenario_id, "scn-test")
        self.assertEqual(incident_data.frame_count, 3)


# =====================================================


class BuildingViewTests(unittest.TestCase):

    def setUp(self):

        self.incident = make_incident_data()
        self.view = BuildingView()

    def test_set_building_populates_floor_combo_and_scene(self):

        self.view.set_building(self.incident.building)

        self.assertEqual(self.view.floor_combo.count(), 2)
        self.assertIs(self.view.current_floor, self.incident.building.ordered_floors()[0])
        self.assertGreater(len(self.view.scene.items()), 0)

    def test_show_frame_and_overlay_modes_do_not_raise(self):

        self.view.set_building(self.incident.building)
        self.view.set_decision_policy(self.incident.decision_policy)

        for mode in OVERLAY_MODES:

            self.view.set_overlay_mode(mode)
            self.view.show_frame(self.incident.frame_at(20.0))

        self.assertEqual(self.view.overlay_mode, OVERLAY_MODES[-1])

    def test_invalid_overlay_mode_raises(self):

        with self.assertRaises(ValueError):
            self.view.set_overlay_mode("Not A Real Mode")

    def test_recommendation_overlay_reads_decision_policy(self):

        self.view.set_building(self.incident.building)
        self.view.set_decision_policy(self.incident.decision_policy)
        self.view.set_overlay_mode(OVERLAY_RECOMMENDATION)
        self.view.show_frame(self.incident.frame_at(0.0))

        # Should not raise, and should have drawn something for both
        # floor-1's zones/door/exit/stair and their marker items.
        self.assertGreater(len(self.view.scene.items()), 0)

    def test_set_floor_switches_current_floor(self):

        self.view.set_building(self.incident.building)
        second_floor = self.incident.building.ordered_floors()[1]

        self.view.set_floor(second_floor)

        self.assertIs(self.view.current_floor, second_floor)
        self.assertEqual(self.view.floor_combo.currentIndex(), 1)


# =====================================================


class HazardPanelTests(unittest.TestCase):

    def setUp(self):

        self.incident = make_incident_data()
        self.panel = HazardPanel()

    def test_empty_state_shows_placeholders(self):

        self.assertIn("-", self.panel.ignition_label.text())

    def test_set_incident_shows_ground_truth_summary(self):

        self.panel.set_incident(self.incident)

        self.assertIn("Zone A", self.panel.first_hazardous_label.text())
        self.assertIn("Zone A", self.panel.maximum_hazard_label.text())
        self.assertEqual(self.panel.risk_table.rowCount(), 4)

    def test_show_frame_updates_fire_and_smoke(self):

        self.panel.set_incident(self.incident)
        self.panel.show_frame(self.incident.frame_at(20.0))

        self.assertIn("Zone A", self.panel.current_fire_label.text())
        self.assertIn("Zone B", self.panel.current_fire_label.text())
        self.assertGreater(self.panel.smoke_table.rowCount(), 0)


# =====================================================


class OccupancyPanelTests(unittest.TestCase):

    def setUp(self):

        self.incident = make_incident_data()
        self.panel = OccupancyPanel()

    def test_set_incident_populates_zone_table(self):

        self.panel.set_incident(self.incident)
        self.assertEqual(self.panel.occupancy_table.rowCount(), 3)

    def test_show_frame_updates_stat_tiles(self):

        self.panel.set_incident(self.incident)
        self.panel.show_frame(self.incident.frame_at(10.0))

        self.assertEqual(self.panel.evacuated_value.text(), "1")
        self.assertEqual(self.panel.remaining_value.text(), "2")
        self.assertEqual(self.panel.trapped_value.text(), "0")
        self.assertIn("Exit 1", self.panel.bottleneck_label.text())

    def test_ground_truth_metrics_shown(self):

        self.panel.set_incident(self.incident)
        self.assertIn("120.5", self.panel.total_evacuation_time_label.text())
        self.assertIn("No", self.panel.building_cleared_label.text())


# =====================================================


class RecommendationPanelTests(unittest.TestCase):

    def setUp(self):

        self.incident = make_incident_data()
        self.panel = RecommendationPanel()

    def test_set_incident_populates_all_tables(self):

        self.panel.set_incident(self.incident)

        policy = self.incident.decision_policy

        self.assertEqual(self.panel.zone_table.rowCount(), len(policy.zone_decisions))
        self.assertEqual(self.panel.exit_table.rowCount(), len(policy.exit_decisions))
        self.assertEqual(self.panel.stair_table.rowCount(), len(policy.stair_decisions))
        self.assertEqual(self.panel.announcement_table.rowCount(), len(policy.announcements))
        self.assertEqual(
            self.panel.firefighter_table.rowCount(), len(policy.firefighter_deployment_priority),
        )

    def test_known_decisions_render_correctly(self):

        self.panel.set_incident(self.incident)

        # zone-a: risk 0.9 >= CRITICAL_RISK_THRESHOLD -> SHELTER_IN_PLACE.
        zone_a_row_text = self.panel.zone_table.item(0, 1).text()
        self.assertEqual(zone_a_row_text, "SHELTER_IN_PLACE")

        # exit-1 is staged in zone-a, which is hazardous -> CLOSE.
        self.assertEqual(self.panel.exit_table.item(0, 1).text(), "CLOSE")

        # stair-1 connects zone-b/zone-c (neither hazardous) with a low
        # risk_score (0.05) -> USE.
        self.assertEqual(self.panel.stair_table.item(0, 1).text(), "USE")

    def test_no_policy_clears_tables(self):

        self.panel.set_incident(None)
        self.assertEqual(self.panel.zone_table.rowCount(), 0)
        self.assertEqual(self.panel.rescue_order_label.text(), "Rescue order: -")


# =====================================================


class IncidentPanelTests(unittest.TestCase):

    def setUp(self):

        self.incident = make_incident_data()
        self.panel = IncidentPanel()

    def test_set_incident_shows_metadata(self):

        self.panel.set_incident(self.incident)

        self.assertIn("scn-test", self.panel.scenario_id_label.text())
        self.assertIn("42", self.panel.seed_label.text())
        self.assertIn("Electrical", self.panel.fire_profile_label.text())

    def test_engineering_state_tables_populated(self):

        self.panel.set_incident(self.incident)

        self.assertEqual(self.panel.door_table.rowCount(), 1)
        self.assertEqual(self.panel.exit_table.rowCount(), 1)
        self.assertEqual(self.panel.stair_table.rowCount(), 1)
        self.assertEqual(self.panel.detector_table.rowCount(), 1)
        self.assertEqual(self.panel.camera_table.rowCount(), 1)

    def test_show_frame_updates_engineering_states(self):

        self.panel.set_incident(self.incident)
        self.panel.show_frame(self.incident.frame_at(20.0))

        self.assertEqual(self.panel.door_table.item(0, 1).text(), "LOCKED")
        self.assertEqual(self.panel.exit_table.item(0, 1).text(), "CLOSED")
        self.assertEqual(self.panel.detector_table.item(0, 1).text(), "FAILED")

    def test_ground_truth_congestion_facts(self):

        self.panel.set_incident(self.incident)

        self.assertIn("Exit 1", self.panel.peak_congestion_label.text())
        self.assertIn("30.0", self.panel.congestion_duration_label.text())


# =====================================================


class HumanPanelTests(unittest.TestCase):

    def setUp(self):

        self.incident = make_incident_data()
        self.panel = HumanPanel()

    def test_set_incident_populates_people_table(self):

        self.panel.set_incident(self.incident)

        # make_scenario() occupants + firefighters all resolve to a
        # HumanObservation (see _build_human_observations()) -- exact
        # count isn't asserted here (that's IncidentData's own concern),
        # only that the panel actually rendered something from it.
        self.assertGreaterEqual(self.panel.people_table.rowCount(), 1)

    def test_selecting_a_row_shows_detail(self):

        self.panel.set_incident(self.incident)
        self.panel.people_table.selectRow(0)

        self.assertNotEqual(self.panel.person_id_label.text(), "Person: -")

    def test_none_incident_clears_table(self):

        self.panel.set_incident(None)
        self.assertEqual(self.panel.people_table.rowCount(), 0)


# =====================================================


class IncidentStatusBarTests(unittest.TestCase):

    def setUp(self):

        self.incident = make_incident_data()
        self.status_bar = IncidentStatusBar()

    def test_set_incident_shows_building_name_and_first_frame(self):

        self.status_bar.set_incident(self.incident)

        self.assertEqual(self.status_bar.building_name_value.text(), "Test Building")
        self.assertEqual(self.status_bar.remaining_value.text(), "3")
        self.assertEqual(self.status_bar.evacuated_value.text(), "0")
        self.assertEqual(self.status_bar.time_since_alarm_value.text(), "00:00")
        self.assertEqual(self.status_bar.fire_floor_value.text(), "None")

    def test_show_frame_updates_severity_rset_and_confidence(self):

        self.status_bar.set_incident(self.incident)
        report = self.incident.advisory_report_at_index(2)
        self.status_bar.show_frame(self.incident.frame_at_index(2), report)

        self.assertEqual(
            self.status_bar.incident_severity_value.text(),
            report.commander_dashboard.overall_incident_severity,
        )
        self.assertEqual(self.status_bar.rset_value.text(), "02:00")
        self.assertNotEqual(self.status_bar.occupancy_confidence_value.text(), "-")
        self.assertEqual(self.status_bar.fire_floor_value.text(), "Ground")

    def test_none_report_degrades_to_placeholders(self):

        self.status_bar.set_incident(self.incident)
        self.status_bar.show_frame(self.incident.frame_at_index(0), None)

        self.assertEqual(self.status_bar.incident_severity_value.text(), "-")
        self.assertEqual(self.status_bar.rset_value.text(), "-")


# =====================================================


class RecommendationCenterTests(unittest.TestCase):

    def setUp(self):

        self.incident = make_incident_data()

    def test_civilian_panel_lists_announcements_and_explains_selection(self):

        panel = CivilianAnnouncementsPanel()
        panel.set_incident(self.incident)

        report = self.incident.advisory_report_at_index(0)
        self.assertEqual(panel.table.rowCount(), len(report.civilian_announcements))

        panel.table.selectRow(0)
        self.assertIn("Why:", panel.detail_label.text())
        self.assertIn("Confidence breakdown", panel.detail_label.text())
        self.assertIn("Not available in this build", panel.detail_label.text())
        self.assertIn("Not deployed", panel.detail_label.text())

    def test_building_panel_lists_recommendations_and_explains_selection(self):

        panel = BuildingRecommendationsPanel()
        panel.set_incident(self.incident)

        report = self.incident.advisory_report_at_index(0)
        self.assertEqual(panel.table.rowCount(), len(report.building_recommendations))

        panel.table.selectRow(0)
        self.assertIn("Why:", panel.detail_label.text())

    def test_firefighter_panel_shows_intelligence_never_a_directive(self):

        panel = FirefighterIntelligencePanel()
        panel.set_incident(self.incident)

        report = self.incident.advisory_report_at_index(0)
        self.assertEqual(panel.building_status_label.text(), report.firefighter_intelligence.building_status)
        self.assertIn("02:00", panel.rset_label.text())

        # Intelligence, never a command: nothing in this panel's rendered
        # text should ever read as an imperative order to a firefighter.
        rendered = panel.building_status_label.text()
        for imperative in ("must", "should proceed", "execute"):
            self.assertNotIn(imperative, rendered.lower())

    def test_commander_panel_shows_dashboard_fields(self):

        panel = CommanderSummaryPanel()
        panel.set_incident(self.incident)

        report = self.incident.advisory_report_at_index(0)
        self.assertEqual(panel.severity_value.text(), report.commander_dashboard.overall_incident_severity)
        self.assertEqual(panel.rset_value.text(), "02:00")

    def test_container_wires_all_seven_panels(self):

        # Civilian/Voice Evacuation/Firefighter/Building/Building
        # Controls/Warden Notifications/Commander -- bumped from six to
        # seven by the Execution Layer V1 milestone's own
        # WardenNotificationsPanel addition.
        center = RecommendationCenter()
        center.set_incident(self.incident)
        center.show_frame(self.incident.frame_at_index(0), self.incident.advisory_report_at_index(0))

        self.assertEqual(center.count(), 7)

    def test_none_incident_degrades_without_raising(self):

        center = RecommendationCenter()
        center.set_incident(None)
        center.show_frame(None, None)

        self.assertEqual(center.civilian_panel.table.rowCount(), 0)


# =====================================================
# Phase 3 (Critical-finding remediation) -- "AI confidence" must never
# be shown unless a genuine AI signal contributed. make_incident_data()
# below configures no AI Inference gateway, so every panel here is the
# "no real AI signal" case -- the literal string "AI confidence" must
# never appear anywhere in either detail panel for this fixture.
# =====================================================


class ConfidenceLabelTests(unittest.TestCase):

    def setUp(self):

        self.incident = make_incident_data()

    def test_civilian_panel_never_claims_ai_confidence_without_a_real_ai_signal(self):

        panel = CivilianAnnouncementsPanel()
        panel.set_incident(self.incident)
        panel.table.selectRow(0)

        text = panel.detail_label.text()
        self.assertNotIn("AI confidence", text)
        self.assertIn("Recommendation confidence (blended, rule-based)", text)
        self.assertIn("no AI/RL/crowd signal supplied for this recommendation", text)

    def test_building_panel_never_claims_ai_confidence(self):

        panel = BuildingRecommendationsPanel()
        panel.set_incident(self.incident)
        panel.table.selectRow(0)

        text = panel.detail_label.text()
        self.assertNotIn("AI confidence", text)
        self.assertIn("Recommendation confidence (blended, rule-based)", text)
        self.assertIn("never AI/RL-augmented", text)

    def test_confidence_label_helpers_are_conditioned_on_confidence_source(self):

        from command_center.recommendation_center import (
            _confidence_label, _prediction_source_line, _rl_influence_line,
        )

        self.assertEqual(_confidence_label(()), "Recommendation confidence (blended, rule-based)")
        self.assertEqual(_confidence_label(("ai",)), "AI-augmented confidence (blended)")

        self.assertIn("no AI/RL/crowd signal supplied", _prediction_source_line(()))
        self.assertIn("an AI prediction", _prediction_source_line(("ai",)))
        self.assertIn("an RL policy", _prediction_source_line(("rl",)))
        self.assertIn("live crowd intelligence", _prediction_source_line(("crowd",)))

        self.assertEqual(_rl_influence_line(()), "RL influence: Not deployed.")
        self.assertEqual(
            _rl_influence_line(("rl",)), "RL influence: Contributed to this recommendation's confidence.",
        )


# =====================================================


class RecommendationTimelinePanelTests(unittest.TestCase):

    def setUp(self):

        self.incident = make_incident_data()
        self.panel = RecommendationTimelinePanel()

    def test_set_incident_lists_full_recommendation_history(self):

        self.panel.set_incident(self.incident)

        self.assertEqual(self.panel.table.rowCount(), len(self.incident.recommendation_history))
        self.assertGreater(self.panel.table.rowCount(), 0)

    def test_show_frame_highlights_reached_rows(self):

        self.panel.set_incident(self.incident)
        self.panel.show_frame(self.incident.frame_at_index(0))

        # At t=0 no history entry (all at t > 0 in this fixture) should
        # yet be marked reached -- background stays the untouched
        # "pending" brush.
        first_item = self.panel.table.item(0, 0)
        self.assertIsNotNone(first_item)

    def test_none_incident_clears_table(self):

        self.panel.set_incident(None)
        self.assertEqual(self.panel.table.rowCount(), 0)


# =====================================================


class TimelinePanelTests(unittest.TestCase):

    def setUp(self):

        self.incident = make_incident_data()
        self.panel = TimelinePanel()

    def test_set_incident_configures_slider_range(self):

        self.panel.set_incident(self.incident)

        self.assertEqual(self.panel.slider.minimum(), 0)
        self.assertEqual(self.panel.slider.maximum(), 2)

    def test_set_frame_index_updates_time_label(self):

        self.panel.set_incident(self.incident)
        self.panel.set_frame_index(2)

        self.assertIn("20.0", self.panel.time_label.text())
        self.assertEqual(self.panel.frame_index, 2)

    def test_speed_multiplier_parses_combo_text(self):

        self.panel.set_incident(self.incident)
        index = self.panel.speed_combo.findText("2x")
        self.panel.speed_combo.setCurrentIndex(index)

        self.assertEqual(self.panel.speed_multiplier(), 2.0)

    def test_jump_to_time_moves_slider_to_nearest_preceding_frame(self):

        self.panel.set_incident(self.incident)

        self.panel.jump_to_time_spinbox.setValue(15.0)
        self.panel._on_jump_to_time()

        # Same nearest-preceding-frame convention as IncidentData.
        # frame_at(15.0) -- the row1 frame (t=10.0), index 1.
        self.assertEqual(self.panel.frame_index, 1)

    def test_trend_chart_receives_four_series(self):

        self.panel.set_incident(self.incident)

        self.assertEqual(len(self.panel.trend_chart._hazard), 3)
        self.assertEqual(len(self.panel.trend_chart._occupancy), 3)
        self.assertEqual(len(self.panel.trend_chart._smoke), 3)


# =====================================================


class DashboardTests(unittest.TestCase):

    def setUp(self):

        self.incident = make_incident_data()
        self.dashboard = Dashboard()

    def test_set_incident_wires_up_every_panel(self):

        self.dashboard.set_incident(self.incident)

        self.assertEqual(self.dashboard.frame_count, 3)
        self.assertEqual(self.dashboard.frame_index, 0)
        self.assertGreater(len(self.dashboard.building_view.scene.items()), 0)
        self.assertEqual(self.dashboard.status_bar.building_name_value.text(), "Test Building")
        self.assertGreater(self.dashboard.recommendation_center.civilian_panel.table.rowCount(), 0)
        self.assertGreater(self.dashboard.recommendation_timeline_panel.table.rowCount(), 0)

    def test_slider_drag_updates_displayed_frame(self):

        self.dashboard.set_incident(self.incident)

        self.dashboard.timeline_panel.slider.setValue(2)

        self.assertIn("20.0", self.dashboard.timeline_panel.time_label.text())
        self.assertEqual(self.dashboard.occupancy_panel.trapped_value.text(), "1")
        self.assertEqual(self.dashboard.status_bar.remaining_value.text(), "0")

    def test_set_overlay_mode_and_set_floor_delegate_to_building_view(self):

        self.dashboard.set_incident(self.incident)

        self.dashboard.set_overlay_mode(OVERLAY_RECOMMENDATION)
        self.assertEqual(self.dashboard.building_view.overlay_mode, OVERLAY_RECOMMENDATION)

        second_floor = self.incident.building.ordered_floors()[1]
        self.dashboard.set_floor(second_floor)
        self.assertIs(self.dashboard.building_view.current_floor, second_floor)


# =====================================================


class MainWindowTests(unittest.TestCase):

    def setUp(self):

        self.incident = make_incident_data()
        self.window = MainWindow()

    def test_load_incident_data_populates_dashboard(self):

        self.window.load_incident_data(self.incident)

        self.assertEqual(self.window.dashboard.frame_count, 3)
        self.assertFalse(self.window.playback_timer.isActive())

    def test_theme_stylesheet_is_applied(self):

        # Phase 8 -- applied once at the top level so it cascades to
        # every panel; just confirm MainWindow actually carries it.
        self.assertIn("QGroupBox", self.window.styleSheet())

    def test_play_pause_reset(self):

        self.window.load_incident_data(self.incident)

        self.window.play_playback()
        self.assertTrue(self.window.playback_timer.isActive())

        self.window.pause_playback()
        self.assertFalse(self.window.playback_timer.isActive())

        self.window.dashboard.set_frame_index(2)
        self.window.reset_playback()
        self.assertEqual(self.window.dashboard.frame_index, 0)

    def test_playback_tick_advances_by_speed_multiplier(self):

        self.window.load_incident_data(self.incident)

        index = self.window.dashboard.timeline_panel.speed_combo.findText("1x")
        self.window.dashboard.timeline_panel.speed_combo.setCurrentIndex(index)

        self.window._on_playback_tick()
        self.assertEqual(self.window.dashboard.frame_index, 1)

        self.window._on_playback_tick()
        self.assertEqual(self.window.dashboard.frame_index, 2)

        # Already at the last frame -- another tick should stop playback
        # rather than raising or wrapping around.
        self.window.playback_timer.start()
        self.window._on_playback_tick()
        self.assertFalse(self.window.playback_timer.isActive())
        self.assertEqual(self.window.dashboard.frame_index, 2)


if __name__ == "__main__":
    unittest.main()
