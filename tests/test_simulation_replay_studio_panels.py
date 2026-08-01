import sys
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from navigation.edge import Edge

from command_center.building_view import BuildingView
from command_center.dashboard import Dashboard
from command_center.event_timeline_panel import EventTimelinePanel
from command_center.incident_data import IncidentData
from command_center.occupant_inspector_panel import OccupantInspectorPanel
from command_center.statistics_panel import StatisticsPanel

from simulation_recording.occupant_routes import OccupantRouteHop, OccupantRouteRecord

from tests.test_command_center import (
    make_building,
    make_decision_policy,
    make_ground_truth,
    make_scenario,
    make_timeline_rows,
)


def make_occupant_routes():

    return (
        OccupantRouteRecord(
            occupant_id="occ-1", state="ARRIVED", depart_time=0.0, arrival_time=8.0,
            hops=(
                OccupantRouteHop(
                    from_node_id="zone-a", to_node_id="zone-b", edge_id="door-1", edge_type=Edge.DOOR,
                    start_time=1.0, end_time=8.0, distance=5.0, queue_wait_time=0.0,
                ),
            ),
        ),
        OccupantRouteRecord(occupant_id="occ-2", state="STATIONARY", depart_time=0.0, arrival_time=None),
        OccupantRouteRecord(occupant_id="occ-3", state="UNREACHABLE", depart_time=0.0, arrival_time=None),
    )


def make_decision_events():

    return (
        {
            "event_type": "Help_Decision", "occupant_id": "occ-1", "related_occupant_id": "occ-3",
            "reason": "clear_to_assist", "metadata": {"assistance_type": "ESCORT"},
        },
    )


def make_full_incident_data():

    building = make_building()
    scenario = make_scenario(building)
    ground_truth = make_ground_truth()
    decision_policy = make_decision_policy(building, scenario, ground_truth)
    timeline_rows = make_timeline_rows(building)

    return IncidentData(
        building=building, scenario=scenario, ground_truth=ground_truth,
        decision_policy=decision_policy, timeline_rows=timeline_rows,
        occupant_routes=make_occupant_routes(), decision_events=make_decision_events(),
    )


class BuildingViewOccupantRenderingTests(unittest.TestCase):

    def test_set_occupant_routes_and_select_occupant_do_not_crash(self):

        incident = make_full_incident_data()
        view = BuildingView()

        view.set_building(incident.building)
        view.set_occupant_routes(incident.occupant_routes)
        view.show_frame(incident.frame_at(5.0))

        view.select_occupant("occ-1")

        self.assertEqual(view.selected_occupant_id, "occ-1")

    # =====================================================

    def test_no_occupant_routes_renders_without_crash(self):

        incident = make_full_incident_data()
        view = BuildingView()

        view.set_building(incident.building)
        view.show_frame(incident.frame_at(0.0))

        # No exception -- an empty occupant_routes/occupant_positions is
        # exactly as safe to render as never having called either setter.


class OccupantInspectorPanelTests(unittest.TestCase):

    def setUp(self):

        self.incident = make_full_incident_data()
        self.panel = OccupantInspectorPanel()

    # =====================================================

    def test_set_incident_populates_occupant_combo(self):

        self.panel.set_incident(self.incident)

        items = [self.panel.occupant_combo.itemText(i) for i in range(self.panel.occupant_combo.count())]
        self.assertEqual(items, ["occ-1", "occ-2", "occ-3"])

    # =====================================================

    def test_select_occupant_shows_profile_and_base_strategies(self):

        self.panel.set_incident(self.incident)
        self.panel.select_occupant("occ-1")
        self.panel.show_frame(self.incident.frame_at(5.0))

        self.assertEqual(self.panel.profile_value.text(), "Adult_Default")
        self.assertNotEqual(self.panel.decision_strategy_value.text(), "-")
        self.assertNotEqual(self.panel.route_choice_strategy_value.text(), "-")

    # =====================================================

    def test_assistance_from_decision_events_is_reflected(self):

        self.panel.set_incident(self.incident)

        self.panel.select_occupant("occ-1")
        self.assertTrue(self.panel.assisting_value.text().startswith("Yes"))

        self.panel.select_occupant("occ-3")
        self.assertTrue(self.panel.assisted_value.text().startswith("Yes"))

    # =====================================================

    def test_none_incident_clears_fields(self):

        self.panel.set_incident(self.incident)
        self.panel.set_incident(None)

        self.assertEqual(self.panel.profile_value.text(), "-")


class EventTimelinePanelTests(unittest.TestCase):

    def test_merges_and_sorts_every_source(self):

        incident = make_full_incident_data()
        panel = EventTimelinePanel()

        panel.set_incident(incident)

        categories = {panel.table.item(row, 1).text() for row in range(panel.table.rowCount())}

        # Occupant depart/arrive (occ-1 has hops), Engineering (door-1/
        # exit-1 change state at t=20 in make_timeline_rows()), and
        # Decision (the Help_Decision event above) must all appear.
        self.assertIn("Occupant", categories)
        self.assertIn("Engineering", categories)
        self.assertIn("Decision", categories)

        times = []
        for row in range(panel.table.rowCount()):
            text = panel.table.item(row, 0).text()
            if text != "Pre-departure":
                times.append(float(text.rstrip("s")))

        self.assertEqual(times, sorted(times))

    # =====================================================

    def test_double_click_emits_jump_to_time(self):

        incident = make_full_incident_data()
        panel = EventTimelinePanel()
        panel.set_incident(incident)

        received = []
        panel.jump_to_time.connect(received.append)

        panel._on_row_double_clicked(0, 0)

        self.assertEqual(len(received), 1)


class StatisticsPanelTests(unittest.TestCase):

    def test_set_incident_builds_profile_distribution(self):

        incident = make_full_incident_data()
        panel = StatisticsPanel()

        panel.set_incident(incident)

        self.assertEqual(panel.profile_table.rowCount(), 1)
        self.assertEqual(panel.profile_table.item(0, 0).text(), "Adult_Default")
        self.assertEqual(panel.profile_table.item(0, 1).text(), "3")

    # =====================================================

    def test_none_incident_clears_charts_without_crash(self):

        panel = StatisticsPanel()
        panel.set_incident(None)

        self.assertEqual(panel.profile_table.rowCount(), 0)


class DashboardIntegrationTests(unittest.TestCase):

    def test_set_incident_wires_every_new_panel_without_crash(self):

        incident = make_full_incident_data()
        dashboard = Dashboard()

        dashboard.set_incident(incident)

        self.assertEqual(dashboard.building_view.selected_occupant_id, None)

        dashboard.building_view.occupant_clicked.emit("occ-1")

        self.assertEqual(dashboard.occupant_inspector_panel._selected_occupant_id, "occ-1")

    # =====================================================

    def test_jump_to_time_moves_the_slider(self):

        incident = make_full_incident_data()
        dashboard = Dashboard()
        dashboard.set_incident(incident)

        dashboard.jump_to_time(20.0)

        self.assertEqual(dashboard.frame_index, dashboard._incident.frame_count - 1)


if __name__ == "__main__":
    unittest.main()
