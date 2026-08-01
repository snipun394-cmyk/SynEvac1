import sys
import unittest

from PyQt6.QtWidgets import QApplication

# command_center.incident_data pulls in PyQt6-adjacent machinery
# transitively via other command_center/advisory_system imports -- same
# module-level QApplication singleton convention
# tests/test_campaign_pipeline_integration.py already establishes.
_app = QApplication.instance() or QApplication(sys.argv)

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from scenario.occupant import ScenarioOccupant
from scenario.scenario import Scenario, ScenarioMetadata

from navigation.edge import Edge

from command_center.incident_data import IncidentData

from simulation_recording.occupant_routes import OccupantRouteHop, OccupantRouteRecord


def make_building():

    floor = Floor(
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
                id="exit-1", name="Exit 1", start_point=(5.0, 4.0), end_point=(5.0, 5.0),
                floor_id="floor-1", zone_id="zone-b", capacity=10, is_blocked=False,
            ),
        ],
    )

    return Building(name="Test Building", id="building-1", floors=[floor])


def make_scenario():

    metadata = ScenarioMetadata(
        scenario_id="scn-test", definition_id="def-1", definition_content_hash="hash-1",
        generation_version="v1", seed=7, created_at="2026-01-01T00:00:00",
    )

    occupants = (
        ScenarioOccupant(
            occupant_id="occ-1", zone_id="zone-a", floor_id="floor-1",
            position=(0.0, 0.0), behaviour_profile_id="Adult_Default",
        ),
    )

    return Scenario(metadata=metadata, occupants=occupants)


class IncidentDataOccupantPositionsTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()
        self.scenario = make_scenario()

        self.occupant_routes = (
            OccupantRouteRecord(
                occupant_id="occ-1", state="ARRIVED", depart_time=0.0, arrival_time=10.0,
                hops=(
                    OccupantRouteHop(
                        from_node_id="zone-a", to_node_id="zone-b", edge_id="door-1", edge_type=Edge.DOOR,
                        start_time=0.0, end_time=10.0, distance=5.0, queue_wait_time=0.0,
                    ),
                ),
            ),
        )

    # =====================================================

    def test_frame_at_interpolates_occupant_position(self):

        # A row per timeline tick is required here: with no timeline_rows
        # at all, IncidentData falls back to a single baseline frame at
        # time=0.0 (see _baseline_frame()'s own docstring) -- there would
        # be nothing at t=5.0 to resolve occupant_positions against.
        timeline_rows = [{"simulation_time": t} for t in (0.0, 5.0, 10.0, 15.0)]

        incident = IncidentData(
            building=self.building, scenario=self.scenario,
            occupant_routes=self.occupant_routes, timeline_rows=timeline_rows,
        )

        frame = incident.frame_at(5.0)
        self.assertEqual(frame.time, 5.0)

        position = frame.occupant_positions["occ-1"]

        self.assertAlmostEqual(position.x, 4.5)
        self.assertAlmostEqual(position.y, 2.5)
        self.assertEqual(position.state, "TRAVERSING")

    # =====================================================

    def test_no_occupant_routes_means_empty_positions_every_frame(self):

        incident = IncidentData(building=self.building, scenario=self.scenario)

        frame = incident.frame_at(0.0)

        self.assertEqual(dict(frame.occupant_positions), {})

    # =====================================================

    def test_decision_events_round_trip_onto_incident_data(self):

        events = ({"event_type": "Help_Decision", "occupant_id": "occ-1", "reason": "clear_to_assist"},)

        incident = IncidentData(building=self.building, scenario=self.scenario, decision_events=events)

        self.assertEqual(incident.decision_events, events)


if __name__ == "__main__":
    unittest.main()
