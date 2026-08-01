import json
import os
import shutil
import tempfile
import unittest

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from navigation.edge import Edge
from navigation.node import Node
from pathfinding.route import Route

from simulator.multi_agent_result import MultiAgentSimulationResult, OccupantTimeline, OccupantTimelineStep
from simulator.occupant import OccupantState

from simulation_recording.decision_events import load_decision_events, save_decision_events
from simulation_recording.occupant_position import (
    BuildingPositionIndex,
    interpolate_occupant_position,
)
from simulation_recording.occupant_routes import (
    OccupantRouteHop,
    OccupantRouteRecord,
    build_occupant_route_records,
    load_occupant_routes,
    save_occupant_routes,
)


class _TempDir:

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="simulation_recording_test_")
        return self.path

    def __exit__(self, *args):
        shutil.rmtree(self.path, ignore_errors=True)


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


# =====================================================
# build_occupant_route_records() -- pure extraction from an already-
# built MultiAgentSimulationResult, never a live coordinator.
# =====================================================


class BuildOccupantRouteRecordsTests(unittest.TestCase):

    def test_extracts_plain_ids_never_live_objects(self):

        from_node = Node(id="zone-a", name="Zone A", floor_id="floor-1", node_type=Node.ZONE)
        to_node = Node(id="zone-b", name="Zone B", floor_id="floor-1", node_type=Node.ZONE)
        edge = Edge(id="door-1", edge_type=Edge.DOOR, from_node="zone-a", to_node="zone-b", walking_distance=5.0)

        route = Route(nodes=[from_node, to_node], edges=[edge], total_cost=5.0, total_distance=5.0)

        step = OccupantTimelineStep(
            index=0, from_node=from_node, to_node=to_node, edge=edge,
            queue_wait_time=1.5, start_time=2.0, end_time=12.0,
        )

        timeline = OccupantTimeline(
            occupant_id="occ-1", route=route, steps=[step],
            state=OccupantState.ARRIVED, depart_time=2.0, arrival_time=12.0,
        )

        result = MultiAgentSimulationResult(occupants={"occ-1": timeline}, total_evacuation_time=12.0)

        records = build_occupant_route_records(result)

        self.assertEqual(len(records), 1)
        record = records[0]

        self.assertEqual(record.occupant_id, "occ-1")
        self.assertEqual(record.state, "ARRIVED")
        self.assertEqual(record.depart_time, 2.0)
        self.assertEqual(record.arrival_time, 12.0)
        self.assertEqual(len(record.hops), 1)

        hop = record.hops[0]
        self.assertEqual(hop.from_node_id, "zone-a")
        self.assertEqual(hop.to_node_id, "zone-b")
        self.assertEqual(hop.edge_id, "door-1")
        self.assertEqual(hop.edge_type, Edge.DOOR)
        self.assertEqual(hop.start_time, 2.0)
        self.assertEqual(hop.end_time, 12.0)
        self.assertEqual(hop.distance, 5.0)
        self.assertEqual(hop.queue_wait_time, 1.5)

        # Every field is a plain str/float -- never a Node/Edge instance.
        self.assertIsInstance(hop.from_node_id, str)
        self.assertIsInstance(hop.edge_type, str)

    # =====================================================

    def test_stationary_occupant_has_no_hops(self):

        timeline = OccupantTimeline(
            occupant_id="occ-2", route=None, steps=[],
            state=OccupantState.STATIONARY, depart_time=0.0, arrival_time=None,
        )
        result = MultiAgentSimulationResult(occupants={"occ-2": timeline}, total_evacuation_time=None)

        records = build_occupant_route_records(result)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].state, "STATIONARY")
        self.assertEqual(records[0].hops, ())
        self.assertIsNone(records[0].arrival_time)


# =====================================================
# Round-trip persistence -- JsonWriter/JsonReader, the same convention
# scenario_storage already uses.
# =====================================================


class RoundTripTests(unittest.TestCase):

    def test_occupant_routes_round_trip_exactly(self):

        records = (
            OccupantRouteRecord(
                occupant_id="occ-1", state="ARRIVED", depart_time=2.0, arrival_time=12.0,
                hops=(
                    OccupantRouteHop(
                        from_node_id="zone-a", to_node_id="zone-b", edge_id="door-1", edge_type=Edge.DOOR,
                        start_time=2.0, end_time=12.0, distance=5.0, queue_wait_time=1.5,
                    ),
                ),
            ),
            OccupantRouteRecord(occupant_id="occ-2", state="UNREACHABLE", depart_time=0.0, arrival_time=None),
        )

        with _TempDir() as directory:

            path = os.path.join(directory, "occupant_routes.json")
            save_occupant_routes(records, path)

            self.assertTrue(os.path.isfile(path))

            with open(path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)

            self.assertIsInstance(raw, list)
            self.assertEqual(raw[0]["occupant_id"], "occ-1")

            loaded = load_occupant_routes(path)

        self.assertEqual(loaded, records)

    # =====================================================

    def test_decision_events_round_trip_exactly(self):

        events = (
            {"event_type": "Help_Decision", "occupant_id": "occ-1", "related_occupant_id": "occ-2",
             "reason": "clear_to_assist", "metadata": {"assistance_type": "PUSH_WHEELCHAIR"}},
        )

        with _TempDir() as directory:

            path = os.path.join(directory, "decision_events.json")
            save_decision_events(events, path)
            loaded = load_decision_events(path)

        self.assertEqual(loaded, events)


# =====================================================
# interpolate_occupant_position() -- see command_center.building_view/
# occupant_inspector_panel for the consumers of this exact contract.
# =====================================================


class InterpolateOccupantPositionTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()
        self.index = BuildingPositionIndex(self.building)

        self.record = OccupantRouteRecord(
            occupant_id="occ-1", state="ARRIVED", depart_time=2.0, arrival_time=13.0,
            hops=(
                OccupantRouteHop(
                    from_node_id="zone-a", to_node_id="zone-b", edge_id="door-1", edge_type=Edge.DOOR,
                    start_time=3.0, end_time=13.0, distance=5.0, queue_wait_time=0.0,
                ),
            ),
        )

    # =====================================================

    def test_before_depart_time_holds_at_start_zone(self):

        position = interpolate_occupant_position(self.record, "zone-a", 0.0, self.index)

        self.assertEqual((position.x, position.y), (2.0, 2.5))
        self.assertEqual(position.state, "PENDING")
        self.assertEqual(position.speed, 0.0)

    # =====================================================

    def test_after_depart_before_first_hop_is_at_node(self):

        position = interpolate_occupant_position(self.record, "zone-a", 2.5, self.index)

        self.assertEqual((position.x, position.y), (2.0, 2.5))
        self.assertEqual(position.state, "AT_NODE")

    # =====================================================

    def test_mid_hop_interpolates_linearly(self):

        position = interpolate_occupant_position(self.record, "zone-a", 8.0, self.index)

        # zone-a center (2, 2.5) -> zone-b center (7, 2.5), t = 0.5.
        self.assertAlmostEqual(position.x, 4.5)
        self.assertAlmostEqual(position.y, 2.5)
        self.assertEqual(position.state, "TRAVERSING")
        self.assertAlmostEqual(position.speed, 0.5)

    # =====================================================

    def test_after_arrival_holds_at_destination(self):

        position = interpolate_occupant_position(self.record, "zone-a", 20.0, self.index)

        self.assertEqual((position.x, position.y), (7.0, 2.5))
        self.assertEqual(position.zone_id, "zone-b")
        self.assertEqual(position.state, "ARRIVED")

    # =====================================================

    def test_stair_hop_holds_position_across_floors(self):

        stair_record = OccupantRouteRecord(
            occupant_id="occ-3", state="ARRIVED", depart_time=0.0, arrival_time=10.0,
            hops=(
                OccupantRouteHop(
                    from_node_id="zone-a", to_node_id="zone-b", edge_id="stair-1", edge_type=Edge.STAIR,
                    start_time=0.0, end_time=10.0, distance=5.0, queue_wait_time=0.0,
                ),
            ),
        )

        position = interpolate_occupant_position(stair_record, "zone-a", 5.0, self.index)

        # Holds at the FROM zone's own position -- never interpolated
        # across a Stair hop's two, non-comparable floor coordinate
        # spaces.
        self.assertEqual((position.x, position.y), (2.0, 2.5))
        self.assertEqual(position.current_stair_id, "stair-1")
        self.assertEqual(position.state, "TRAVERSING")

    # =====================================================

    def test_stationary_record_has_no_hops_and_holds_at_start_zone(self):

        stationary_record = OccupantRouteRecord(
            occupant_id="occ-4", state="STATIONARY", depart_time=0.0, arrival_time=None,
        )

        position = interpolate_occupant_position(stationary_record, "zone-b", 50.0, self.index)

        self.assertEqual((position.x, position.y), (7.0, 2.5))
        self.assertEqual(position.state, "STATIONARY")

    # =====================================================

    def test_unresolvable_start_zone_never_fabricates_a_point(self):

        position = interpolate_occupant_position(self.record, None, 0.0, self.index)

        self.assertIsNone(position.x)
        self.assertIsNone(position.y)


if __name__ == "__main__":
    unittest.main()
