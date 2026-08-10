import unittest

from models.building import Building
from models.door import Door
from models.zone import Zone

from navigation.graph_builder import NavigationGraphGenerator

from pathfinding.engine import PathfindingEngine

from simulator.coordinator import MultiAgentSimulation

from simulation_recording.occupant_position import BuildingPositionIndex, interpolate_occupant_position
from simulation_recording.occupant_routes import build_occupant_route_records


# Occupant Movement Path Bug Investigation -- confirms, with a real,
# minimal, executed scenario (two perpendicular corridors meeting at a
# right-angle corner node, no congestion/hazard/human-behavior
# randomness), that an occupant's route is stored per-edge ([A->B, B->C],
# never a flat start/goal pair), the simulator produces one
# OccupantTimelineStep PER EDGE with that edge's own timing, and the
# replay/rendering interpolation layer (simulation_recording.
# occupant_position.interpolate_occupant_position()) moves the occupant
# along EACH edge's own two endpoints in turn -- never in a straight line
# directly from the route's start to its final destination. No bug was
# found in this pipeline; this is a permanent regression guard against
# ever introducing one, not a fix for a defect that was located.


def _build_two_corridor_corner_building():

    # RoomA --(door_ab, horizontal)--> CornerB --(door_bc, vertical)--> RoomC
    # A genuine ~90-degree bend: A and B share the same y; B and C share
    # the same x. A straight line from A directly to C would move both
    # x and y simultaneously throughout -- the failure mode this test
    # exists to rule out.

    building = Building(name="B")
    floor = building.create_floor(name="Ground Floor")

    room_a = Zone(name="RoomA", x=0.0, y=0.0, width=4.0, height=4.0)
    corner_b = Zone(name="CornerB", x=10.0, y=0.0, width=4.0, height=4.0)
    room_c = Zone(name="RoomC", x=10.0, y=-14.0, width=4.0, height=4.0)

    floor.add_zone(room_a)
    floor.add_zone(corner_b)
    floor.add_zone(room_c)

    # Real start_point/end_point (never left at the Door dataclass's own
    # (0.0, 0.0) default) -- Door.center is now a load-bearing via-point
    # for movement interpolation (see the Occupant Movement Path Fix),
    # so a real project's own door geometry must be reproduced here, not
    # left unset the way it safely could be before that fix existed.
    # Positioned in the gap between each zone pair, aligned with each
    # zone's own shared axis (y=2 for the horizontal leg, x=12 for the
    # vertical leg) -- exactly reproducing a real drawn door, not an
    # arbitrary point.
    floor.add_door(Door(
        name="D_AB", zone_a_id=room_a.id, zone_b_id=corner_b.id, floor_id=floor.id, width=2.0,
        start_point=(4.0, 2.0), end_point=(10.0, 2.0),
    ))
    floor.add_door(Door(
        name="D_BC", zone_a_id=corner_b.id, zone_b_id=room_c.id, floor_id=floor.id, width=2.0,
        start_point=(12.0, 0.0), end_point=(12.0, -10.0),
    ))

    graph = NavigationGraphGenerator().build(building)
    engine = PathfindingEngine(graph)

    return building, engine, room_a, corner_b, room_c


class MultiEdgeRouteTimelineTests(unittest.TestCase):

    def setUp(self):

        self.building, self.engine, self.room_a, self.corner_b, self.room_c = _build_two_corridor_corner_building()

        sim = MultiAgentSimulation(self.engine)
        sim.add_occupant(self.room_a.id, self.room_c.id, occupant_id="solo", walking_speed=1.2)

        self.result = sim.run()
        self.timeline = self.result.occupants["solo"]

    def test_route_produces_one_timeline_step_per_edge_not_one_step_for_the_whole_route(self):

        self.assertEqual(len(self.timeline.steps), 2)

        step_a_to_b, step_b_to_c = self.timeline.steps

        self.assertEqual(step_a_to_b.from_node.id, self.room_a.id)
        self.assertEqual(step_a_to_b.to_node.id, self.corner_b.id)

        self.assertEqual(step_b_to_c.from_node.id, self.corner_b.id)
        self.assertEqual(step_b_to_c.to_node.id, self.room_c.id)

        # The second hop starts exactly where the first one ends -- no
        # gap, no overlap, no direct A->C shortcut.
        self.assertAlmostEqual(step_a_to_b.end_time, step_b_to_c.start_time, places=9)

    def test_interpolated_position_bends_at_the_corner_instead_of_cutting_diagonally(self):

        records = build_occupant_route_records(self.result)
        record = next(r for r in records if r.occupant_id == "solo")
        index = BuildingPositionIndex(self.building)

        first_hop_midpoint = (record.hops[0].start_time + record.hops[0].end_time) / 2.0
        second_hop_midpoint = (record.hops[1].start_time + record.hops[1].end_time) / 2.0

        position_during_first_hop = interpolate_occupant_position(
            record, self.room_a.id, first_hop_midpoint, index,
        )
        position_during_second_hop = interpolate_occupant_position(
            record, self.room_a.id, second_hop_midpoint, index,
        )

        # Mid-way through A->B: still on the A-B horizontal, y unchanged
        # from A's own y, x strictly between A and B (never reaching C's
        # y at all -- a diagonal shortcut would already show meaningful
        # y movement here).
        self.assertAlmostEqual(position_during_first_hop.y, self.room_a.center[1], places=6)
        self.assertGreater(position_during_first_hop.x, self.room_a.center[0])
        self.assertLess(position_during_first_hop.x, self.corner_b.center[0])

        # Mid-way through B->C: already AT the corner's own x (fully
        # arrived horizontally), only y is still moving toward C. A
        # diagonal A->C shortcut would show x still short of B's x here.
        self.assertAlmostEqual(position_during_second_hop.x, self.corner_b.center[0], places=6)
        self.assertLess(position_during_second_hop.y, self.corner_b.center[1])
        self.assertGreater(position_during_second_hop.y, self.room_c.center[1])

    def test_final_position_is_room_c_not_a_straight_line_endpoint_reached_early(self):

        records = build_occupant_route_records(self.result)
        record = next(r for r in records if r.occupant_id == "solo")
        index = BuildingPositionIndex(self.building)

        final_position = interpolate_occupant_position(
            record, self.room_a.id, record.hops[-1].end_time, index,
        )

        self.assertAlmostEqual(final_position.x, self.room_c.center[0], places=6)
        self.assertAlmostEqual(final_position.y, self.room_c.center[1], places=6)
        self.assertEqual(final_position.state, "ARRIVED")


if __name__ == "__main__":
    unittest.main()
