import math
import unittest

from models.building import Building
from models.door import Door
from models.zone import Zone
from models.staircase import Staircase

from navigation.edge import Edge
from navigation.graph_builder import NavigationGraphGenerator

from pathfinding.engine import PathfindingEngine

from simulator.coordinator import MultiAgentSimulation

from simulation_recording.occupant_position import BuildingPositionIndex, interpolate_occupant_position
from simulation_recording.occupant_routes import build_occupant_route_records

from sandbox.manager import SandboxManager
from sandbox.occupant import SandboxOccupantState


# Occupant Movement Path Fix -- Door-Aware Interpolation. Proves both
# rendering paths (Command Center Replay Studio's
# simulation_recording.occupant_position.interpolate_occupant_position()
# and Designer's live sandbox.manager.SandboxManager._sync_position())
# now interpolate through an edge's own physical connection point
# (Door/Exit.center) instead of directly between the two Zone centroids
# it connects. Geometry below mirrors p3.syn's own real Zone 5/Door 6/
# Zone 8 corridor bend (elongated horizontal strip -> elongated vertical
# strip, door off-center from both centroids) -- reproduced here as a
# self-contained fixture rather than loading the external .syn file, so
# this test runs anywhere and stays fast.


def _build_p3_shaped_building():

    # RoomStart --(door_1)--> ZoneH (horizontal strip) --(door_2, THE
    # BEND)--> ZoneV (vertical strip) --(door_3)--> RoomEnd
    #
    # door_2 is deliberately off-center from BOTH ZoneH's and ZoneV's
    # own centroids -- exactly the p3.syn characteristic that produced
    # the diagonal-cut bug: ZoneH center=(20,1), ZoneV center=(29,10),
    # door_2 center=(29,2) sits near ZoneV's own x but far from either
    # zone's y-center.

    building = Building(name="B")
    floor = building.create_floor(name="Ground Floor")

    room_start = Zone(name="RoomStart", x=0.0, y=-1.0, width=4.0, height=4.0)
    zone_h = Zone(name="ZoneH", x=10.0, y=0.0, width=20.0, height=2.0)
    zone_v = Zone(name="ZoneV", x=28.0, y=2.0, width=2.0, height=16.0)
    room_end = Zone(name="RoomEnd", x=27.0, y=20.0, width=4.0, height=4.0)

    floor.add_zone(room_start)
    floor.add_zone(zone_h)
    floor.add_zone(zone_v)
    floor.add_zone(room_end)

    door_1 = Door(
        name="D1", zone_a_id=room_start.id, zone_b_id=zone_h.id, floor_id=floor.id, width=2.0,
        start_point=(4.0, 1.0), end_point=(10.0, 1.0),
    )
    door_2 = Door(
        name="D2", zone_a_id=zone_h.id, zone_b_id=zone_v.id, floor_id=floor.id, width=2.0,
        start_point=(28.0, 2.0), end_point=(30.0, 2.0),
    )
    door_3 = Door(
        name="D3", zone_a_id=zone_v.id, zone_b_id=room_end.id, floor_id=floor.id, width=2.0,
        start_point=(28.0, 18.0), end_point=(30.0, 18.0),
    )
    floor.add_door(door_1)
    floor.add_door(door_2)
    floor.add_door(door_3)

    graph = NavigationGraphGenerator().build(building)
    engine = PathfindingEngine(graph)

    return building, engine, room_start, zone_h, zone_v, room_end, door_1, door_2, door_3


def _build_two_floor_stair_building():

    building = Building(name="B")
    ground = building.create_floor(name="Ground Floor")
    upper = building.create_floor(name="Floor 1", height=3.0)

    lobby = Zone(name="Lobby", x=0.0, y=0.0, width=4.0, height=4.0)
    upstairs = Zone(name="Upstairs", x=0.0, y=0.0, width=4.0, height=4.0, floor_id=upper.id)

    ground.add_zone(lobby)
    upper.add_zone(upstairs)

    stair = Staircase(
        name="S1", from_floor_id=ground.id, to_floor_id=upper.id,
        from_zone_id=lobby.id, to_zone_id=upstairs.id, width=2.0,
    )
    ground.add_stair(stair)

    graph = NavigationGraphGenerator().build(building)
    engine = PathfindingEngine(graph)

    return building, engine, lobby, upstairs


class ReplayStudioDoorAwareInterpolationTests(unittest.TestCase):

    def setUp(self):

        (
            self.building, self.engine, self.room_start, self.zone_h, self.zone_v, self.room_end,
            self.door_1, self.door_2, self.door_3,
        ) = _build_p3_shaped_building()

        sim = MultiAgentSimulation(self.engine)
        sim.add_occupant(self.room_start.id, self.room_end.id, occupant_id="solo", walking_speed=1.2)

        self.result = sim.run()
        self.records = build_occupant_route_records(self.result)
        self.record = next(r for r in self.records if r.occupant_id == "solo")
        self.index = BuildingPositionIndex(self.building)

    def test_offset_door_produces_a_bend_not_a_diagonal(self):

        # The ZoneH -> ZoneV hop is the second hop (RoomStart->ZoneH,
        # ZoneH->ZoneV, ZoneV->RoomEnd).
        bend_hop = self.record.hops[1]
        self.assertEqual(bend_hop.edge_id, self.door_2.id)

        duration = bend_hop.end_time - bend_hop.start_time

        early = interpolate_occupant_position(self.record, self.room_start.id, bend_hop.start_time + 0.1 * duration, self.index)
        late = interpolate_occupant_position(self.record, self.room_start.id, bend_hop.start_time + 0.9 * duration, self.index)

        # Early in the hop: still close to ZoneH's own y (1.0) -- not
        # already deep into ZoneV's own y range, the way a direct
        # ZoneH-center -> ZoneV-center diagonal would already show.
        self.assertLess(early.y, 3.0)

        # Late in the hop: x has already arrived at ZoneV's own x (29.0,
        # matching door_2/zone_v's shared x) -- not still short of it,
        # the way a direct diagonal would show at 90% of the way through.
        self.assertAlmostEqual(late.x, self.zone_v.center[0], places=1)

    def test_split_point_equals_door_center_exactly(self):

        bend_hop = self.record.hops[1]

        from_position = self.zone_h.center
        via_position = self.door_2.center
        to_position = self.zone_v.center

        leg1 = math.dist(from_position, via_position)
        leg2 = math.dist(via_position, to_position)
        split = leg1 / (leg1 + leg2)

        duration = bend_hop.end_time - bend_hop.start_time
        split_time = bend_hop.start_time + split * duration

        position = interpolate_occupant_position(self.record, self.room_start.id, split_time, self.index)

        self.assertAlmostEqual(position.x, via_position[0], places=6)
        self.assertAlmostEqual(position.y, via_position[1], places=6)

    def test_multi_edge_route_uses_each_edges_own_door_in_order(self):

        self.assertEqual(len(self.record.hops), 3)

        expected_doors = (self.door_1, self.door_2, self.door_3)

        for hop, door in zip(self.record.hops, expected_doors):

            self.assertEqual(hop.edge_id, door.id)

            midpoint = hop.start_time + 0.5 * (hop.end_time - hop.start_time)
            position = interpolate_occupant_position(self.record, self.room_start.id, midpoint, self.index)

            # The midpoint-by-time position is never farther from THIS
            # hop's own door than the hop's own from/to endpoints are --
            # a weak but edge-order-sensitive sanity check that a later
            # hop's sample isn't accidentally reusing an earlier hop's
            # via-point (which would place it far outside this hop's own
            # geometry entirely).
            from_pos, _ = self.index.node_position(hop.from_node_id)
            to_pos, _ = self.index.node_position(hop.to_node_id, arriving_edge_id=hop.edge_id)
            span = math.dist(from_pos, to_pos) + 1.0

            self.assertLessEqual(math.dist((position.x, position.y), door.center), span)

    def test_final_position_and_arrival_time_are_the_real_destination(self):

        last_hop = self.record.hops[-1]

        final_position = interpolate_occupant_position(self.record, self.room_start.id, last_hop.end_time, self.index)

        self.assertAlmostEqual(final_position.x, self.room_end.center[0], places=6)
        self.assertAlmostEqual(final_position.y, self.room_end.center[1], places=6)
        self.assertEqual(final_position.state, "ARRIVED")
        self.assertAlmostEqual(last_hop.end_time, self.record.arrival_time, places=6)


class StairHopUnchangedTests(unittest.TestCase):

    def test_stair_hop_still_holds_position_at_entrance(self):

        building, engine, lobby, upstairs = _build_two_floor_stair_building()

        sim = MultiAgentSimulation(engine)
        sim.add_occupant(lobby.id, upstairs.id, occupant_id="solo", walking_speed=1.2, stair_speed=0.71)

        result = sim.run()
        records = build_occupant_route_records(result)
        record = next(r for r in records if r.occupant_id == "solo")
        index = BuildingPositionIndex(building)

        self.assertEqual(len(record.hops), 1)
        hop = record.hops[0]
        self.assertEqual(hop.edge_type, Edge.STAIR)

        midpoint = hop.start_time + 0.5 * (hop.end_time - hop.start_time)
        position = interpolate_occupant_position(record, lobby.id, midpoint, index)

        # Unchanged stair behavior: holds at the entrance (lobby's own
        # center), never interpolated through a door/via-point.
        self.assertAlmostEqual(position.x, lobby.center[0], places=6)
        self.assertAlmostEqual(position.y, lobby.center[1], places=6)
        self.assertEqual(position.current_stair_id, hop.edge_id)


class SandboxTrajectoryParityTests(unittest.TestCase):

    def test_sandbox_produces_the_same_bend_as_replay_studio(self):

        (
            building, engine, room_start, zone_h, zone_v, room_end, door_1, door_2, door_3,
        ) = _build_p3_shaped_building()

        route = engine.shortest_path(room_start.id, room_end.id)
        self.assertEqual([n.name for n in route.nodes], ["RoomStart", "ZoneH", "ZoneV", "RoomEnd"])

        manager = SandboxManager()
        occupant = manager.place_occupant(building.ordered_floors()[0], *room_start.center)
        self.assertIsNotNone(occupant)

        occupant.route = route
        occupant.node_index = 1  # ZoneH -> ZoneV, the bend hop
        occupant.state = SandboxOccupantState.MOVING

        # Same split-point math as the replay-studio test above.
        leg1 = math.dist(zone_h.center, door_2.center)
        leg2 = math.dist(door_2.center, zone_v.center)
        split = leg1 / (leg1 + leg2)

        occupant.edge_progress = split
        manager._sync_position(occupant)

        self.assertAlmostEqual(occupant.position[0], door_2.center[0], places=6)
        self.assertAlmostEqual(occupant.position[1], door_2.center[1], places=6)

        # Before the split: still close to ZoneH's own y, not already
        # deep into ZoneV's y range -- the same bend shape asserted
        # against interpolate_occupant_position() above.
        occupant.edge_progress = 0.1
        manager._sync_position(occupant)
        self.assertLess(occupant.position[1], 3.0)


if __name__ == "__main__":
    unittest.main()
