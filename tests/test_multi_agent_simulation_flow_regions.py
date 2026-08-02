import unittest

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.staircase import Staircase
from models.zone import Zone

from navigation.edge import Edge
from navigation.flow_region import FlowRegion
from navigation.graph_builder import NavigationGraphGenerator

from pathfinding.engine import PathfindingEngine

from simulator.congestion import StairAwareCongestionModel
from simulator.coordinator import MultiAgentSimulation
from simulator.flow_region_capacity import FlowRegionCapacityModel
from simulator.flow_region_congestion import FlowRegionCongestionModel
from simulator.occupant import OccupantState


def make_zone(name, **kwargs):

    fields = dict(x=0.0, y=0.0, width=2.0, height=2.0)
    fields.update(kwargs)

    return Zone(name=name, **fields)


def make_merge_building():

    # Two stairs (from two separate upper floors) converging into one
    # ground-floor Lobby that has exactly one way out -- the same
    # "stair chain plus the door/exit it discharges into" shape the
    # whole Option D campaign targets, and the same topology already
    # proven to infer as one MERGE FlowRegion in
    # tests/test_flow_region_inference.py.

    building = Building(name="B")
    ground = building.create_floor(name="Ground Floor")
    floor1 = building.create_floor(name="Floor 1")
    floor2 = building.create_floor(name="Floor 2")

    lobby = make_zone("Lobby")
    zone_a = make_zone("Zone A")
    zone_b = make_zone("Zone B")

    ground.add_zone(lobby)
    floor1.add_zone(zone_a)
    floor2.add_zone(zone_b)

    stair_a = Staircase(
        name="Stair A",
        from_floor_id=floor1.id,
        to_floor_id=ground.id,
        from_zone_id=zone_a.id,
        to_zone_id=lobby.id,
    )
    floor1.add_stair(stair_a)

    stair_b = Staircase(
        name="Stair B",
        from_floor_id=floor2.id,
        to_floor_id=ground.id,
        from_zone_id=zone_b.id,
        to_zone_id=lobby.id,
    )
    floor2.add_stair(stair_b)

    exit_obj = Exit(name="Front Exit", zone_id=lobby.id, floor_id=ground.id)
    ground.add_exit(exit_obj)

    return building, zone_a, zone_b, lobby, stair_a, stair_b, exit_obj


def make_shared_region(edge_ids, total_length, representative_width, region_kind=FlowRegion.MERGE):

    # A hand-built FlowRegion with deliberately chosen dimensions, the
    # same "pick numbers that force an exact, deterministic capacity"
    # convention tests/test_multi_agent_simulation.py itself already
    # uses for its own narrow-door fixture (width=0.5 -> capacity 1).
    # FlowRegionCapacityModel.capacity == int(total_length *
    # representative_width * JAM_DENSITY_PEOPLE_PER_SQUARE_METER).

    return FlowRegion(
        id="flow-region-merge-test",
        edge_ids=tuple(sorted(edge_ids)),
        region_kind=region_kind,
        total_length=total_length,
        representative_width=representative_width,
    )


class BackwardCompatibilityTests(unittest.TestCase):

    # Milestone 3's own primary regression gate: every existing caller
    # (which never passes flow_region_map) must behave identically to
    # before this milestone.

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.room = make_zone("Room", x=0.0, y=0.0)
        self.corridor = make_zone("Corridor", x=10.0, y=0.0)
        self.floor.add_zone(self.room)
        self.floor.add_zone(self.corridor)

        self.door = Door(
            name="Narrow", zone_a_id=self.room.id, zone_b_id=self.corridor.id,
            floor_id=self.floor.id, width=0.5,
        )
        self.floor.add_door(self.door)

        self.exit_obj = Exit(name="Ex", zone_id=self.corridor.id, floor_id=self.floor.id)
        self.floor.add_exit(self.exit_obj)

        self.graph = NavigationGraphGenerator().build(self.building)
        self.engine = PathfindingEngine(self.graph)

    def test_flow_region_map_defaults_to_none(self):

        sim = MultiAgentSimulation(self.engine)

        self.assertIsNone(sim.flow_region_map)

    def test_omitting_flow_region_map_preserves_todays_queueing_behavior(self):

        sim = MultiAgentSimulation(self.engine)

        sim.add_occupant(self.room.id, occupant_id="first")
        sim.add_occupant(self.room.id, occupant_id="second")

        result = sim.run()

        first_step = result.occupants["first"].steps[0]
        second_step = result.occupants["second"].steps[0]

        self.assertEqual(first_step.queue_wait_time, 0.0)
        self.assertGreater(second_step.queue_wait_time, 0.0)
        self.assertGreaterEqual(second_step.start_time, first_step.end_time)

    def test_none_flow_region_map_never_touches_a_dict_lookup(self):

        # Structural check, not just behavioral: with no map supplied,
        # _resolve_admission is a literal passthrough.
        sim = MultiAgentSimulation(self.engine)

        edge = self.graph.edges[0]
        admission_object, admission_key = sim._resolve_admission(edge)

        self.assertIs(admission_object, edge)
        self.assertEqual(admission_key, edge.id)

    def test_a_graph_with_only_a_single_dead_end_exit_produces_a_full_passthrough(self):

        # A building with exactly one edge overall has nothing to
        # chain or merge with -- FlowRegionInferencer gives it one
        # trivial SINGLE region, and _resolve_admission's own SINGLE
        # special case (see coordinator.py) means wiring that real,
        # non-empty map in is still a full passthrough to the edge
        # itself, identical to omitting it entirely.
        lone_building = Building(name="Lone")
        lone_floor = lone_building.create_floor(name="Ground Floor")
        lone_zone = make_zone("Room")
        lone_floor.add_zone(lone_zone)
        lone_exit = Exit(name="Ex", zone_id=lone_zone.id, floor_id=lone_floor.id)
        lone_floor.add_exit(lone_exit)

        lone_graph = NavigationGraphGenerator().build(lone_building)

        self.assertEqual(
            lone_graph.flow_regions[lone_exit.id].region_kind,
            FlowRegion.SINGLE,
        )

        sim_without_map = MultiAgentSimulation(PathfindingEngine(lone_graph))
        sim_without_map.add_occupant(lone_zone.id, occupant_id="walker")
        result_without_map = sim_without_map.run()

        sim_with_map = MultiAgentSimulation(
            PathfindingEngine(lone_graph),
            flow_region_map=lone_graph.flow_regions,
        )
        sim_with_map.add_occupant(lone_zone.id, occupant_id="walker")
        result_with_map = sim_with_map.run()

        self.assertEqual(
            result_without_map.total_evacuation_time,
            result_with_map.total_evacuation_time,
        )


class MergePointSharedQueueTests(unittest.TestCase):

    # The core new capability: two structurally different edges (two
    # separate Stairs) sharing one FlowRegion must share one admission
    # slot and one FIFO queue, even though neither occupant is ever on
    # the other's physical edge.

    def setUp(self):

        (
            self.building,
            self.zone_a,
            self.zone_b,
            self.lobby,
            self.stair_a,
            self.stair_b,
            self.exit_obj,
        ) = make_merge_building()

        self.graph = NavigationGraphGenerator().build(self.building)

        # capacity 1: total_length=1.0 * representative_width=0.5 * 2.0
        # (JAM_DENSITY_PEOPLE_PER_SQUARE_METER) = 1.0 -> int() = 1.
        self.region = make_shared_region(
            edge_ids=(self.stair_a.id, self.stair_b.id, self.exit_obj.id),
            total_length=1.0,
            representative_width=0.5,
        )

        self.flow_region_map = {
            self.stair_a.id: self.region,
            self.stair_b.id: self.region,
            self.exit_obj.id: self.region,
        }

    def make_sim(self):

        return MultiAgentSimulation(
            PathfindingEngine(self.graph),
            capacity_model=FlowRegionCapacityModel(),
            congestion_model=FlowRegionCongestionModel(),
            flow_region_map=self.flow_region_map,
        )

    def test_capacity_is_shared_across_both_stairs(self):

        sim = self.make_sim()

        sim.add_occupant(self.zone_a.id, occupant_id="on_a")
        sim.add_occupant(self.zone_b.id, occupant_id="on_b")

        result = sim.run()

        on_a_stair_step = result.occupants["on_a"].steps[0]
        on_b_stair_step = result.occupants["on_b"].steps[0]

        # on_a registered first, and the shared region's capacity is 1
        # -- on_b must queue for the SAME slot even though it is trying
        # to enter a completely different edge (Stair B, not Stair A).
        self.assertEqual(on_a_stair_step.queue_wait_time, 0.0)
        self.assertGreater(on_b_stair_step.queue_wait_time, 0.0)

    def test_queued_occupant_is_released_by_a_departure_on_the_other_member_edge(self):

        sim = self.make_sim()

        sim.add_occupant(self.zone_a.id, occupant_id="on_a")
        sim.add_occupant(self.zone_b.id, occupant_id="on_b")

        result = sim.run()

        on_a_step = result.occupants["on_a"].steps[0]
        on_b_step = result.occupants["on_b"].steps[0]

        # on_b (queued for Stair B) is released exactly when on_a
        # finishes crossing Stair A -- a departure on a DIFFERENT edge
        # than the one on_b is actually waiting for.
        self.assertAlmostEqual(on_b_step.queue_wait_time, on_a_step.end_time)
        self.assertAlmostEqual(on_b_step.start_time, on_a_step.end_time)

    def test_without_a_shared_region_both_occupants_are_admitted_immediately(self):

        # Control case: the same two occupants, same topology, but
        # WITHOUT the flow_region_map -- each Stair has its own
        # independent (and, by default, ample) capacity, so neither
        # should need to queue at all. Demonstrates the shared-queueing
        # behavior above is caused by the FlowRegion, not the topology.
        sim = MultiAgentSimulation(PathfindingEngine(self.graph))

        sim.add_occupant(self.zone_a.id, occupant_id="on_a")
        sim.add_occupant(self.zone_b.id, occupant_id="on_b")

        result = sim.run()

        self.assertEqual(result.occupants["on_a"].steps[0].queue_wait_time, 0.0)
        self.assertEqual(result.occupants["on_b"].steps[0].queue_wait_time, 0.0)


class DualTrackingPeakEdgeOccupancyTests(unittest.TestCase):

    # peak_edge_occupancy must keep reporting TRUE per-edge peaks (how
    # many occupants were literally on that one edge at once), never
    # the region's own shared, larger admission count.

    def setUp(self):

        (
            self.building,
            self.zone_a,
            self.zone_b,
            self.lobby,
            self.stair_a,
            self.stair_b,
            self.exit_obj,
        ) = make_merge_building()

        self.graph = NavigationGraphGenerator().build(self.building)

        # capacity 2: total_length=2.0 * representative_width=0.5 *
        # 2.0 = 2.0 -> int() = 2, so both stairs' occupants can be
        # admitted into the shared region at the same time.
        self.region = make_shared_region(
            edge_ids=(self.stair_a.id, self.stair_b.id, self.exit_obj.id),
            total_length=2.0,
            representative_width=0.5,
        )

        self.flow_region_map = {
            self.stair_a.id: self.region,
            self.stair_b.id: self.region,
            self.exit_obj.id: self.region,
        }

    def test_peak_edge_occupancy_reflects_only_that_edges_own_occupants(self):

        sim = MultiAgentSimulation(
            PathfindingEngine(self.graph),
            capacity_model=FlowRegionCapacityModel(),
            congestion_model=FlowRegionCongestionModel(),
            flow_region_map=self.flow_region_map,
        )

        sim.add_occupant(self.zone_a.id, occupant_id="on_a")
        sim.add_occupant(self.zone_b.id, occupant_id="on_b")

        result = sim.run()

        # Both admitted into the shared region simultaneously (region
        # capacity 2), but each is physically on their OWN, different
        # stair -- exactly one occupant per edge, never two.
        self.assertEqual(result.peak_edge_occupancy[self.stair_a.id], 1)
        self.assertEqual(result.peak_edge_occupancy[self.stair_b.id], 1)

        self.assertEqual(
            result.occupants["on_a"].steps[0].queue_wait_time, 0.0,
        )
        self.assertEqual(
            result.occupants["on_b"].steps[0].queue_wait_time, 0.0,
        )


class SingleEdgeRegionFidelityTests(unittest.TestCase):

    # FlowRegion.SINGLE's own docstring promise: a trivial, one-edge
    # region must be "identical in effect to today's per-edge
    # behavior" all the way through congestion, not just admission
    # bookkeeping -- including a solo Stair's counterflow penalty,
    # which FlowRegionCongestionModel's region-shaped formula does not
    # apply.

    def setUp(self):

        self.building = Building(name="B")
        self.ground = self.building.create_floor(name="Ground Floor")
        self.floor1 = self.building.create_floor(name="Floor 1")

        self.zone_ground = make_zone("Ground Zone")
        self.zone_floor1 = make_zone("Floor 1 Zone")
        self.ground.add_zone(self.zone_ground)
        self.floor1.add_zone(self.zone_floor1)

        self.stair = Staircase(
            name="Solo Stair",
            from_floor_id=self.floor1.id,
            to_floor_id=self.ground.id,
            from_zone_id=self.zone_floor1.id,
            to_zone_id=self.zone_ground.id,
            width=4.0,
        )
        self.floor1.add_stair(self.stair)

        # TWO exits from the ground zone -- a genuine fork, so the
        # stair (which feeds into the ground zone) is never merged
        # into either exit's own chain, and stays a real SINGLE
        # region. A ground zone with only one exit would instead chain
        # the stair and that exit together (see
        # test_flow_region_inference.py's own chain test) -- not the
        # scenario this fixture needs.
        self.exit_obj = Exit(name="Ex1", zone_id=self.zone_ground.id, floor_id=self.ground.id)
        self.ground.add_exit(self.exit_obj)

        self.exit_obj_2 = Exit(name="Ex2", zone_id=self.zone_ground.id, floor_id=self.ground.id)
        self.ground.add_exit(self.exit_obj_2)

        self.graph = NavigationGraphGenerator().build(self.building)

    def test_solo_stair_resolves_to_the_edge_itself_not_the_trivial_region(self):

        sim = MultiAgentSimulation(
            PathfindingEngine(self.graph),
            flow_region_map=self.graph.flow_regions,
        )

        stair_edge = next(e for e in self.graph.edges if e.edge_type == Edge.STAIR)

        # This building's stair has no chain/merge partner --
        # FlowRegionInferencer gives it a SINGLE-kind trivial region.
        self.assertEqual(
            self.graph.flow_regions[stair_edge.id].region_kind,
            FlowRegion.SINGLE,
        )

        admission_object, admission_key = sim._resolve_admission(stair_edge)

        self.assertIs(admission_object, stair_edge)
        self.assertEqual(admission_key, stair_edge.id)

    def test_solo_stair_counterflow_penalty_still_applies_with_a_flow_region_map_active(self):

        # Two occupants crossing the same solo stair in opposite
        # directions -- both routes forced explicitly via `route=`
        # rather than pathfinding. Speed is computed once, at the
        # moment each occupant is admitted (same as everywhere else in
        # this simulation, see StairCounterflowIntegrationTests in
        # tests/test_multi_agent_simulation.py): "going_down" is
        # registered first and is admitted before "going_up" exists on
        # the edge at all, so it sees zero opposing occupants and pays
        # no penalty; "going_up" is admitted second and sees
        # "going_down" already crossing, so it does.
        from pathfinding.route import Route

        stair_edge = next(e for e in self.graph.edges if e.edge_type == Edge.STAIR)
        node_up = self.graph.find_node(self.zone_floor1.id)
        node_down = self.graph.find_node(self.zone_ground.id)

        sim = MultiAgentSimulation(
            PathfindingEngine(self.graph),
            capacity_model=FlowRegionCapacityModel(),
            congestion_model=FlowRegionCongestionModel(),
            flow_region_map=self.graph.flow_regions,
        )

        down_route = Route(
            nodes=[node_up, node_down], edges=[stair_edge],
            total_cost=stair_edge.traversal_cost, total_distance=stair_edge.walking_distance,
        )
        up_route = Route(
            nodes=[node_down, node_up], edges=[stair_edge],
            total_cost=stair_edge.traversal_cost, total_distance=stair_edge.walking_distance,
        )

        sim.add_occupant(node_up.id, occupant_id="going_down", route=down_route)
        sim.add_occupant(node_down.id, occupant_id="going_up", route=up_route, depart_time=0.0)

        result = sim.run()

        going_up_step = result.occupants["going_up"].steps[0]

        base_duration = stair_edge.traversal_cost / Edge.ASSUMED_WALK_SPEED_M_PER_S
        actual_duration = going_up_step.end_time - going_up_step.start_time

        # StairAwareCongestionModel's counterflow penalty must have
        # slowed this occupant down relative to walking alone -- if the
        # solo stair had instead resolved to the trivial FlowRegion
        # (ignoring opposing_occupants, see FlowRegionCongestionModel),
        # this crossing would take exactly base_duration instead.
        self.assertGreater(actual_duration, base_duration)


class MixedEdgeAndRegionTransitionTests(unittest.TestCase):

    # A single occupant's route crosses BOTH a plain (unmapped/no-
    # region-partner) edge AND a multi-edge FlowRegion edge, back to
    # back -- confirms _resolve_admission's per-edge dispatch works
    # correctly hop by hop along one fixed route, not just in isolated
    # single-edge scenarios.

    def test_route_through_an_unmapped_door_then_a_mapped_merge_region(self):

        building = Building(name="B")
        ground = building.create_floor(name="Ground Floor")
        floor1 = building.create_floor(name="Floor 1")

        office = make_zone("Office", x=0.0, y=0.0)
        landing = make_zone("Landing", x=10.0, y=0.0)
        floor1.add_zone(office)
        floor1.add_zone(landing)

        lobby = make_zone("Lobby")
        other_zone = make_zone("Other Zone")
        ground.add_zone(lobby)

        floor2 = building.create_floor(name="Floor 2")
        floor2.add_zone(other_zone)

        door = Door(
            name="Office Door", zone_a_id=office.id, zone_b_id=landing.id,
            floor_id=floor1.id,
        )
        floor1.add_door(door)

        stair_a = Staircase(
            name="Stair A", from_floor_id=floor1.id, to_floor_id=ground.id,
            from_zone_id=landing.id, to_zone_id=lobby.id,
        )
        floor1.add_stair(stair_a)

        stair_b = Staircase(
            name="Stair B", from_floor_id=floor2.id, to_floor_id=ground.id,
            from_zone_id=other_zone.id, to_zone_id=lobby.id,
        )
        floor2.add_stair(stair_b)

        exit_obj = Exit(name="Front Exit", zone_id=lobby.id, floor_id=ground.id)
        ground.add_exit(exit_obj)

        graph = NavigationGraphGenerator().build(building)

        # The Office Door is on its own, unshared chain segment (Office
        # has only one neighbor, Landing has only the door and the
        # stair) -- FlowRegionInferencer should still group Door+StairA
        # together as one CHAIN (Landing has out-degree 1), which lets
        # us assert the door and stair share a region while remaining a
        # genuinely different edge type than the merge below.
        door_region = graph.flow_regions[door.id]
        self.assertEqual(door_region.id, graph.flow_regions[stair_a.id].id)

        sim = MultiAgentSimulation(
            PathfindingEngine(graph),
            capacity_model=FlowRegionCapacityModel(),
            congestion_model=FlowRegionCongestionModel(),
            flow_region_map=graph.flow_regions,
        )

        sim.add_occupant(office.id, occupant_id="walker")

        result = sim.run()

        steps = result.occupants["walker"].steps

        # Door, then Stair A, then Exit -- three hops, all successfully
        # simulated end to end despite crossing two different
        # FlowRegions (the Door+StairA chain, then the Exit's own
        # region, which also includes StairB even though this occupant
        # never sets foot on StairB).
        self.assertEqual([step.edge.id for step in steps], [door.id, stair_a.id, exit_obj.id])
        self.assertEqual(result.occupants["walker"].state, OccupantState.ARRIVED)


if __name__ == "__main__":
    unittest.main()
