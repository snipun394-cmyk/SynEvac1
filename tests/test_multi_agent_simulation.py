import unittest

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.staircase import Staircase
from models.zone import Zone

from navigation.edge import Edge
from navigation.graph_builder import NavigationGraphGenerator

from pathfinding.engine import PathfindingEngine

from simulator.capacity import DefaultCapacityModel, StairCapacityModel, derive_stair_capacity
from simulator.congestion import DefaultCongestionModel, StairAwareCongestionModel
from simulator.coordinator import MultiAgentSimulation
from simulator.decision import ActionType, BehaviorDecision
from simulator.engine import OccupantSimulator
from simulator.occupant import OccupantState


def make_zone(name, **kwargs):

    fields = dict(x=0.0, y=0.0, width=2.0, height=2.0)
    fields.update(kwargs)

    return Zone(name=name, **fields)


class QueueFormationTests(unittest.TestCase):

    # A single capacity-1 door between a room and the exit corridor --
    # occupants must queue strictly in registration order.

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.room = make_zone("Room", x=0.0, y=0.0)
        self.corridor = make_zone("Corridor", x=10.0, y=0.0)

        self.floor.add_zone(self.room)
        self.floor.add_zone(self.corridor)

        # width chosen so DefaultCapacityModel derives capacity 1
        # (0.5 * 1.5 = 0.75 -> int() = 0, floored to 1).
        self.door = Door(
            name="Narrow", zone_a_id=self.room.id, zone_b_id=self.corridor.id,
            floor_id=self.floor.id, width=0.5,
        )
        self.floor.add_door(self.door)

        self.exit_obj = Exit(name="Ex", zone_id=self.corridor.id, floor_id=self.floor.id)
        self.floor.add_exit(self.exit_obj)

        self.graph = NavigationGraphGenerator().build(self.building)
        self.engine = PathfindingEngine(self.graph)

    def test_second_occupant_queues_behind_the_first(self):

        sim = MultiAgentSimulation(self.engine)

        sim.add_occupant(self.room.id, occupant_id="first")
        sim.add_occupant(self.room.id, occupant_id="second")

        result = sim.run()

        first_door_step = result.occupants["first"].steps[0]
        second_door_step = result.occupants["second"].steps[0]

        self.assertEqual(first_door_step.queue_wait_time, 0.0)
        self.assertGreater(second_door_step.queue_wait_time, 0.0)

        self.assertGreaterEqual(second_door_step.start_time, first_door_step.end_time)

    def test_queue_wait_time_matches_predecessor_crossing_time(self):

        sim = MultiAgentSimulation(self.engine)

        sim.add_occupant(self.room.id, occupant_id="first")
        sim.add_occupant(self.room.id, occupant_id="second")

        result = sim.run()

        first_step = result.occupants["first"].steps[0]
        second_step = result.occupants["second"].steps[0]

        # second joined the queue at t=0 (same depart time), released
        # exactly when first finished crossing the door.
        self.assertAlmostEqual(second_step.queue_wait_time, first_step.end_time)

    def test_total_queue_events_counts_every_wait(self):

        sim = MultiAgentSimulation(self.engine)

        for i in range(4):
            sim.add_occupant(self.room.id, occupant_id=f"p{i}")

        result = sim.run()

        # p0 never waits; p1, p2, p3 each wait once for the door.
        self.assertEqual(result.total_queue_events, 3)

    def test_ample_capacity_means_nobody_queues(self):

        # A separate building (not this class's narrow-door fixture)
        # with a single wide door, so there's no ambiguity about which
        # door PathfindingEngine's route uses.
        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        room = make_zone("Room", x=0.0, y=0.0)
        corridor = make_zone("Corridor", x=10.0, y=0.0)
        floor.add_zone(room)
        floor.add_zone(corridor)

        wide_door = Door(
            name="Wide", zone_a_id=room.id, zone_b_id=corridor.id,
            floor_id=floor.id, width=5.0,
        )
        floor.add_door(wide_door)
        floor.add_exit(Exit(name="Ex", zone_id=corridor.id, floor_id=floor.id))

        graph = NavigationGraphGenerator().build(building)
        engine = PathfindingEngine(graph)
        sim = MultiAgentSimulation(engine)

        for i in range(3):
            sim.add_occupant(room.id, occupant_id=f"p{i}")

        result = sim.run()

        # Capacity on the wide door (5.0m -> capacity 7) comfortably
        # exceeds 3 occupants, so nobody should need to queue at all.
        self.assertEqual(result.total_queue_events, 0)

        for timeline in result.occupants.values():
            for step in timeline.steps:
                self.assertEqual(step.queue_wait_time, 0.0)


class OccupancyTrackingTests(unittest.TestCase):

    def test_peak_edge_occupancy_reflects_concurrent_crossings(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        room = make_zone("Room", x=0.0, y=0.0)
        corridor = make_zone("Corridor", x=10.0, y=0.0)
        floor.add_zone(room)
        floor.add_zone(corridor)

        door = Door(
            name="Wide", zone_a_id=room.id, zone_b_id=corridor.id,
            floor_id=floor.id, width=5.0,
        )
        floor.add_door(door)
        floor.add_exit(Exit(name="Ex", zone_id=corridor.id, floor_id=floor.id))

        graph = NavigationGraphGenerator().build(building)
        engine = PathfindingEngine(graph)
        sim = MultiAgentSimulation(engine)

        for i in range(3):
            sim.add_occupant(room.id, occupant_id=f"p{i}")

        result = sim.run()

        self.assertEqual(result.peak_edge_occupancy[door.id], 3)


class CongestionEffectTests(unittest.TestCase):

    # A door wide enough for several people at once but narrow enough
    # that concurrent crossings measurably slow each other down.

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.room = make_zone("Room", x=0.0, y=0.0)
        self.corridor = make_zone("Corridor", x=30.0, y=0.0)

        self.floor.add_zone(self.room)
        self.floor.add_zone(self.corridor)

        # width 3.0 -> capacity int(3.0*1.5) = 4
        self.door = Door(
            name="D1", zone_a_id=self.room.id, zone_b_id=self.corridor.id,
            floor_id=self.floor.id, width=3.0,
        )
        self.floor.add_door(self.door)

        self.graph = NavigationGraphGenerator().build(self.building)
        self.engine = PathfindingEngine(self.graph)

    def test_crossing_with_others_present_is_slower_than_crossing_alone(self):

        solo_sim = MultiAgentSimulation(self.engine)
        solo_sim.add_occupant(self.room.id, self.corridor.id, occupant_id="solo")
        solo_result = solo_sim.run()
        solo_duration = solo_result.occupants["solo"].steps[0].end_time

        crowded_sim = MultiAgentSimulation(self.engine)
        for i in range(4):
            crowded_sim.add_occupant(self.room.id, self.corridor.id, occupant_id=f"p{i}")
        crowded_result = crowded_sim.run()

        # Every one of the 4 is admitted immediately (capacity 4), so
        # none of them queue -- any extra time is purely the
        # congestion speed effect, not queueing.
        for timeline in crowded_result.occupants.values():
            self.assertEqual(timeline.steps[0].queue_wait_time, 0.0)

        # p0 is admitted first, with nobody else yet on the edge --
        # by design (congestion reflects OTHER occupants sharing the
        # edge, see simulator/congestion.py), it experiences zero
        # slowdown, identical to the solo run.
        p0_duration = crowded_result.occupants["p0"].steps[0].end_time
        self.assertAlmostEqual(p0_duration, solo_duration)

        # p1/p2/p3 are each admitted onto the edge alongside more
        # occupants already present (1, 2, and 3 respectively), so
        # each should be measurably slower than crossing alone, with
        # crossing time increasing as more people are already on it.
        durations = [
            crowded_result.occupants[f"p{i}"].steps[0].end_time
            for i in range(4)
        ]

        for later_duration in durations[1:]:
            self.assertGreater(later_duration, solo_duration)

        self.assertEqual(durations, sorted(durations))


class MultiFloorCapacityTests(unittest.TestCase):

    def test_stair_capacity_forces_queueing_like_a_door(self):

        # Both occupants start at the same node (directly at the top
        # of the stair) and depart at the same time, so their attempts
        # to enter the capacity-1 stair genuinely overlap -- the same
        # scenario shape as QueueFormationTests, but for a Stair edge.
        building = Building(name="B")
        ground = building.create_floor(name="Ground Floor")
        floor1 = building.create_floor(name="Floor 1", height=3.0)

        lobby = make_zone("Lobby", x=0.0, y=0.0)
        upstairs = make_zone("Upstairs", x=0.0, y=0.0, floor_id=floor1.id)

        ground.add_zone(lobby)
        floor1.add_zone(upstairs)

        ground.add_exit(Exit(name="Ex", zone_id=lobby.id, floor_id=ground.id))

        stair = Staircase(
            name="Narrow Stair", from_floor_id=ground.id, to_floor_id=floor1.id,
            from_zone_id=lobby.id, to_zone_id=upstairs.id, width=0.5,
        )
        ground.add_stair(stair)

        graph = NavigationGraphGenerator().build(building)
        engine = PathfindingEngine(graph)
        sim = MultiAgentSimulation(engine)

        sim.add_occupant(upstairs.id, lobby.id, occupant_id="first")
        sim.add_occupant(upstairs.id, lobby.id, occupant_id="second")

        result = sim.run()

        first_stair_step = result.occupants["first"].steps[0]
        second_stair_step = result.occupants["second"].steps[0]

        self.assertEqual(first_stair_step.edge.id, stair.id)
        self.assertTrue(first_stair_step.is_floor_transition)

        self.assertEqual(first_stair_step.queue_wait_time, 0.0)
        self.assertGreater(second_stair_step.queue_wait_time, 0.0)


class StairCapacityModelTests(unittest.TestCase):

    def _make_stair_edge(self, width=1.5, vertical_height=3.0):

        building = Building(name="B")
        ground = building.create_floor(name="Ground Floor")
        floor1 = building.create_floor(name="Floor 1", height=vertical_height)

        lobby = make_zone("Lobby", x=0.0, y=0.0)
        upstairs = make_zone("Upstairs", x=0.0, y=0.0, floor_id=floor1.id)

        ground.add_zone(lobby)
        floor1.add_zone(upstairs)

        stair = Staircase(
            name="Stair", from_floor_id=ground.id, to_floor_id=floor1.id,
            from_zone_id=lobby.id, to_zone_id=upstairs.id, width=width,
        )
        ground.add_stair(stair)

        graph = NavigationGraphGenerator().build(building)

        return next(e for e in graph.edges if e.edge_type == Edge.STAIR)

    def test_stair_capacity_is_lower_than_default_model_for_the_same_edge(self):

        edge = self._make_stair_edge(width=1.5, vertical_height=3.0)

        default_capacity = DefaultCapacityModel().capacity(edge)
        stair_capacity = StairCapacityModel().capacity(edge)

        self.assertLess(stair_capacity, default_capacity)

    def test_non_stair_edges_are_delegated_to_the_base_model_unchanged(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")
        zone_a = make_zone("A", x=0.0, y=0.0)
        zone_b = make_zone("B", x=5.0, y=0.0)
        floor.add_zone(zone_a)
        floor.add_zone(zone_b)

        door = Door(name="D", zone_a_id=zone_a.id, zone_b_id=zone_b.id, floor_id=floor.id, width=2.0)
        floor.add_door(door)

        graph = NavigationGraphGenerator().build(building)
        edge = next(e for e in graph.edges if e.edge_type == Edge.DOOR)

        self.assertEqual(StairCapacityModel().capacity(edge), DefaultCapacityModel().capacity(edge))

    def test_capacity_is_always_floored_at_one(self):

        edge = self._make_stair_edge(width=0.1, vertical_height=30.0)

        self.assertGreaterEqual(StairCapacityModel().capacity(edge), 1)

    def test_derive_stair_capacity_matches_the_model(self):

        edge = self._make_stair_edge(width=2.0, vertical_height=6.0)

        self.assertEqual(
            derive_stair_capacity(edge.width, edge.walking_distance),
            StairCapacityModel().capacity(edge),
        )

    def test_derive_stair_capacity_handles_missing_width(self):

        self.assertEqual(derive_stair_capacity(None), StairCapacityModel.MINIMUM_CAPACITY)


class StairAwareCongestionModelTests(unittest.TestCase):

    def test_delegates_non_stair_edges_to_the_base_model_unchanged(self):

        door_edge = Edge(id="d1", edge_type=Edge.DOOR, from_node="a", to_node="b")

        base = DefaultCongestionModel()
        wrapped = StairAwareCongestionModel(base)

        self.assertAlmostEqual(
            wrapped.speed_factor(door_edge, 2, 3, opposing_occupants=5),
            base.speed_factor(door_edge, 2, 3),
        )

    def test_opposing_occupants_slow_a_stair_below_the_base_factor(self):

        stair_edge = Edge(id="s1", edge_type=Edge.STAIR, from_node="a", to_node="b")

        base = DefaultCongestionModel()
        wrapped = StairAwareCongestionModel(base)

        base_factor = base.speed_factor(stair_edge, 0, 2)
        counterflow_factor = wrapped.speed_factor(stair_edge, 0, 2, opposing_occupants=1)

        self.assertLess(counterflow_factor, base_factor)

    def test_speed_factor_never_drops_below_the_base_models_minimum(self):

        stair_edge = Edge(id="s1", edge_type=Edge.STAIR, from_node="a", to_node="b")

        wrapped = StairAwareCongestionModel(DefaultCongestionModel())

        factor = wrapped.speed_factor(stair_edge, 0, 2, opposing_occupants=50)

        self.assertGreaterEqual(factor, DefaultCongestionModel.MINIMUM_SPEED_FACTOR)


class StairCounterflowIntegrationTests(unittest.TestCase):

    # End-to-end through MultiAgentSimulation itself (not the
    # congestion model in isolation): two occupants cross the same
    # wide stair concurrently in opposite directions and should take
    # longer than a single occupant crossing alone, purely from the
    # counter-flow penalty (capacity is wide enough that neither one
    # ever queues).

    def _build(self):

        building = Building(name="B")
        ground = building.create_floor(name="Ground Floor")
        floor1 = building.create_floor(name="Floor 1", height=3.0)

        lobby = make_zone("Lobby", x=0.0, y=0.0)
        upstairs = make_zone("Upstairs", x=0.0, y=0.0, floor_id=floor1.id)

        ground.add_zone(lobby)
        floor1.add_zone(upstairs)

        # Deliberately only one Exit, at lobby -- an Exit at upstairs
        # too would give pathfinding a same-cost-or-cheaper shortcut
        # through the shared "outside" node instead of the stair
        # (both zones sit at the same (0, 0) coordinates here), which
        # would defeat this test's whole premise.
        ground.add_exit(Exit(name="Ex", zone_id=lobby.id, floor_id=ground.id))

        stair = Staircase(
            name="Wide Stair", from_floor_id=ground.id, to_floor_id=floor1.id,
            from_zone_id=lobby.id, to_zone_id=upstairs.id, width=4.0,
        )
        ground.add_stair(stair)

        graph = NavigationGraphGenerator().build(building)
        engine = PathfindingEngine(graph)

        return building, lobby, upstairs, engine

    def test_concurrent_opposite_direction_crossing_is_slower_than_alone(self):

        # Speed is computed once, at the moment each occupant is
        # admitted onto the edge (same as every other congestion
        # factor in this simulation -- it is never recomputed later
        # as new occupants join). "up" is registered first and is
        # admitted before "down" exists on the edge at all, so it
        # sees zero opposing occupants and pays no penalty; "down" is
        # admitted second and sees "up" already crossing, so it does.

        _, lobby, upstairs, solo_engine = self._build()

        solo_sim = MultiAgentSimulation(
            solo_engine, capacity_model=StairCapacityModel(),
            congestion_model=StairAwareCongestionModel(),
        )
        solo_sim.add_occupant(lobby.id, upstairs.id, occupant_id="solo")
        solo_result = solo_sim.run()
        solo_step = solo_result.occupants["solo"].steps[0]
        solo_duration = solo_step.end_time - solo_step.start_time

        _, lobby2, upstairs2, counterflow_engine = self._build()

        counterflow_sim = MultiAgentSimulation(
            counterflow_engine, capacity_model=StairCapacityModel(),
            congestion_model=StairAwareCongestionModel(),
        )
        counterflow_sim.add_occupant(lobby2.id, upstairs2.id, occupant_id="up")
        counterflow_sim.add_occupant(upstairs2.id, lobby2.id, occupant_id="down")
        counterflow_result = counterflow_sim.run()

        up_step = counterflow_result.occupants["up"].steps[0]
        up_duration = up_step.end_time - up_step.start_time
        down_step = counterflow_result.occupants["down"].steps[0]
        down_duration = down_step.end_time - down_step.start_time

        self.assertAlmostEqual(up_duration, solo_duration)
        self.assertGreater(down_duration, solo_duration)


class UnreachableOccupantTests(unittest.TestCase):

    def test_unreachable_occupant_is_recorded_and_never_blocks_run(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        reachable = make_zone("Reachable", x=0.0, y=0.0)
        lonely = make_zone("Lonely", x=100.0, y=100.0)

        floor.add_zone(reachable)
        floor.add_zone(lonely)

        floor.add_exit(Exit(name="Ex", zone_id=reachable.id, floor_id=floor.id))

        graph = NavigationGraphGenerator().build(building)
        engine = PathfindingEngine(graph)
        sim = MultiAgentSimulation(engine)

        sim.add_occupant(lonely.id, occupant_id="stuck")
        sim.add_occupant(reachable.id, occupant_id="fine")

        result = sim.run()

        self.assertIn("stuck", result.unreachable_occupant_ids)
        self.assertEqual(result.occupants["stuck"].state, OccupantState.UNREACHABLE)
        self.assertIsNone(result.occupants["stuck"].arrival_time)

        self.assertEqual(result.occupants["fine"].state, OccupantState.ARRIVED)
        self.assertIsNotNone(result.occupants["fine"].arrival_time)


class IndividualWalkingSpeedTests(unittest.TestCase):

    def test_faster_occupant_arrives_sooner_on_an_uncontested_route(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        room = make_zone("Room", x=0.0, y=0.0)
        corridor = make_zone("Corridor", x=20.0, y=0.0)

        floor.add_zone(room)
        floor.add_zone(corridor)

        door = Door(
            name="Wide", zone_a_id=room.id, zone_b_id=corridor.id,
            floor_id=floor.id, width=5.0,
        )
        floor.add_door(door)

        graph = NavigationGraphGenerator().build(building)
        engine = PathfindingEngine(graph)
        sim = MultiAgentSimulation(engine)

        sim.add_occupant(room.id, corridor.id, occupant_id="slow", walking_speed=0.5)
        sim.add_occupant(room.id, corridor.id, occupant_id="fast", walking_speed=2.0)

        result = sim.run()

        self.assertLess(
            result.occupants["fast"].arrival_time,
            result.occupants["slow"].arrival_time,
        )


class TrivialRouteTests(unittest.TestCase):

    def test_occupant_already_at_goal_arrives_immediately(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        zone = make_zone("Zone", x=0.0, y=0.0)
        floor.add_zone(zone)

        graph = NavigationGraphGenerator().build(building)
        engine = PathfindingEngine(graph)
        sim = MultiAgentSimulation(engine)

        sim.add_occupant(zone.id, zone.id, occupant_id="already-there", depart_time=5.0)

        result = sim.run()

        timeline = result.occupants["already-there"]
        self.assertEqual(timeline.state, OccupantState.ARRIVED)
        self.assertEqual(timeline.arrival_time, 5.0)
        self.assertEqual(timeline.steps, [])


class TerminationGuaranteeTests(unittest.TestCase):

    def test_many_occupants_through_one_narrow_door_all_eventually_arrive(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        room = make_zone("Room", x=0.0, y=0.0)
        corridor = make_zone("Corridor", x=10.0, y=0.0)
        floor.add_zone(room)
        floor.add_zone(corridor)

        door = Door(
            name="Bottleneck", zone_a_id=room.id, zone_b_id=corridor.id,
            floor_id=floor.id, width=0.1,
        )
        floor.add_door(door)
        floor.add_exit(Exit(name="Ex", zone_id=corridor.id, floor_id=floor.id))

        graph = NavigationGraphGenerator().build(building)
        engine = PathfindingEngine(graph)
        sim = MultiAgentSimulation(engine)

        occupant_count = 20

        for i in range(occupant_count):
            sim.add_occupant(room.id, occupant_id=f"p{i}")

        result = sim.run()

        for i in range(occupant_count):
            self.assertEqual(result.occupants[f"p{i}"].state, OccupantState.ARRIVED)

        self.assertIsNotNone(result.total_evacuation_time)


class DeterminismTests(unittest.TestCase):

    def test_identical_scenario_produces_identical_arrival_times(self):

        def build_and_run():

            building = Building(name="B")
            floor = building.create_floor(name="Ground Floor")

            room = make_zone("Room", x=0.0, y=0.0)
            corridor = make_zone("Corridor", x=10.0, y=0.0)
            floor.add_zone(room)
            floor.add_zone(corridor)

            door = Door(
                name="D1", zone_a_id=room.id, zone_b_id=corridor.id,
                floor_id=floor.id, width=0.6,
            )
            floor.add_door(door)
            floor.add_exit(Exit(name="Ex", zone_id=corridor.id, floor_id=floor.id))

            graph = NavigationGraphGenerator().build(building)
            engine = PathfindingEngine(graph)
            sim = MultiAgentSimulation(engine)

            for i in range(5):
                sim.add_occupant(room.id, occupant_id=f"p{i}")

            result = sim.run()

            return {oid: t.arrival_time for oid, t in result.occupants.items()}

        first_run = build_and_run()
        second_run = build_and_run()

        self.assertEqual(first_run, second_run)


class DefaultCapacityModelTests(unittest.TestCase):

    def test_exit_uses_its_own_authored_capacity(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")
        zone = make_zone("Zone", x=0.0, y=0.0)
        floor.add_zone(zone)

        exit_obj = Exit(name="Ex", zone_id=zone.id, floor_id=floor.id, capacity=12)
        floor.add_exit(exit_obj)

        graph = NavigationGraphGenerator().build(building)
        edge = next(e for e in graph.edges if e.edge_type == Edge.EXIT)

        self.assertEqual(DefaultCapacityModel().capacity(edge), 12)

    def test_door_capacity_is_derived_from_width_and_floored_at_one(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")
        zone_a = make_zone("A", x=0.0, y=0.0)
        zone_b = make_zone("B", x=5.0, y=0.0)
        floor.add_zone(zone_a)
        floor.add_zone(zone_b)

        narrow_door = Door(
            name="Narrow", zone_a_id=zone_a.id, zone_b_id=zone_b.id,
            floor_id=floor.id, width=0.1,
        )
        floor.add_door(narrow_door)

        graph = NavigationGraphGenerator().build(building)
        edge = next(e for e in graph.edges if e.edge_type == Edge.DOOR)

        self.assertGreaterEqual(DefaultCapacityModel().capacity(edge), 1)


class AddOccupantRoutePassthroughTests(unittest.TestCase):

    # add_occupant(route=...) skips internal OccupantSimulator planning
    # and uses the caller-supplied Route directly -- the primitive
    # that makes route-level (not just goal-level) choice possible.

    def test_supplied_route_is_used_as_is(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        zone_a = make_zone("A", x=0.0, y=0.0)
        zone_b = make_zone("B", x=5.0, y=0.0)
        floor.add_zone(zone_a)
        floor.add_zone(zone_b)

        door = Door(
            name="D1", zone_a_id=zone_a.id, zone_b_id=zone_b.id, floor_id=floor.id,
        )
        floor.add_door(door)
        floor.add_exit(Exit(name="Ex", zone_id=zone_b.id, floor_id=floor.id))

        graph = NavigationGraphGenerator().build(building)
        engine = PathfindingEngine(graph)

        route = engine.nearest_exit(zone_a.id)

        sim = MultiAgentSimulation(engine)
        sim.add_occupant(start_id=zone_a.id, occupant_id="p1", route=route)

        result = sim.run()

        self.assertEqual(result.occupants["p1"].traversed_edge_ids, route.edge_ids)
        self.assertEqual(result.occupants["p1"].state, OccupantState.ARRIVED)


class SubmitDecisionTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone_a = make_zone("A", x=0.0, y=0.0)
        self.zone_b = make_zone("B", x=5.0, y=0.0)
        self.floor.add_zone(self.zone_a)
        self.floor.add_zone(self.zone_b)

        door = Door(
            name="D1", zone_a_id=self.zone_a.id, zone_b_id=self.zone_b.id,
            floor_id=self.floor.id,
        )
        self.floor.add_door(door)
        self.exit_obj = Exit(name="Ex", zone_id=self.zone_b.id, floor_id=self.floor.id)
        self.floor.add_exit(self.exit_obj)

        self.graph = NavigationGraphGenerator().build(self.building)
        self.engine = PathfindingEngine(self.graph)

    def test_movement_decision_is_executed_like_add_occupant(self):

        sim = MultiAgentSimulation(self.engine)

        decision = BehaviorDecision(
            occupant_id="p1",
            action_type=ActionType.EVACUATE,
            start_id=self.zone_a.id,
            goal_id="outside",
        )
        sim.submit_decision(decision)

        result = sim.run()

        self.assertEqual(result.occupants["p1"].state, OccupantState.ARRIVED)
        self.assertEqual(result.occupants["p1"].traversed_edge_ids[-1], self.exit_obj.id)

    def test_non_movement_decision_registers_stationary_occupant(self):

        sim = MultiAgentSimulation(self.engine)

        decision = BehaviorDecision(
            occupant_id="p1",
            action_type=ActionType.WAIT,
            start_id=self.zone_a.id,
        )
        sim.submit_decision(decision)

        result = sim.run()

        self.assertEqual(result.occupants["p1"].state, OccupantState.STATIONARY)
        self.assertIsNone(result.occupants["p1"].arrival_time)
        self.assertEqual(result.occupants["p1"].steps, [])
        self.assertNotIn("p1", result.unreachable_occupant_ids)

    def test_route_unavailable_decision_registers_unreachable_not_stationary(self):

        # Ground-truth-classification correctness fix (validation Phase
        # 5 finding, docs/validation/technical_report.md §6): a
        # BehaviorDecision with goal_id=None/route=None looks identical
        # whether it came from a genuine WAIT/IGNORE (the case above) or
        # from a movement-required decision for which no route to any
        # exit exists. route_unavailable=True disambiguates the latter
        # -- submit_decision() must register it as UNREACHABLE, not
        # STATIONARY, and it must be counted in unreachable_occupant_ids.

        sim = MultiAgentSimulation(self.engine)

        decision = BehaviorDecision(
            occupant_id="p1",
            action_type=ActionType.EVACUATE,
            start_id=self.zone_a.id,
            route_unavailable=True,
        )
        sim.submit_decision(decision)

        result = sim.run()

        self.assertEqual(result.occupants["p1"].state, OccupantState.UNREACHABLE)
        self.assertIsNone(result.occupants["p1"].arrival_time)
        self.assertEqual(result.occupants["p1"].steps, [])
        self.assertIn("p1", result.unreachable_occupant_ids)

    def test_route_unavailable_true_takes_priority_over_a_supplied_goal_or_route(self):

        # route_unavailable is set by HumanBehaviorLayer.register() only
        # when goal_id/route are already both None (see
        # behavior/orchestrator.py) -- but submit_decision() itself
        # checks route_unavailable first, so it is the authoritative
        # signal regardless of what goal_id/route happen to carry.

        route = self.engine.nearest_exit(self.zone_a.id)

        sim = MultiAgentSimulation(self.engine)

        decision = BehaviorDecision(
            occupant_id="p1",
            action_type=ActionType.EVACUATE,
            start_id=self.zone_a.id,
            goal_id=route.goal.id,
            route=route,
            route_unavailable=True,
        )
        sim.submit_decision(decision)

        result = sim.run()

        self.assertEqual(result.occupants["p1"].state, OccupantState.UNREACHABLE)
        self.assertIn("p1", result.unreachable_occupant_ids)

    def test_decision_with_explicit_route_is_used_as_is(self):

        route = self.engine.nearest_exit(self.zone_a.id)

        sim = MultiAgentSimulation(self.engine)

        decision = BehaviorDecision(
            occupant_id="p1",
            action_type=ActionType.EVACUATE,
            start_id=self.zone_a.id,
            route=route,
        )
        sim.submit_decision(decision)

        result = sim.run()

        self.assertEqual(result.occupants["p1"].traversed_edge_ids, route.edge_ids)


class MostRecentDecisionWinsTests(unittest.TestCase):

    # BehaviorDecision is immutable -- a changed decision is a NEW
    # object submitted again for the same occupant_id, never a
    # mutation. These tests prove Simulation always executes the most
    # recently submitted decision, with no stale/duplicate events from
    # a superseded one.

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone_a = make_zone("A", x=0.0, y=0.0)
        self.zone_b = make_zone("B", x=5.0, y=0.0)
        self.floor.add_zone(self.zone_a)
        self.floor.add_zone(self.zone_b)

        door = Door(
            name="D1", zone_a_id=self.zone_a.id, zone_b_id=self.zone_b.id,
            floor_id=self.floor.id,
        )
        self.floor.add_door(door)
        self.exit_obj = Exit(name="Ex", zone_id=self.zone_b.id, floor_id=self.floor.id)
        self.floor.add_exit(self.exit_obj)

        self.graph = NavigationGraphGenerator().build(self.building)
        self.engine = PathfindingEngine(self.graph)

    def test_wait_then_evacuate_ends_up_evacuated_not_stationary(self):

        sim = MultiAgentSimulation(self.engine)

        sim.submit_decision(
            BehaviorDecision(
                occupant_id="p1", action_type=ActionType.WAIT, start_id=self.zone_a.id,
            )
        )
        sim.submit_decision(
            BehaviorDecision(
                occupant_id="p1", action_type=ActionType.EVACUATE,
                start_id=self.zone_a.id, goal_id="outside",
            )
        )

        result = sim.run()

        self.assertEqual(result.occupants["p1"].state, OccupantState.ARRIVED)
        self.assertEqual(len(result.occupants), 1)

    def test_two_movement_decisions_only_the_second_executes(self):

        # Both decisions would move p1, but to different places --
        # only the second (most recent) submission's route should
        # ever actually be walked; no doubled/duplicate steps.
        second_zone = make_zone("C", x=20.0, y=0.0)
        self.floor.add_zone(second_zone)
        door2 = Door(
            name="D2", zone_a_id=self.zone_a.id, zone_b_id=second_zone.id,
            floor_id=self.floor.id,
        )
        self.floor.add_door(door2)

        graph = NavigationGraphGenerator().build(self.building)
        engine = PathfindingEngine(graph)

        sim = MultiAgentSimulation(engine)

        sim.submit_decision(
            BehaviorDecision(
                occupant_id="p1", action_type=ActionType.EVACUATE,
                start_id=self.zone_a.id, goal_id="outside",
            )
        )
        sim.submit_decision(
            BehaviorDecision(
                occupant_id="p1", action_type=ActionType.EVACUATE,
                start_id=self.zone_a.id, goal_id=second_zone.id,
            )
        )

        result = sim.run()

        self.assertEqual(result.occupants["p1"].traversed_edge_ids, [door2.id])
        self.assertEqual(len(result.occupants["p1"].steps), 1)

    def test_re_decision_does_not_leave_stale_queue_entries(self):

        # p1 registers WAIT, then EVACUATE; p2 also evacuates through
        # the same (capacity-1) door. p1's stale WAIT registration
        # must not leave a phantom queue slot that ever gets released.
        narrow_door = Door(
            name="Narrow", zone_a_id=self.zone_a.id, zone_b_id=self.zone_b.id,
            floor_id=self.floor.id, width=0.5,
        )
        # Replace the wide setUp door with a narrow one for this test.
        self.floor.remove_door(
            next(d for d in self.floor.doors if d.name == "D1")
        )
        self.floor.add_door(narrow_door)

        graph = NavigationGraphGenerator().build(self.building)
        engine = PathfindingEngine(graph)

        sim = MultiAgentSimulation(engine)

        sim.submit_decision(
            BehaviorDecision(
                occupant_id="p1", action_type=ActionType.WAIT, start_id=self.zone_a.id,
            )
        )
        sim.submit_decision(
            BehaviorDecision(
                occupant_id="p1", action_type=ActionType.EVACUATE,
                start_id=self.zone_a.id, goal_id="outside",
            )
        )
        sim.add_occupant(self.zone_a.id, occupant_id="p2")

        result = sim.run()

        self.assertEqual(result.occupants["p1"].state, OccupantState.ARRIVED)
        self.assertEqual(result.occupants["p2"].state, OccupantState.ARRIVED)
        # Exactly one of the two had to queue behind the other for the
        # capacity-1 door -- not zero, not both.
        self.assertEqual(sim._total_queue_events, 1)


class MultiAgentIndependenceTests(unittest.TestCase):

    def test_multi_agent_files_never_touch_reference_or_engineering_models(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "simulator"

        new_files = [
            "occupant.py", "capacity.py", "congestion.py",
            "multi_agent_result.py", "coordinator.py", "decision.py",
        ]

        for filename in new_files:

            text = (package_dir / filename).read_text()

            self.assertNotIn(
                ".reference", text, f"{filename} touches .reference directly"
            )
            self.assertIsNone(
                re.search(r"^\s*(from|import)\s+(models|designer)\b", text, re.MULTILINE),
                f"{filename} imports models/designer directly",
            )

    def test_multi_agent_never_touches_pathfinding_search_internals(self):

        import pathlib

        text = (
            pathlib.Path(__file__).resolve().parent.parent
            / "simulator" / "coordinator.py"
        ).read_text()

        self.assertNotIn("_search", text)
        self.assertNotIn("_relax", text)

    def test_single_occupant_simulation_files_are_unmodified(self):

        import hashlib
        import pathlib
        import subprocess

        repo_root = pathlib.Path(__file__).resolve().parent.parent

        for filename in ("engine.py", "result.py"):

            diff = subprocess.run(
                ["git", "diff", "--quiet", "HEAD", "--", f"simulator/{filename}"],
                cwd=repo_root,
            )

            self.assertEqual(
                diff.returncode, 0,
                f"simulator/{filename} has uncommitted changes -- V1 must stay frozen",
            )


if __name__ == "__main__":
    unittest.main()
