import unittest

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.obstacle import Obstacle
from models.staircase import Staircase
from models.zone import Zone

from navigation.edge import Edge
from navigation.graph_builder import NavigationGraphGenerator
from navigation.node import Node
from navigation.validation import ValidationReport

from pathfinding.engine import PathfindingEngine

from simulator.capacity import StairCapacityModel
from simulator.congestion import StairAwareCongestionModel
from simulator.coordinator import MultiAgentSimulation
from simulator.occupant import OccupantState


# =====================================================
# Stair Simulation Reliability & Multi-Floor Reachability Audit
# milestone -- Phases 2-9, 16, 19. Empirical, executable proof for every
# claim in docs/architecture/stair_simulation_reliability_audit.md.
# Every building fixture is constructed directly (Building/Floor/Zone/
# Door/Exit/Staircase), run through the REAL NavigationGraphGenerator ->
# PathfindingEngine -> MultiAgentSimulation chain -- never a mock or a
# shortcut around production code.
# =====================================================


def make_zone(zone_id, floor_id, x=0.0, y=0.0, width=10.0, height=10.0, name=None):

    return Zone(id=zone_id, name=name or zone_id, floor_id=floor_id, x=x, y=y, width=width, height=height)


def make_stair(stair_id, from_floor, to_floor, from_zone_id, to_zone_id, width=1.5):

    stair = Staircase(
        id=stair_id, name=stair_id, from_floor_id=from_floor.id, to_floor_id=to_floor.id,
        from_zone_id=from_zone_id, to_zone_id=to_zone_id, width=width,
    )
    from_floor.add_stair(stair)
    return stair


# =====================================================
# Phase 2 -- historical zero-duration bug reproduction
# =====================================================


class HistoricalZeroDurationBugReproductionTests(unittest.TestCase):

    def test_missing_from_floor_id_no_longer_produces_instantaneous_traversal(self):

        building = Building(name="B")
        ground = building.create_floor(name="Ground")
        floor1 = building.create_floor(name="Floor 1", height=3.0)

        lobby = make_zone("lobby", ground.id)
        upstairs = make_zone("upstairs", floor1.id)
        ground.add_zone(lobby)
        floor1.add_zone(upstairs)

        # The historical bug: from_floor_id left blank (never set), the
        # single most common real-world way this field goes missing.
        stair = Staircase(
            id="S1", name="Broken Stair", from_floor_id="", to_floor_id=floor1.id,
            from_zone_id=lobby.id, to_zone_id=upstairs.id, width=1.5,
        )
        ground.add_stair(stair)

        graph = NavigationGraphGenerator().build(building)
        edge = next(e for e in graph.edges if e.edge_type == Edge.STAIR)

        # Reproduced: Staircase.travel_distance() itself still honestly
        # returns 0.0 (vertical_height() cannot resolve from_floor_id) --
        # this class is untouched, per this milestone's own "preserve
        # architecture" instruction.
        self.assertEqual(stair.vertical_height(building), 0.0)
        self.assertEqual(stair.travel_distance(building), 0.0)

        # FIXED: the Edge itself no longer silently carries a 0.0
        # walking_distance -- it honestly falls back to None ("not
        # derivable"), Edge's own pre-existing contract.
        self.assertIsNone(edge.walking_distance)
        self.assertEqual(edge.traversal_cost, Edge.DEFAULT_TRAVERSAL_COST)

        engine = PathfindingEngine(graph)
        sim = MultiAgentSimulation(engine)
        sim.add_occupant(lobby.id, upstairs.id, occupant_id="occ")
        result = sim.run()

        step = result.occupants["occ"].steps[0]

        self.assertGreater(step.end_time, step.start_time)  # no longer instantaneous
        self.assertEqual(result.occupants["occ"].state, OccupantState.ARRIVED)

    def test_validation_flags_the_degenerate_stair(self):

        building = Building(name="B")
        ground = building.create_floor(name="Ground")
        floor1 = building.create_floor(name="Floor 1", height=3.0)

        lobby = make_zone("lobby", ground.id)
        upstairs = make_zone("upstairs", floor1.id)
        ground.add_zone(lobby)
        floor1.add_zone(upstairs)

        stair = Staircase(
            id="S1", name="Broken Stair", from_floor_id="", to_floor_id=floor1.id,
            from_zone_id=lobby.id, to_zone_id=upstairs.id, width=1.5,
        )
        ground.add_stair(stair)

        graph = NavigationGraphGenerator().build(building)

        issues = graph.validate().by_code("stair_zero_traversal_distance")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, ValidationReport.WARNING)
        self.assertEqual(issues[0].object_id, "S1")

    def test_correctly_configured_stair_never_flagged(self):

        building = Building(name="B")
        ground = building.create_floor(name="Ground")
        floor1 = building.create_floor(name="Floor 1", height=3.0)

        lobby = make_zone("lobby", ground.id)
        upstairs = make_zone("upstairs", floor1.id)
        ground.add_zone(lobby)
        floor1.add_zone(upstairs)

        make_stair("S1", ground, floor1, lobby.id, upstairs.id)

        graph = NavigationGraphGenerator().build(building)

        self.assertEqual(graph.validate().by_code("stair_zero_traversal_distance"), [])
        edge = next(e for e in graph.edges if e.edge_type == Edge.STAIR)
        self.assertGreater(edge.walking_distance, 0.0)


# =====================================================
# Phase 3 -- multi-floor reachability matrix
# =====================================================


class MultiFloorReachabilityMatrixTests(unittest.TestCase):

    def _linear_building(self, floor_count):

        building = Building(name=f"{floor_count}-Floor Building")
        floors = [building.create_floor(name=f"Floor {i}", height=3.0) for i in range(floor_count)]

        zones = []
        for i, floor in enumerate(floors):
            zone = make_zone(f"zone-{i}", floor.id)
            floor.add_zone(zone)
            zones.append(zone)

        floors[0].add_exit(Exit(id="EXIT-0", name="Exit", zone_id=zones[0].id, floor_id=floors[0].id))

        stairs = []
        for i in range(floor_count - 1):
            stairs.append(make_stair(f"S{i}", floors[i], floors[i + 1], zones[i].id, zones[i + 1].id))

        return building, floors, zones, stairs

    def _run_matrix(self, floor_count):

        building, floors, zones, stairs = self._linear_building(floor_count)
        graph = NavigationGraphGenerator().build(building)
        engine = PathfindingEngine(graph)

        rows = []

        for i, floor in enumerate(floors):

            sim = MultiAgentSimulation(engine)
            sim.add_occupant(zones[i].id, occupant_id=f"occ-{i}")
            result = sim.run()

            occupant = result.occupants[f"occ-{i}"]
            route_exists = occupant.route is not None and (i == 0 or len(occupant.route.edges) > 0)
            evacuated = occupant.state == OccupantState.ARRIVED

            rows.append((floor.name, route_exists, evacuated, len(occupant.steps)))

            with self.subTest(floor=floor.name):
                self.assertTrue(route_exists, f"{floor.name}: no route found")
                self.assertTrue(evacuated, f"{floor.name}: did not evacuate")
                self.assertEqual(occupant.steps[-1].edge.edge_type if occupant.steps else None,
                                  Edge.EXIT if i == 0 else Edge.EXIT)

        return rows

    def test_2_floor_building_full_matrix(self):
        self._run_matrix(2)

    def test_3_floor_building_full_matrix(self):
        self._run_matrix(3)

    def test_4_floor_building_full_matrix(self):
        self._run_matrix(4)


# =====================================================
# Phase 4 -- chained stairs (Floor 3 -> Floor 2 -> Floor 1 -> Ground -> Exit)
# =====================================================


class ChainedStairsTests(unittest.TestCase):

    def test_occupant_traverses_three_chained_stairs_to_exit(self):

        building = Building(name="Chained Building")
        ground = building.create_floor(name="Ground", height=3.0)
        floor1 = building.create_floor(name="Floor 1", height=3.0)
        floor2 = building.create_floor(name="Floor 2", height=3.0)
        floor3 = building.create_floor(name="Floor 3", height=3.0)

        z_ground = make_zone("z-ground", ground.id)
        z1 = make_zone("z1", floor1.id)
        z2 = make_zone("z2", floor2.id)
        z3 = make_zone("z3", floor3.id)

        for floor, zone in ((ground, z_ground), (floor1, z1), (floor2, z2), (floor3, z3)):
            floor.add_zone(zone)

        ground.add_exit(Exit(id="EXIT-G", name="Exit", zone_id=z_ground.id, floor_id=ground.id))

        s1 = make_stair("STAIR-1", ground, floor1, z_ground.id, z1.id)
        s2 = make_stair("STAIR-2", floor1, floor2, z1.id, z2.id)
        s3 = make_stair("STAIR-3", floor2, floor3, z2.id, z3.id)

        graph = NavigationGraphGenerator().build(building)
        engine = PathfindingEngine(graph)
        sim = MultiAgentSimulation(engine)

        sim.add_occupant(z3.id, occupant_id="occ")
        result = sim.run()

        occupant = result.occupants["occ"]

        self.assertEqual(occupant.state, OccupantState.ARRIVED)
        self.assertEqual(len(occupant.steps), 4)  # STAIR-3, STAIR-2, STAIR-1, EXIT-G

        stair_steps = [s for s in occupant.steps if s.edge.edge_type == Edge.STAIR]
        self.assertEqual([s.edge.id for s in stair_steps], ["STAIR-3", "STAIR-2", "STAIR-1"])

        # No teleportation, no zero-distance stair, correct floor after
        # every single traversal (traced via from_node/to_node.floor_id).
        expected_floor_sequence = [floor3.id, floor2.id, floor1.id, ground.id]
        for index, step in enumerate(occupant.steps):
            self.assertEqual(step.from_node.floor_id, expected_floor_sequence[index])
            self.assertGreater(step.end_time, step.start_time, f"step {index} ({step.edge.id}) was instantaneous")
            if step.edge.edge_type == Edge.STAIR:
                self.assertGreater(step.distance, 0.0)

        self.assertEqual(occupant.steps[-1].to_node.id, Node.OUTSIDE_NODE_ID)


# =====================================================
# Phase 5 -- shared multi-floor staircase architecture
# =====================================================


class SharedMultiFloorStaircaseArchitectureTests(unittest.TestCase):

    def test_a_single_staircase_object_connects_exactly_two_floors(self):

        # models.staircase.Staircase's own structural shape -- from_
        # floor_id/to_floor_id, exactly one pair. There is no field
        # allowing a third floor.
        stair = Staircase(from_floor_id="a", to_floor_id="b")
        field_names = {f.name for f in stair.__dataclass_fields__.values()}
        self.assertIn("from_floor_id", field_names)
        self.assertIn("to_floor_id", field_names)
        self.assertNotIn("floor_ids", field_names)
        self.assertNotIn("intermediate_floor_id", field_names)

    def test_three_floor_stairwell_requires_two_separate_staircase_objects_sharing_a_landing_zone(self):

        # The correct Designer-authoring procedure for a 3+ floor
        # stairwell: one Staircase per adjacent floor pair, each ending
        # at the SAME physical landing Zone on the shared middle floor.
        building = Building(name="B")
        ground = building.create_floor(name="Ground", height=3.0)
        floor1 = building.create_floor(name="Floor 1", height=3.0)
        floor2 = building.create_floor(name="Floor 2", height=3.0)

        z_ground = make_zone("z-ground", ground.id)
        landing = make_zone("landing", floor1.id)  # the SAME zone both flights use
        z2 = make_zone("z2", floor2.id)

        ground.add_zone(z_ground)
        floor1.add_zone(landing)
        floor2.add_zone(z2)

        lower_flight = make_stair("LOWER-FLIGHT", ground, floor1, z_ground.id, landing.id)
        upper_flight = make_stair("UPPER-FLIGHT", floor1, floor2, landing.id, z2.id)

        graph = NavigationGraphGenerator().build(building)

        stair_edges = [e for e in graph.edges if e.edge_type == Edge.STAIR]
        self.assertEqual(len(stair_edges), 2)

        engine = PathfindingEngine(graph)
        route = engine.dijkstra(z2.id, z_ground.id)

        self.assertIsNotNone(route)
        self.assertEqual([e.id for e in route.edges], ["UPPER-FLIGHT", "LOWER-FLIGHT"])
        # The shared landing zone appears exactly once, as the pivot
        # node between the two flights -- never duplicated, never a
        # separate "in-between" node.
        self.assertEqual(route.node_ids.count(landing.id), 1)


# =====================================================
# Phase 6 -- stair directionality
# =====================================================


class StairDirectionalityTests(unittest.TestCase):

    def _two_floor_building(self):

        building = Building(name="B")
        ground = building.create_floor(name="Ground", height=3.0)
        floor1 = building.create_floor(name="Floor 1", height=3.0)

        lobby = make_zone("lobby", ground.id)
        upstairs = make_zone("upstairs", floor1.id)
        ground.add_zone(lobby)
        floor1.add_zone(upstairs)
        ground.add_exit(Exit(id="EXIT-G", name="Exit", zone_id=lobby.id, floor_id=ground.id))

        make_stair("S1", ground, floor1, lobby.id, upstairs.id)

        graph = NavigationGraphGenerator().build(building)
        return building, ground, floor1, lobby, upstairs, graph

    def test_both_directions_traversable(self):

        building, ground, floor1, lobby, upstairs, graph = self._two_floor_building()
        engine = PathfindingEngine(graph)

        up_route = engine.dijkstra(lobby.id, upstairs.id)
        down_route = engine.dijkstra(upstairs.id, lobby.id)

        self.assertIsNotNone(up_route)
        self.assertIsNotNone(down_route)
        self.assertEqual(up_route.edges[0].id, "S1")
        self.assertEqual(down_route.edges[0].id, "S1")

    def test_downward_evacuation_from_upper_floor_reaches_ground_exit(self):

        building, ground, floor1, lobby, upstairs, graph = self._two_floor_building()
        engine = PathfindingEngine(graph)
        sim = MultiAgentSimulation(engine)

        sim.add_occupant(upstairs.id, occupant_id="occ")
        result = sim.run()

        self.assertEqual(result.occupants["occ"].state, OccupantState.ARRIVED)
        self.assertEqual(result.occupants["occ"].steps[-1].edge.id, "EXIT-G")

    def test_from_floor_id_reversed_still_derives_correct_elevation(self):

        # A Staircase authored "backwards" (from_floor_id is the
        # HIGHER floor) -- direction must be derived from real
        # Building.floor_elevation(), never assumed from field naming.
        building = Building(name="B")
        ground = building.create_floor(name="Ground", height=3.0)
        floor1 = building.create_floor(name="Floor 1", height=4.0)

        lobby = make_zone("lobby", ground.id)
        upstairs = make_zone("upstairs", floor1.id)
        ground.add_zone(lobby)
        floor1.add_zone(upstairs)

        # from_floor_id is floor1 (the HIGHER floor) -- reversed vs.
        # the usual convention.
        stair = Staircase(
            id="S1", from_floor_id=floor1.id, to_floor_id=ground.id,
            from_zone_id=upstairs.id, to_zone_id=lobby.id, width=1.5,
        )
        floor1.add_stair(stair)

        self.assertEqual(building.floor_elevation(ground), 0.0)
        self.assertEqual(building.floor_elevation(floor1), 3.0)  # ground's own height
        self.assertGreater(stair.vertical_height(building), 0.0)

        graph = NavigationGraphGenerator().build(building)
        edge = next(e for e in graph.edges if e.edge_type == Edge.STAIR)
        self.assertGreater(edge.walking_distance, 0.0)


# =====================================================
# Phase 7 -- approachability
# =====================================================


class ApproachabilityTests(unittest.TestCase):

    def test_occupants_at_various_positions_all_reach_the_stair(self):

        building = Building(name="B")
        ground = building.create_floor(name="Ground", height=3.0)
        floor1 = building.create_floor(name="Floor 1", height=3.0)

        near_zone = make_zone("near", ground.id, x=0.0, y=0.0, width=5.0, height=5.0)
        far_zone = make_zone("far", ground.id, x=100.0, y=100.0, width=5.0, height=5.0)
        behind_door_zone = make_zone("behind-door", ground.id, x=200.0, y=0.0, width=5.0, height=5.0)
        upstairs = make_zone("upstairs", floor1.id)

        ground.add_zone(near_zone)
        ground.add_zone(far_zone)
        ground.add_zone(behind_door_zone)
        floor1.add_zone(upstairs)

        ground.add_door(Door(
            id="D-far", name="D-far", floor_id=ground.id, start_point=(50.0, 50.0), end_point=(55.0, 50.0),
            zone_a_id=near_zone.id, zone_b_id=far_zone.id,
        ))
        ground.add_door(Door(
            id="D-behind", name="D-behind", floor_id=ground.id, start_point=(100.0, 50.0), end_point=(105.0, 50.0),
            zone_a_id=far_zone.id, zone_b_id=behind_door_zone.id,
        ))

        make_stair("S1", ground, floor1, near_zone.id, upstairs.id)

        graph = NavigationGraphGenerator().build(building)
        engine = PathfindingEngine(graph)

        for zone_id in (near_zone.id, far_zone.id, behind_door_zone.id):
            with self.subTest(zone=zone_id):
                route = engine.dijkstra(zone_id, upstairs.id)
                self.assertIsNotNone(route, f"{zone_id} cannot reach the Stair")
                self.assertEqual(route.edges[-1].id, "S1")

    def test_stair_with_no_connectivity_to_a_floors_zone_is_unreachable_and_flagged(self):

        # If a Stair's own approach zone has no Door connecting it to
        # the rest of the floor's zones, the Stair genuinely cannot be
        # reached from those other zones -- a REAL failure, not a bug
        # in this milestone's own reasoning (Phase 7's own "if the
        # Stair exists but cannot be reached... treat that as a real
        # failure" instruction). This is a navigation-completeness
        # property of the whole floor, not something Stair-specific
        # code could fix without inventing connectivity that was never
        # authored.
        building = Building(name="B")
        ground = building.create_floor(name="Ground", height=3.0)
        floor1 = building.create_floor(name="Floor 1", height=3.0)

        isolated_zone = make_zone("isolated", ground.id)
        stair_approach = make_zone("stair-approach", ground.id, x=100.0, y=100.0)
        upstairs = make_zone("upstairs", floor1.id)

        ground.add_zone(isolated_zone)
        ground.add_zone(stair_approach)
        floor1.add_zone(upstairs)
        # Deliberately NO Door between isolated_zone and stair_approach.

        make_stair("S1", ground, floor1, stair_approach.id, upstairs.id)

        graph = NavigationGraphGenerator().build(building)
        engine = PathfindingEngine(graph)

        route = engine.dijkstra(isolated_zone.id, upstairs.id)
        self.assertIsNone(route)  # genuinely, honestly unreachable

        # But the Stair edge itself was still built correctly -- the
        # gap is in the FLOOR's own Zone connectivity, not the Stair.
        stair_edge = next(e for e in graph.edges if e.edge_type == Edge.STAIR)
        self.assertGreater(stair_edge.walking_distance, 0.0)
        route_from_approach = engine.dijkstra(stair_approach.id, upstairs.id)
        self.assertIsNotNone(route_from_approach)


# =====================================================
# Phase 8 -- multiple stairs
# =====================================================


class MultipleStairsTests(unittest.TestCase):

    def _building_with_two_stairs(self):

        building = Building(name="B")
        ground = building.create_floor(name="Ground", height=3.0)
        floor1 = building.create_floor(name="Floor 1", height=3.0)

        start = make_zone("start", ground.id, x=0.0, y=0.0)
        approach_a = make_zone("approach-a", ground.id, x=20.0, y=0.0)
        approach_b = make_zone("approach-b", ground.id, x=0.0, y=20.0)
        upstairs = make_zone("upstairs", floor1.id)

        for zone in (start, approach_a, approach_b):
            ground.add_zone(zone)
        floor1.add_zone(upstairs)

        door_a = Door(
            id="D-A", name="D-A", floor_id=ground.id, start_point=(10.0, 5.0), end_point=(20.0, 5.0),
            zone_a_id=start.id, zone_b_id=approach_a.id,
        )
        door_b = Door(
            id="D-B", name="D-B", floor_id=ground.id, start_point=(5.0, 10.0), end_point=(5.0, 20.0),
            zone_a_id=start.id, zone_b_id=approach_b.id,
        )
        ground.add_door(door_a)
        ground.add_door(door_b)

        make_stair("STAIR-A", ground, floor1, approach_a.id, upstairs.id)
        make_stair("STAIR-B", ground, floor1, approach_b.id, upstairs.id)

        return building, start, approach_a, approach_b, upstairs, door_a, door_b

    def test_both_stairs_reachable(self):

        building, start, approach_a, approach_b, upstairs, door_a, door_b = self._building_with_two_stairs()
        graph = NavigationGraphGenerator().build(building)
        engine = PathfindingEngine(graph)

        route_via_a = engine.dijkstra(approach_a.id, upstairs.id)
        route_via_b = engine.dijkstra(approach_b.id, upstairs.id)

        self.assertEqual(route_via_a.edges[-1].id, "STAIR-A")
        self.assertEqual(route_via_b.edges[-1].id, "STAIR-B")

    def test_blocking_route_to_stair_a_migrates_to_stair_b(self):

        building, start, approach_a, approach_b, upstairs, door_a, door_b = self._building_with_two_stairs()

        graph = NavigationGraphGenerator().build(building)
        engine = PathfindingEngine(graph)

        route_before = engine.dijkstra(start.id, upstairs.id)
        stair_used_before = route_before.edges[-1].id
        self.assertEqual(stair_used_before, "STAIR-A")

        # Block Door A's approach with an obstacle across its segment.
        ground_floor = building.get_floor(start.floor_id)
        ground_floor.obstacles.append(
            Obstacle(id="O1", name="O1", floor_id=ground_floor.id, x=9.0, y=4.0, length=2.0, width=2.0, active=True, traversability="Blocked"),
        )

        graph_after = NavigationGraphGenerator().build(building)
        engine_after = PathfindingEngine(graph_after)

        route_after = engine_after.dijkstra(start.id, upstairs.id)

        self.assertIsNotNone(route_after)
        self.assertEqual(route_after.edges[-1].id, "STAIR-B")
        self.assertNotEqual(stair_used_before, route_after.edges[-1].id)

    def test_occupants_not_permanently_tied_to_one_stair(self):

        # Two independent MultiAgentSimulation runs against the SAME
        # building, differing only in whether Door A is blocked --
        # proves an occupant is never hard-wired to a specific Stair id.
        building, start, approach_a, approach_b, upstairs, door_a, door_b = self._building_with_two_stairs()

        graph = NavigationGraphGenerator().build(building)
        engine = PathfindingEngine(graph)
        sim = MultiAgentSimulation(engine)
        sim.add_occupant(start.id, upstairs.id, occupant_id="occ")
        result = sim.run()
        first_choice = next(s.edge.id for s in result.occupants["occ"].steps if s.edge.edge_type == Edge.STAIR)

        ground_floor = building.get_floor(start.floor_id)
        ground_floor.obstacles.append(
            Obstacle(id="O1", name="O1", floor_id=ground_floor.id, x=9.0, y=4.0, length=2.0, width=2.0, active=True, traversability="Blocked"),
        )

        graph_2 = NavigationGraphGenerator().build(building)
        engine_2 = PathfindingEngine(graph_2)
        sim_2 = MultiAgentSimulation(engine_2)
        sim_2.add_occupant(start.id, upstairs.id, occupant_id="occ")
        result_2 = sim_2.run()
        second_choice = next(s.edge.id for s in result_2.occupants["occ"].steps if s.edge.edge_type == Edge.STAIR)

        self.assertEqual(first_choice, "STAIR-A")
        self.assertEqual(second_choice, "STAIR-B")


# =====================================================
# Phase 9 -- blocked stair approach (existing Obstacle behavior only)
# =====================================================


class BlockedStairApproachTests(unittest.TestCase):

    def test_obstacle_blocks_stair_a_approach_new_route_uses_stair_b(self):

        building = Building(name="B")
        ground = building.create_floor(name="Ground", height=3.0)
        floor1 = building.create_floor(name="Floor 1", height=3.0)

        start = make_zone("start", ground.id, x=0.0, y=0.0)
        approach_a = make_zone("approach-a", ground.id, x=20.0, y=0.0)
        approach_b = make_zone("approach-b", ground.id, x=0.0, y=20.0)
        upstairs = make_zone("upstairs", floor1.id)

        for zone in (start, approach_a, approach_b):
            ground.add_zone(zone)
        floor1.add_zone(upstairs)

        door_a = Door(
            id="D-A", name="D-A", floor_id=ground.id, start_point=(10.0, 5.0), end_point=(20.0, 5.0),
            zone_a_id=start.id, zone_b_id=approach_a.id,
        )
        ground.add_door(door_a)
        ground.add_door(Door(
            id="D-B", name="D-B", floor_id=ground.id, start_point=(5.0, 10.0), end_point=(5.0, 20.0),
            zone_a_id=start.id, zone_b_id=approach_b.id,
        ))

        make_stair("STAIR-A", ground, floor1, approach_a.id, upstairs.id)
        make_stair("STAIR-B", ground, floor1, approach_b.id, upstairs.id)

        graph = NavigationGraphGenerator().build(building)
        engine = PathfindingEngine(graph)
        sim = MultiAgentSimulation(engine)
        sim.add_occupant(start.id, upstairs.id, occupant_id="before")
        result_before = sim.run()
        self.assertEqual(
            next(s.edge.id for s in result_before.occupants["before"].steps if s.edge.edge_type == Edge.STAIR),
            "STAIR-A",
        )

        # Obstacle activates, directly across Door A's own segment.
        ground.obstacles.append(
            Obstacle(id="OBS-1", name="Blockage", floor_id=ground.id, x=9.0, y=4.0, length=2.0, width=2.0, active=True, traversability="Blocked"),
        )

        graph_2 = NavigationGraphGenerator().build(building)
        door_a_edge = next(e for e in graph_2.edges if e.id == "D-A")
        self.assertFalse(door_a_edge.traversable)  # Stair A's own approach is now unreachable via D-A

        engine_2 = PathfindingEngine(graph_2)
        sim_2 = MultiAgentSimulation(engine_2)
        sim_2.add_occupant(start.id, upstairs.id, occupant_id="after")
        result_after = sim_2.run()

        self.assertEqual(result_after.occupants["after"].state, OccupantState.ARRIVED)
        self.assertEqual(
            next(s.edge.id for s in result_after.occupants["after"].steps if s.edge.edge_type == Edge.STAIR),
            "STAIR-B",
        )

        # Only Stair A's approach is affected -- the rest of the floor
        # (Door B / Stair B) stays completely usable, never globally
        # disabled.
        door_b_edge = next(e for e in graph_2.edges if e.id == "D-B")
        self.assertTrue(door_b_edge.traversable)


# =====================================================
# Phase 10 -- stair capacity & congestion
# =====================================================


class StairCapacityCongestionTests(unittest.TestCase):

    def _building(self, width=0.5):

        building = Building(name="B")
        ground = building.create_floor(name="Ground", height=3.0)
        floor1 = building.create_floor(name="Floor 1", height=3.0)

        lobby = make_zone("lobby", ground.id)
        upstairs = make_zone("upstairs", floor1.id)
        ground.add_zone(lobby)
        floor1.add_zone(upstairs)

        stair = make_stair("S1", ground, floor1, lobby.id, upstairs.id, width=width)
        return building, lobby, upstairs, stair

    def _run_n_occupants(self, n, width=0.5):

        building, lobby, upstairs, stair = self._building(width=width)
        graph = NavigationGraphGenerator().build(building)
        engine = PathfindingEngine(graph)

        sim = MultiAgentSimulation(engine, capacity_model=StairCapacityModel(), congestion_model=StairAwareCongestionModel())
        for i in range(n):
            sim.add_occupant(lobby.id, upstairs.id, occupant_id=f"occ-{i}")

        return sim.run()

    def test_capacity_1_stair_intentional_meaning(self):

        # A narrow (0.5m) stair floors to capacity 1 -- confirmed
        # intentional: DefaultCapacityModel/StairCapacityModel both
        # document MINIMUM_CAPACITY=1 specifically so the event queue
        # can never deadlock. capacity=1 means "one occupant may
        # physically be on this edge at a time," NOT "only one
        # occupant may ever use this stair."
        building, lobby, upstairs, stair = self._building(width=0.5)
        graph = NavigationGraphGenerator().build(building)
        edge = next(e for e in graph.edges if e.edge_type == Edge.STAIR)
        self.assertEqual(StairCapacityModel().capacity(edge), 1)

    def test_1_2_10_50_occupants_all_eventually_evacuate_no_disappearance(self):

        for n in (1, 2, 10, 50):
            with self.subTest(occupant_count=n):

                result = self._run_n_occupants(n, width=0.5)

                self.assertEqual(len(result.occupants), n)  # nobody disappeared

                for occupant_id, timeline in result.occupants.items():
                    self.assertEqual(timeline.state, OccupantState.ARRIVED, f"{occupant_id} did not evacuate")
                    for step in timeline.steps:
                        self.assertGreater(step.end_time, step.start_time, f"{occupant_id}: instantaneous step")

    def test_queueing_behaves_correctly_under_capacity_1(self):

        result = self._run_n_occupants(10, width=0.5)

        stair_steps = sorted(
            (step for timeline in result.occupants.values() for step in timeline.steps if step.edge.edge_type == Edge.STAIR),
            key=lambda s: s.start_time,
        )

        # Capacity 1 -- no two occupants may overlap on the stair edge
        # at once.
        for a, b in zip(stair_steps, stair_steps[1:]):
            self.assertLessEqual(a.end_time, b.start_time)

        self.assertGreater(result.total_queue_events, 0)
        self.assertGreaterEqual(result.peak_edge_occupancy.get("S1", 0), 1)
        self.assertLessEqual(result.peak_edge_occupancy.get("S1", 0), 1)

    def test_capacity_never_makes_stair_permanently_unreachable(self):

        # NAVIGATION UNREACHABLE (no route exists at all) is a
        # different, structural condition from WAITING FOR CAPACITY
        # (a route exists, and the occupant eventually gets to use it).
        # 50 occupants through a capacity-1 stair -- every single one
        # still reaches ARRIVED, proving capacity constrains THROUGHPUT
        # (queueing), never REACHABILITY.
        result = self._run_n_occupants(50, width=0.5)

        self.assertEqual(len(result.unreachable_occupant_ids), 0)
        self.assertTrue(all(t.state == OccupantState.ARRIVED for t in result.occupants.values()))


# =====================================================
# Phase 16 -- explicit failure cases
# =====================================================


class StairFailureCasesTests(unittest.TestCase):

    def _base_building(self):

        building = Building(name="B")
        ground = building.create_floor(name="Ground", height=3.0)
        floor1 = building.create_floor(name="Floor 1", height=3.0)
        lobby = make_zone("lobby", ground.id)
        upstairs = make_zone("upstairs", floor1.id)
        ground.add_zone(lobby)
        floor1.add_zone(upstairs)
        return building, ground, floor1, lobby, upstairs

    def test_missing_from_floor_id(self):

        building, ground, floor1, lobby, upstairs = self._base_building()
        stair = Staircase(id="S1", from_floor_id="", to_floor_id=floor1.id, from_zone_id=lobby.id, to_zone_id=upstairs.id, width=1.5)
        ground.add_stair(stair)

        graph = NavigationGraphGenerator().build(building)
        edge = next(e for e in graph.edges if e.edge_type == Edge.STAIR)

        self.assertIsNone(edge.walking_distance)  # never fabricated, degraded conservatively
        self.assertTrue(edge.traversable)  # connectivity unaffected
        self.assertEqual(len(graph.validate().by_code("stair_zero_traversal_distance")), 1)

    def test_missing_to_floor_id(self):

        building, ground, floor1, lobby, upstairs = self._base_building()
        stair = Staircase(id="S1", from_floor_id=ground.id, to_floor_id="", from_zone_id=lobby.id, to_zone_id=upstairs.id, width=1.5)
        ground.add_stair(stair)

        graph = NavigationGraphGenerator().build(building)

        # No edge at all -- an unresolvable destination floor is a
        # harder failure than a distance-computation gap (existing,
        # pre-milestone behavior, unchanged).
        self.assertEqual([e for e in graph.edges if e.edge_type == Edge.STAIR], [])
        self.assertGreater(len(graph.validate().by_code("stair_missing_destination_floor")), 0)

    def test_same_from_and_to_floor(self):

        building, ground, floor1, lobby, upstairs = self._base_building()
        other_zone = make_zone("other", ground.id, x=50.0, y=50.0)
        ground.add_zone(other_zone)

        stair = Staircase(id="S1", from_floor_id=ground.id, to_floor_id=ground.id, from_zone_id=lobby.id, to_zone_id=other_zone.id, width=1.5)
        ground.add_stair(stair)

        graph = NavigationGraphGenerator().build(building)

        self.assertEqual(len(graph.validate().by_code("stair_same_floor_both_ends")), 1)
        self.assertEqual(len(graph.validate().by_code("stair_zero_traversal_distance")), 1)

        edge = next(e for e in graph.edges if e.edge_type == Edge.STAIR)
        self.assertIsNone(edge.walking_distance)

    def test_deleted_floor(self):

        building, ground, floor1, lobby, upstairs = self._base_building()
        stair = make_stair("S1", ground, floor1, lobby.id, upstairs.id)

        building.remove_floor(floor1)

        graph = NavigationGraphGenerator().build(building)
        self.assertEqual([e for e in graph.edges if e.edge_type == Edge.STAIR], [])
        self.assertGreater(len(graph.validate().by_code("invalid_reference")), 0)

    def test_stair_has_no_active_or_locked_concept_in_v1(self):

        # A genuine finding, not a bug: unlike Door (active/locked) and
        # Exit (is_blocked), Staircase carries no inactive/disabled
        # flag at all in V1 -- Edge.traversable always returns True for
        # a Stair edge that was built at all (see navigation/edge.py).
        stair_fields = {f.name for f in Staircase.__dataclass_fields__.values()}
        self.assertNotIn("active", stair_fields)
        self.assertNotIn("locked", stair_fields)

        building, ground, floor1, lobby, upstairs = self._base_building()
        make_stair("S1", ground, floor1, lobby.id, upstairs.id)
        graph = NavigationGraphGenerator().build(building)
        edge = next(e for e in graph.edges if e.edge_type == Edge.STAIR)
        self.assertTrue(edge.traversable)

    def test_disconnected_stair_approach(self):

        building, ground, floor1, lobby, upstairs = self._base_building()
        isolated = make_zone("isolated", ground.id, x=500.0, y=500.0)
        ground.add_zone(isolated)
        make_stair("S1", ground, floor1, lobby.id, upstairs.id)

        graph = NavigationGraphGenerator().build(building)
        engine = PathfindingEngine(graph)
        route = engine.dijkstra(isolated.id, upstairs.id)

        self.assertIsNone(route)

    def test_one_of_multiple_stairs_unavailable_others_remain_usable(self):

        building, ground, floor1, lobby, upstairs = self._base_building()
        approach_b = make_zone("approach-b", ground.id, x=50.0, y=50.0)
        ground.add_zone(approach_b)
        ground.add_door(Door(
            id="D-B", name="D-B", floor_id=ground.id, start_point=(20.0, 20.0), end_point=(25.0, 20.0),
            zone_a_id=lobby.id, zone_b_id=approach_b.id,
        ))

        make_stair("STAIR-A", ground, floor1, lobby.id, upstairs.id)
        make_stair("STAIR-B", ground, floor1, approach_b.id, upstairs.id)

        # STAIR-A is malformed (missing to_floor_id) -- no edge built.
        broken = Staircase(id="STAIR-BROKEN", from_floor_id=ground.id, to_floor_id="", from_zone_id=lobby.id, to_zone_id=upstairs.id)
        ground.add_stair(broken)

        graph = NavigationGraphGenerator().build(building)
        engine = PathfindingEngine(graph)

        route = engine.dijkstra(lobby.id, upstairs.id)
        self.assertIsNotNone(route)
        self.assertEqual(route.edges[-1].id, "STAIR-A")  # the healthy one is still preferred/usable


# =====================================================
# Phase 19 -- end-to-end evacuation
# =====================================================


class EndToEndEvacuationTests(unittest.TestCase):

    def test_realistic_multi_floor_evacuation_zero_stranded(self):

        building = Building(name="E2E Building")
        ground = building.create_floor(name="Ground", height=3.0)
        floor1 = building.create_floor(name="Floor 1", height=3.0)
        floor2 = building.create_floor(name="Floor 2", height=3.0)

        lobby = make_zone("lobby", ground.id, x=0.0, y=0.0, width=10.0, height=10.0)
        corridor_a = make_zone("corridor-a", ground.id, x=20.0, y=0.0)
        corridor_b = make_zone("corridor-b", ground.id, x=0.0, y=20.0)

        office_1a = make_zone("office-1a", floor1.id, x=0.0, y=0.0)
        office_1b = make_zone("office-1b", floor1.id, x=20.0, y=0.0)

        office_2a = make_zone("office-2a", floor2.id, x=0.0, y=0.0)
        office_2b = make_zone("office-2b", floor2.id, x=20.0, y=0.0)

        for zone in (lobby, corridor_a, corridor_b):
            ground.add_zone(zone)
        for zone in (office_1a, office_1b):
            floor1.add_zone(zone)
        for zone in (office_2a, office_2b):
            floor2.add_zone(zone)

        ground.add_door(Door(id="D-A", name="D-A", floor_id=ground.id, start_point=(10.0, 5.0), end_point=(20.0, 5.0), zone_a_id=lobby.id, zone_b_id=corridor_a.id))
        ground.add_door(Door(id="D-B", name="D-B", floor_id=ground.id, start_point=(5.0, 10.0), end_point=(5.0, 20.0), zone_a_id=lobby.id, zone_b_id=corridor_b.id))

        floor1.add_door(Door(id="D-1", name="D-1", floor_id=floor1.id, start_point=(10.0, 5.0), end_point=(20.0, 5.0), zone_a_id=office_1a.id, zone_b_id=office_1b.id))
        floor2.add_door(Door(id="D-2", name="D-2", floor_id=floor2.id, start_point=(10.0, 5.0), end_point=(20.0, 5.0), zone_a_id=office_2a.id, zone_b_id=office_2b.id))

        ground.add_exit(Exit(id="EXIT-A", name="Exit A", zone_id=corridor_a.id, floor_id=ground.id, capacity=2))
        ground.add_exit(Exit(id="EXIT-B", name="Exit B", zone_id=corridor_b.id, floor_id=ground.id, capacity=2))

        make_stair("STAIR-1A", ground, floor1, corridor_a.id, office_1a.id, width=1.2)
        make_stair("STAIR-2A", floor1, floor2, office_1a.id, office_2a.id, width=1.2)

        graph = NavigationGraphGenerator().build(building)

        self.assertEqual(graph.validate().by_code("stair_zero_traversal_distance"), [])

        engine = PathfindingEngine(graph)
        sim = MultiAgentSimulation(engine, capacity_model=StairCapacityModel(), congestion_model=StairAwareCongestionModel())

        occupant_specs = (
            (lobby.id, "g1"), (corridor_a.id, "g2"), (corridor_b.id, "g3"),
            (office_1a.id, "f1a-1"), (office_1a.id, "f1a-2"), (office_1b.id, "f1b-1"),
            (office_2a.id, "f2a-1"), (office_2a.id, "f2a-2"), (office_2b.id, "f2b-1"), (office_2b.id, "f2b-2"),
        )
        for zone_id, occupant_id in occupant_specs:
            sim.add_occupant(zone_id, occupant_id=occupant_id)

        # One blocked route: block Door D-B (corridor_b's own approach).
        ground.obstacles.append(
            Obstacle(id="OBS-1", name="Blockage", floor_id=ground.id, x=4.0, y=9.0, length=2.0, width=2.0, active=True, traversability="Blocked"),
        )
        graph_blocked = NavigationGraphGenerator().build(building)
        engine_blocked = PathfindingEngine(graph_blocked)
        sim_blocked = MultiAgentSimulation(engine_blocked, capacity_model=StairCapacityModel(), congestion_model=StairAwareCongestionModel())
        for zone_id, occupant_id in occupant_specs:
            sim_blocked.add_occupant(zone_id, occupant_id=occupant_id)

        result = sim_blocked.run()

        evacuated = sum(1 for t in result.occupants.values() if t.state == OccupantState.ARRIVED)
        stranded = sum(1 for t in result.occupants.values() if t.state != OccupantState.ARRIVED)

        max_stair_queue = max(
            (result.peak_edge_occupancy.get(sid, 0) for sid in ("STAIR-1A", "STAIR-2A")), default=0,
        )

        print(
            f"\n[E2E evacuation] initial_occupants={len(occupant_specs)} "
            f"evacuated={evacuated} stranded={stranded} "
            f"max_stair_queue={max_stair_queue} "
            f"total_evacuation_time={result.total_evacuation_time}"
        )

        self.assertEqual(len(occupant_specs), 10)
        self.assertEqual(stranded, 0)
        self.assertEqual(evacuated, 10)
        self.assertEqual(len(result.unreachable_occupant_ids), 0)
        self.assertIsNotNone(result.total_evacuation_time)


if __name__ == "__main__":
    unittest.main()
