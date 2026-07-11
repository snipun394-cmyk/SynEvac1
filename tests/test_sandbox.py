import unittest

from models.assembly_point import AssemblyPoint
from models.building import Building
from models.door import Door
from models.exit import Exit
from models.staircase import Staircase
from models.zone import Zone

from navigation.edge import Edge

from sandbox.manager import SandboxManager
from sandbox.occupant import SandboxDestinationType, SandboxDistribution, SandboxOccupantState


def make_zone(name, **kwargs):

    fields = dict(x=0.0, y=0.0, width=2.0, height=2.0)
    fields.update(kwargs)

    return Zone(name=name, **fields)


def build_sandbox_building():

    # Room --D1--> Corridor --D2--> Lobby --Exit--> Outside
    #                                  \--D3--> AssemblyPoint AP1
    # Room --Stair--> Upstairs (Floor 1)
    # Isolated: a zone with no connections at all.

    building = Building(name="B")
    ground = building.create_floor(name="Ground Floor")
    floor1 = building.create_floor(name="Floor 1")

    room = make_zone("Room", x=0.0, y=0.0)
    corridor = make_zone("Corridor", x=10.0, y=0.0)
    lobby = make_zone("Lobby", x=20.0, y=0.0)
    isolated = make_zone("Isolated", x=40.0, y=40.0)

    ground.add_zone(room)
    ground.add_zone(corridor)
    ground.add_zone(lobby)
    ground.add_zone(isolated)

    upstairs = make_zone("Upstairs", x=0.0, y=0.0)
    floor1.add_zone(upstairs)

    door1 = Door(name="D1", zone_a_id=room.id, zone_b_id=corridor.id, floor_id=ground.id)
    door2 = Door(name="D2", zone_a_id=corridor.id, zone_b_id=lobby.id, floor_id=ground.id)
    ground.add_door(door1)
    ground.add_door(door2)

    exit_obj = Exit(name="Ex", zone_id=lobby.id, floor_id=ground.id)
    ground.add_exit(exit_obj)

    assembly_point = AssemblyPoint(name="AP1", position=(30.0, 10.0), floor_id=ground.id)
    ground.add_assembly_point(assembly_point)

    door3 = Door(name="D3", zone_a_id=lobby.id, zone_b_id=assembly_point.id, floor_id=ground.id)
    ground.add_door(door3)

    stair = Staircase(
        name="S1", from_floor_id=ground.id, to_floor_id=floor1.id,
        from_zone_id=room.id, to_zone_id=upstairs.id,
    )
    ground.add_stair(stair)

    return {
        "building": building,
        "ground": ground,
        "floor1": floor1,
        "room": room,
        "corridor": corridor,
        "lobby": lobby,
        "isolated": isolated,
        "upstairs": upstairs,
        "assembly_point": assembly_point,
        "door1": door1,
        "door2": door2,
        "door3": door3,
        "exit": exit_obj,
        "stair": stair,
    }


class PlacementTests(unittest.TestCase):

    def setUp(self):
        self.world = build_sandbox_building()
        self.manager = SandboxManager()

    def test_place_occupant_resolves_correct_zone_and_floor(self):

        room = self.world["room"]
        occupant = self.manager.place_occupant(self.world["ground"], x_m=1.0, y_m=1.0)

        self.assertIsNotNone(occupant)
        self.assertEqual(occupant.zone_id, room.id)
        self.assertEqual(occupant.floor_id, self.world["ground"].id)
        self.assertEqual(occupant.position, (1.0, 1.0))

    def test_place_occupant_outside_any_zone_returns_none(self):

        occupant = self.manager.place_occupant(self.world["ground"], x_m=999.0, y_m=999.0)

        self.assertIsNone(occupant)
        self.assertEqual(self.manager.occupants, [])

    def test_occupant_naming_is_sequential(self):

        first = self.manager.place_occupant(self.world["ground"], x_m=1.0, y_m=1.0)
        second = self.manager.place_occupant(self.world["ground"], x_m=1.0, y_m=1.0)

        self.assertEqual(first.name, "Occupant 1")
        self.assertEqual(second.name, "Occupant 2")

    def test_placed_occupant_is_tracked_and_removable(self):

        occupant = self.manager.place_occupant(self.world["ground"], x_m=1.0, y_m=1.0)

        self.assertIn(occupant, self.manager.occupants)

        self.manager.remove_occupant(occupant)

        self.assertNotIn(occupant, self.manager.occupants)

    def test_clear_resets_occupants_and_naming_counter(self):

        self.manager.place_occupant(self.world["ground"], x_m=1.0, y_m=1.0)
        self.manager.clear()

        self.assertEqual(self.manager.occupants, [])

        occupant = self.manager.place_occupant(self.world["ground"], x_m=1.0, y_m=1.0)
        self.assertEqual(occupant.name, "Occupant 1")


class GenerationTests(unittest.TestCase):

    # Room is a 2m x 2m Zone at (0,0)-(2,2) -- a rectangle covering it
    # exactly should let generation reach `count` for both
    # distributions with no zone-boundary edge cases to worry about.

    def setUp(self):
        self.world = build_sandbox_building()
        self.manager = SandboxManager()

    def test_uniform_generates_exactly_the_requested_count(self):

        occupants = self.manager.generate_occupants(
            self.world["ground"], 0.0, 0.0, 2.0, 2.0, 5, SandboxDistribution.UNIFORM,
        )

        self.assertEqual(len(occupants), 5)

    def test_random_generates_exactly_the_requested_count(self):

        occupants = self.manager.generate_occupants(
            self.world["ground"], 0.0, 0.0, 2.0, 2.0, 5, SandboxDistribution.RANDOM,
        )

        self.assertEqual(len(occupants), 5)

    def test_generated_occupants_are_assigned_to_the_correct_zone_and_floor(self):

        occupants = self.manager.generate_occupants(
            self.world["ground"], 0.0, 0.0, 2.0, 2.0, 4, SandboxDistribution.UNIFORM,
        )

        for occupant in occupants:
            self.assertEqual(occupant.zone_id, self.world["room"].id)
            self.assertEqual(occupant.floor_id, self.world["ground"].id)

    def test_generated_occupants_are_tracked_by_the_manager(self):

        self.manager.generate_occupants(
            self.world["ground"], 0.0, 0.0, 2.0, 2.0, 3, SandboxDistribution.UNIFORM,
        )

        self.assertEqual(len(self.manager.occupants), 3)

    def test_generated_occupants_have_sequential_names(self):

        occupants = self.manager.generate_occupants(
            self.world["ground"], 0.0, 0.0, 2.0, 2.0, 3, SandboxDistribution.UNIFORM,
        )

        self.assertEqual(
            [o.name for o in occupants],
            ["Occupant 1", "Occupant 2", "Occupant 3"],
        )

    def test_uniform_positions_are_spread_out_not_stacked(self):

        occupants = self.manager.generate_occupants(
            self.world["ground"], 0.0, 0.0, 2.0, 2.0, 4, SandboxDistribution.UNIFORM,
        )

        positions = {o.position for o in occupants}
        self.assertEqual(len(positions), 4)

    def test_positions_stay_distinct_after_routes_are_computed(self):

        # Regression: compute_route() used to always re-derive
        # position from the destination Zone's single shared center,
        # collapsing every occupant generated in the same Zone onto
        # one identical point the instant a route existed -- even
        # though generate_occupants() itself produced distinct
        # positions. This is the actual, full workflow (generate,
        # then immediately compute a route, exactly what
        # GraphicsScene._generate_occupants_in_rectangle does): the
        # distinct positions must survive route computation.
        occupants = self.manager.generate_occupants(
            self.world["ground"], 0.0, 0.0, 2.0, 2.0, 4, SandboxDistribution.UNIFORM,
        )

        for occupant in occupants:
            self.manager.compute_route(
                occupant, self.world["building"], SandboxDestinationType.EXIT,
            )

        positions = {o.position for o in occupants}
        self.assertEqual(len(positions), 4)

    def test_rectangle_can_be_given_in_either_corner_order(self):

        forward = self.manager.generate_occupants(
            self.world["ground"], 0.0, 0.0, 2.0, 2.0, 4, SandboxDistribution.UNIFORM,
        )

        self.manager.clear()

        reversed_corners = self.manager.generate_occupants(
            self.world["ground"], 2.0, 2.0, 0.0, 0.0, 4, SandboxDistribution.UNIFORM,
        )

        self.assertEqual(len(forward), len(reversed_corners))

    def test_rectangle_entirely_outside_any_zone_generates_nothing(self):

        occupants = self.manager.generate_occupants(
            self.world["ground"], 100.0, 100.0, 102.0, 102.0, 5, SandboxDistribution.UNIFORM,
        )

        self.assertEqual(occupants, [])
        self.assertEqual(self.manager.occupants, [])

    def test_random_distribution_never_hangs_on_a_rectangle_outside_any_zone(self):

        # Bounded by MAX_RANDOM_ATTEMPTS_PER_OCCUPANT -- must return
        # promptly with fewer (here, zero) than requested rather than
        # loop forever looking for a valid point that will never exist.
        occupants = self.manager.generate_occupants(
            self.world["ground"], 100.0, 100.0, 102.0, 102.0, 5, SandboxDistribution.RANDOM,
        )

        self.assertEqual(occupants, [])

    def test_zero_count_generates_nothing(self):

        occupants = self.manager.generate_occupants(
            self.world["ground"], 0.0, 0.0, 2.0, 2.0, 0, SandboxDistribution.UNIFORM,
        )

        self.assertEqual(occupants, [])

    def test_rectangle_spanning_two_zones_assigns_each_occupant_its_own_zone(self):

        # Room (0,0)-(2,2) and Corridor (10,0)-(12,2) are far apart in
        # this fixture -- use a rectangle over just Room plus a
        # separate one over just Corridor instead of one giant
        # rectangle spanning the gap between them (which would mostly
        # generate points in neither Zone). Confirms generation
        # correctly attributes each batch to its own Zone.
        room_occupants = self.manager.generate_occupants(
            self.world["ground"], 0.0, 0.0, 2.0, 2.0, 3, SandboxDistribution.UNIFORM,
        )
        corridor_occupants = self.manager.generate_occupants(
            self.world["ground"], 10.0, 0.0, 12.0, 2.0, 3, SandboxDistribution.UNIFORM,
        )

        self.assertTrue(all(o.zone_id == self.world["room"].id for o in room_occupants))
        self.assertTrue(all(o.zone_id == self.world["corridor"].id for o in corridor_occupants))


class RoutingTests(unittest.TestCase):

    def setUp(self):
        self.world = build_sandbox_building()
        self.manager = SandboxManager()
        self.occupant = self.manager.place_occupant(self.world["ground"], x_m=1.0, y_m=1.0)

    def test_route_to_exit_traverses_expected_doors_and_reaches_outside(self):

        route = self.manager.compute_route(
            self.occupant, self.world["building"], SandboxDestinationType.EXIT,
        )

        self.assertIsNotNone(route)
        self.assertEqual(
            route.edge_ids,
            [self.world["door1"].id, self.world["door2"].id, self.world["exit"].id],
        )
        self.assertEqual(route.nodes[-1].id, "outside")
        self.assertEqual(self.occupant.state, SandboxOccupantState.ROUTED)
        self.assertEqual(self.occupant.destination_node_id, "outside")

    def test_route_to_assembly_point(self):

        route = self.manager.compute_route(
            self.occupant, self.world["building"], SandboxDestinationType.ASSEMBLY_POINT,
        )

        self.assertIsNotNone(route)
        self.assertEqual(route.nodes[-1].id, self.world["assembly_point"].id)
        self.assertEqual(
            route.edge_ids,
            [self.world["door1"].id, self.world["door2"].id, self.world["door3"].id],
        )

    def test_multi_floor_route_via_stair(self):

        upstairs_occupant = self.manager.place_occupant(self.world["floor1"], x_m=1.0, y_m=1.0)

        route = self.manager.compute_route(
            upstairs_occupant, self.world["building"], SandboxDestinationType.EXIT,
        )

        self.assertIsNotNone(route)
        self.assertIn(self.world["stair"].id, route.edge_ids)

        floor_ids = [node.floor_id for node in route.nodes if node.floor_id]
        self.assertIn(self.world["floor1"].id, floor_ids)
        self.assertIn(self.world["ground"].id, floor_ids)

    def test_unreachable_zone_produces_no_route(self):

        stuck_occupant = self.manager.place_occupant(self.world["ground"], x_m=41.0, y_m=41.0)
        self.assertEqual(stuck_occupant.zone_id, self.world["isolated"].id)

        route = self.manager.compute_route(
            stuck_occupant, self.world["building"], SandboxDestinationType.EXIT,
        )

        self.assertIsNone(route)
        self.assertEqual(stuck_occupant.state, SandboxOccupantState.UNREACHABLE)

    def test_step_and_tick_are_no_ops_when_unreachable(self):

        stuck_occupant = self.manager.place_occupant(self.world["ground"], x_m=41.0, y_m=41.0)
        self.manager.compute_route(stuck_occupant, self.world["building"], SandboxDestinationType.EXIT)

        self.manager.step(stuck_occupant)
        self.manager.tick(stuck_occupant, 10.0)

        self.assertEqual(stuck_occupant.state, SandboxOccupantState.UNREACHABLE)
        self.assertEqual(stuck_occupant.node_index, 0)


class MovementTests(unittest.TestCase):

    def setUp(self):
        self.world = build_sandbox_building()
        self.manager = SandboxManager()
        self.occupant = self.manager.place_occupant(self.world["ground"], x_m=1.0, y_m=1.0)
        self.manager.compute_route(self.occupant, self.world["building"], SandboxDestinationType.EXIT)

    def test_step_mode_advances_exactly_one_node_per_call(self):

        self.assertEqual(self.occupant.node_index, 0)
        self.assertEqual(self.occupant.current_node_id, self.world["room"].id)
        self.assertEqual(self.occupant.next_node_id, self.world["corridor"].id)

        self.manager.step(self.occupant)

        self.assertEqual(self.occupant.node_index, 1)
        self.assertEqual(self.occupant.current_node_id, self.world["corridor"].id)
        self.assertEqual(self.occupant.state, SandboxOccupantState.MOVING)

    def test_stepping_through_the_whole_route_arrives(self):

        for _ in range(len(self.occupant.route.edges)):
            self.manager.step(self.occupant)

        self.assertEqual(self.occupant.state, SandboxOccupantState.ARRIVED)
        self.assertEqual(self.occupant.current_node_id, "outside")

    def test_further_steps_after_arrival_are_no_ops(self):

        for _ in range(len(self.occupant.route.edges) + 3):
            self.manager.step(self.occupant)

        self.assertEqual(self.occupant.state, SandboxOccupantState.ARRIVED)
        self.assertEqual(self.occupant.node_index, len(self.occupant.route.nodes) - 1)

    def test_small_tick_only_advances_edge_progress(self):

        edge = self.occupant.route.edges[0]
        distance = edge.walking_distance or Edge.DEFAULT_TRAVERSAL_COST

        half_time = (distance / 2.0) / self.occupant.speed

        self.manager.tick(self.occupant, half_time)

        self.assertEqual(self.occupant.node_index, 0)
        self.assertAlmostEqual(self.occupant.edge_progress, 0.5, places=3)
        self.assertEqual(self.occupant.state, SandboxOccupantState.MOVING)

    def test_large_tick_reaches_arrival(self):

        self.manager.tick(self.occupant, 10_000.0)

        self.assertEqual(self.occupant.state, SandboxOccupantState.ARRIVED)
        self.assertEqual(self.occupant.remaining_distance, 0.0)

    def test_current_speed_is_zero_until_movement_starts(self):

        self.assertEqual(self.occupant.current_speed, 0.0)

        self.manager.step(self.occupant)

        self.assertEqual(self.occupant.current_speed, self.occupant.speed)

    def test_remaining_distance_decreases_as_occupant_advances(self):

        initial = self.occupant.remaining_distance

        self.manager.step(self.occupant)

        self.assertLess(self.occupant.remaining_distance, initial)

    def test_position_updates_after_step(self):

        initial_position = self.occupant.position

        self.manager.step(self.occupant)

        self.assertNotEqual(self.occupant.position, initial_position)


class ResetTests(unittest.TestCase):

    def setUp(self):
        self.world = build_sandbox_building()
        self.manager = SandboxManager()

    def test_reset_restores_position_and_route_progress_on_the_same_floor(self):

        occupant = self.manager.place_occupant(self.world["ground"], x_m=1.0, y_m=1.0)
        self.manager.compute_route(occupant, self.world["building"], SandboxDestinationType.EXIT)

        self.manager.step(occupant)
        self.assertNotEqual(occupant.node_index, 0)

        self.manager.reset_occupant(occupant, self.world["building"])

        self.assertEqual(occupant.node_index, 0)
        self.assertEqual(occupant.edge_progress, 0.0)
        self.assertEqual(occupant.position, (1.0, 1.0))
        self.assertEqual(occupant.state, SandboxOccupantState.ROUTED)

    def test_reset_restores_original_floor_and_zone_after_crossing_a_stair(self):

        # Regression: reset used to only recompute a route from
        # wherever the occupant currently stood (its live floor_id/
        # zone_id, already overwritten by crossing the Stair), instead
        # of restoring the floor/zone it actually started on.
        occupant = self.manager.place_occupant(self.world["floor1"], x_m=1.0, y_m=1.0)
        original_floor_id = occupant.floor_id
        original_zone_id = occupant.zone_id
        original_position = occupant.position

        self.manager.compute_route(occupant, self.world["building"], SandboxDestinationType.EXIT)
        self.assertIn(self.world["stair"].id, occupant.route.edge_ids)

        # Cross the Stair (and however many further edges) fully.
        for _ in range(len(occupant.route.edges)):
            self.manager.step(occupant)

        self.assertNotEqual(occupant.floor_id, original_floor_id)
        self.assertNotEqual(occupant.zone_id, original_zone_id)

        self.manager.reset_occupant(occupant, self.world["building"])

        self.assertEqual(occupant.floor_id, original_floor_id)
        self.assertEqual(occupant.zone_id, original_zone_id)
        self.assertEqual(occupant.position, original_position)
        self.assertEqual(occupant.node_index, 0)
        self.assertEqual(occupant.edge_progress, 0.0)

        # The recomputed route must be rooted back at the true start,
        # so it crosses the Stair again rather than skipping straight
        # to whatever came after it.
        self.assertIn(self.world["stair"].id, occupant.route.edge_ids)

    def test_reset_on_an_occupant_with_no_destination_yet_returns_to_idle(self):

        occupant = self.manager.place_occupant(self.world["ground"], x_m=1.0, y_m=1.0)

        self.manager.reset_occupant(occupant, self.world["building"])

        self.assertEqual(occupant.state, SandboxOccupantState.IDLE)
        self.assertIsNone(occupant.route)

    def test_reset_on_an_unreachable_occupant_stays_unreachable(self):

        occupant = self.manager.place_occupant(self.world["ground"], x_m=41.0, y_m=41.0)
        self.manager.compute_route(occupant, self.world["building"], SandboxDestinationType.EXIT)
        self.assertEqual(occupant.state, SandboxOccupantState.UNREACHABLE)

        self.manager.reset_occupant(occupant, self.world["building"])

        self.assertEqual(occupant.state, SandboxOccupantState.UNREACHABLE)
        self.assertEqual(occupant.zone_id, self.world["isolated"].id)


if __name__ == "__main__":
    unittest.main()
