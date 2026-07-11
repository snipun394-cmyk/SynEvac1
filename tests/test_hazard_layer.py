import unittest

from dataclasses import FrozenInstanceError

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.zone import Zone

from navigation.cost import DefaultCostModel
from navigation.graph_builder import NavigationGraphGenerator

from pathfinding.engine import PathfindingEngine

from simulator.capacity import DefaultCapacityModel
from simulator.coordinator import MultiAgentSimulation

from behavior.context import DecisionContext
from behavior.profile import BehaviorProfile

from hazard.capacity_model import HazardAwareCapacityModel
from hazard.cost_model import HazardAwareCostModel
from hazard.edge_state import HazardEdgeState
from hazard.node_state import HazardNodeState
from hazard.provider import ManualHazardProvider
from hazard.severity import HazardSeverity
from hazard.snapshot import HazardSnapshot


def make_zone(name, **kwargs):

    fields = dict(x=0.0, y=0.0, width=2.0, height=2.0)
    fields.update(kwargs)

    return Zone(name=name, **fields)


def build_diamond_building():

    # A -- door AB --> B -- door B_Exit --> D --exit--> Outside
    # A -- door AC --> C -- door C_Exit --> D
    # Two independent routes from A to D, so blocking one still
    # leaves the other usable.

    building = Building(name="B")
    floor = building.create_floor(name="Ground Floor")

    zone_a = make_zone("A", x=0.0, y=0.0)
    zone_b = make_zone("B", x=5.0, y=5.0)
    zone_c = make_zone("C", x=5.0, y=-5.0)
    zone_d = make_zone("D", x=10.0, y=0.0)

    for zone in (zone_a, zone_b, zone_c, zone_d):
        floor.add_zone(zone)

    door_ab = Door(name="AB", zone_a_id=zone_a.id, zone_b_id=zone_b.id, floor_id=floor.id)
    door_bd = Door(name="BD", zone_a_id=zone_b.id, zone_b_id=zone_d.id, floor_id=floor.id)
    door_ac = Door(name="AC", zone_a_id=zone_a.id, zone_b_id=zone_c.id, floor_id=floor.id)
    door_cd = Door(name="CD", zone_a_id=zone_c.id, zone_b_id=zone_d.id, floor_id=floor.id)

    for door in (door_ab, door_bd, door_ac, door_cd):
        floor.add_door(door)

    exit_obj = Exit(name="Ex", zone_id=zone_d.id, floor_id=floor.id)
    floor.add_exit(exit_obj)

    graph = NavigationGraphGenerator().build(building)
    engine = PathfindingEngine(graph)

    zones = dict(A=zone_a, B=zone_b, C=zone_c, D=zone_d)
    doors = dict(AB=door_ab, BD=door_bd, AC=door_ac, CD=door_cd)

    return building, floor, zones, doors, exit_obj, engine


def build_two_zone_building(door_width=4.0):

    building = Building(name="B")
    floor = building.create_floor(name="Ground Floor")

    room = make_zone("Room", x=0.0, y=0.0)
    corridor = make_zone("Corridor", x=10.0, y=0.0)

    floor.add_zone(room)
    floor.add_zone(corridor)

    door = Door(
        name="D1", zone_a_id=room.id, zone_b_id=corridor.id,
        floor_id=floor.id, width=door_width,
    )
    floor.add_door(door)

    exit_obj = Exit(name="Ex", zone_id=corridor.id, floor_id=floor.id)
    floor.add_exit(exit_obj)

    graph = NavigationGraphGenerator().build(building)
    engine = PathfindingEngine(graph)

    return building, floor, room, corridor, door, exit_obj, engine


class HazardSeverityTests(unittest.TestCase):

    def test_boundaries_map_to_expected_buckets(self):

        self.assertEqual(HazardSeverity.from_score(0.0), HazardSeverity.NONE)
        self.assertEqual(HazardSeverity.from_score(0.04), HazardSeverity.NONE)
        self.assertEqual(HazardSeverity.from_score(0.05), HazardSeverity.LOW)
        self.assertEqual(HazardSeverity.from_score(0.29), HazardSeverity.LOW)
        self.assertEqual(HazardSeverity.from_score(0.30), HazardSeverity.MODERATE)
        self.assertEqual(HazardSeverity.from_score(0.59), HazardSeverity.MODERATE)
        self.assertEqual(HazardSeverity.from_score(0.60), HazardSeverity.HIGH)
        self.assertEqual(HazardSeverity.from_score(0.84), HazardSeverity.HIGH)
        self.assertEqual(HazardSeverity.from_score(0.85), HazardSeverity.CRITICAL)
        self.assertEqual(HazardSeverity.from_score(1.0), HazardSeverity.CRITICAL)

    def test_none_score_maps_to_none_severity(self):

        self.assertEqual(HazardSeverity.from_score(None), HazardSeverity.NONE)

    def test_thresholds_are_not_duplicated_as_enum_members(self):

        # Guards against the exact mistake this file's first draft
        # made: threshold constants must never leak into the enum's
        # own member set.
        names = {member.name for member in HazardSeverity}
        self.assertEqual(names, {"NONE", "LOW", "MODERATE", "HIGH", "CRITICAL"})


class HazardNodeStateTests(unittest.TestCase):

    def test_default_state_is_clear(self):

        state = HazardNodeState()

        self.assertEqual(state.hazard_score, 0.0)
        self.assertEqual(state.severity, HazardSeverity.NONE)

    def test_severity_is_derived_not_authored(self):

        state = HazardNodeState(hazard_score=0.9)

        self.assertEqual(state.severity, HazardSeverity.CRITICAL)

    def test_severity_tracks_a_single_centralized_mapping(self):

        for score in (0.0, 0.1, 0.4, 0.7, 0.95):

            state = HazardNodeState(hazard_score=score)
            self.assertEqual(state.severity, HazardSeverity.from_score(score))

    def test_is_frozen(self):

        state = HazardNodeState(hazard_score=0.2)

        with self.assertRaises(FrozenInstanceError):
            state.hazard_score = 0.9


class HazardEdgeStateTests(unittest.TestCase):

    def test_default_state_is_fully_open_and_unmodified(self):

        state = HazardEdgeState()

        self.assertIsNone(state.traversable)
        self.assertEqual(state.capacity_modifier, 1.0)
        self.assertEqual(state.cost_penalty, 0.0)

    def test_negative_cost_penalty_is_rejected(self):

        with self.assertRaises(ValueError):
            HazardEdgeState(cost_penalty=-1.0)

    def test_capacity_modifier_above_one_is_rejected(self):

        with self.assertRaises(ValueError):
            HazardEdgeState(capacity_modifier=1.5)

    def test_capacity_modifier_below_zero_is_rejected(self):

        with self.assertRaises(ValueError):
            HazardEdgeState(capacity_modifier=-0.1)

    def test_blocked_cost_is_infinite_not_negative(self):

        self.assertEqual(HazardEdgeState.BLOCKED_COST, float("inf"))
        self.assertGreater(HazardEdgeState.BLOCKED_COST, 0)


class HazardSnapshotTests(unittest.TestCase):

    def test_missing_node_and_edge_default_to_clear(self):

        snapshot = HazardSnapshot()

        self.assertEqual(snapshot.node_state("unknown"), HazardNodeState())
        self.assertEqual(snapshot.edge_state("unknown"), HazardEdgeState())

    def test_known_ids_return_the_authored_state(self):

        node_state = HazardNodeState(hazard_score=0.7)
        edge_state = HazardEdgeState(cost_penalty=5.0)

        snapshot = HazardSnapshot(
            node_states={"n1": node_state},
            edge_states={"e1": edge_state},
        )

        self.assertEqual(snapshot.node_state("n1"), node_state)
        self.assertEqual(snapshot.edge_state("e1"), edge_state)

    def test_snapshot_id_defaults_to_a_unique_value(self):

        first = HazardSnapshot()
        second = HazardSnapshot()

        self.assertNotEqual(first.snapshot_id, second.snapshot_id)

    def test_explicit_snapshot_id_and_timestamp_are_honored(self):

        snapshot = HazardSnapshot(snapshot_id="incident-t90", timestamp=90.0)

        self.assertEqual(snapshot.snapshot_id, "incident-t90")
        self.assertEqual(snapshot.timestamp, 90.0)

    def test_is_frozen(self):

        snapshot = HazardSnapshot()

        with self.assertRaises(FrozenInstanceError):
            snapshot.timestamp = 5.0

    def test_node_states_mapping_is_truly_read_only(self):

        snapshot = HazardSnapshot(node_states={"n1": HazardNodeState()})

        with self.assertRaises(TypeError):
            snapshot.node_states["n2"] = HazardNodeState()

    def test_edge_states_mapping_is_truly_read_only(self):

        snapshot = HazardSnapshot(edge_states={"e1": HazardEdgeState()})

        with self.assertRaises(TypeError):
            snapshot.edge_states["e2"] = HazardEdgeState()

    def test_mutating_the_source_dict_after_construction_does_not_leak_in(self):

        source = {"n1": HazardNodeState()}
        snapshot = HazardSnapshot(node_states=source)

        source["n2"] = HazardNodeState(hazard_score=0.5)

        self.assertNotIn("n2", snapshot.node_states)


class ManualHazardProviderTests(unittest.TestCase):

    def test_returns_the_same_snapshot_regardless_of_time(self):

        snapshot = HazardSnapshot(timestamp=42.0)
        provider = ManualHazardProvider(snapshot)

        self.assertIs(provider.snapshot_at(0.0), snapshot)
        self.assertIs(provider.snapshot_at(9999.0), snapshot)

    def test_defaults_to_an_empty_clear_snapshot(self):

        provider = ManualHazardProvider()

        snapshot = provider.snapshot_at(0.0)

        self.assertEqual(snapshot.node_state("anything"), HazardNodeState())


class HazardAwareCostModelTests(unittest.TestCase):

    def setUp(self):

        (
            self.building, self.floor, self.room, self.corridor,
            self.door, self.exit_obj, self.engine,
        ) = build_two_zone_building()

    def test_no_hazard_state_matches_the_base_model(self):

        snapshot = HazardSnapshot()
        hazard_cost_model = HazardAwareCostModel(DefaultCostModel(), snapshot)

        graph_edge = self.engine.graph.find_neighbors(
            self.engine.graph.find_node(self.room.id)
        )[0][1]

        self.assertEqual(
            hazard_cost_model.cost(graph_edge),
            DefaultCostModel().cost(graph_edge),
        )

    def test_cost_penalty_is_added_on_top_of_the_base_cost(self):

        graph_edge = self.engine.graph.find_neighbors(
            self.engine.graph.find_node(self.room.id)
        )[0][1]

        snapshot = HazardSnapshot(
            edge_states={graph_edge.id: HazardEdgeState(cost_penalty=100.0)},
        )
        hazard_cost_model = HazardAwareCostModel(DefaultCostModel(), snapshot)

        base_cost = DefaultCostModel().cost(graph_edge)

        self.assertEqual(hazard_cost_model.cost(graph_edge), base_cost + 100.0)

    def test_blocked_edge_returns_infinite_cost_not_negative(self):

        graph_edge = self.engine.graph.find_neighbors(
            self.engine.graph.find_node(self.room.id)
        )[0][1]

        snapshot = HazardSnapshot(
            edge_states={graph_edge.id: HazardEdgeState(traversable=False)},
        )
        hazard_cost_model = HazardAwareCostModel(DefaultCostModel(), snapshot)

        self.assertEqual(hazard_cost_model.cost(graph_edge), float("inf"))
        self.assertGreaterEqual(hazard_cost_model.cost(graph_edge), 0)


class BlockedEdgeRoutingTests(unittest.TestCase):

    def setUp(self):

        (
            self.building, self.floor, self.zones, self.doors,
            self.exit_obj, self.engine,
        ) = build_diamond_building()

    def _engine_with_blocked_door(self, door_name):

        door_id = self.doors[door_name].id

        snapshot = HazardSnapshot(
            edge_states={door_id: HazardEdgeState(traversable=False)},
        )
        hazard_cost_model = HazardAwareCostModel(DefaultCostModel(), snapshot)

        return PathfindingEngine(self.engine.graph, cost_model=hazard_cost_model)

    def test_blocking_one_route_reroutes_through_the_other(self):

        hazard_engine = self._engine_with_blocked_door("AB")

        route = hazard_engine.dijkstra(self.zones["A"].id, self.zones["D"].id)

        self.assertIsNotNone(route)
        self.assertNotIn(self.doors["AB"].id, route.edge_ids)
        self.assertIn(self.doors["AC"].id, route.edge_ids)
        self.assertLess(route.total_cost, float("inf"))

    def test_blocking_every_route_reports_unreachable_like_a_structural_block(self):

        # Both parallel routes closed -- PathfindingEngine._relax only
        # ever relaxes onto a *strictly smaller* tentative cost than
        # its current best (default math.inf), so an infinite-cost
        # edge is never relaxed onto at all. A hazard blocking every
        # path to the goal is therefore indistinguishable from a
        # structurally locked door doing the same: dijkstra() returns
        # None, the same "no Route" contract used everywhere else --
        # no special-cased "infinite-cost but present" Route.
        door_ab_id = self.doors["AB"].id
        door_ac_id = self.doors["AC"].id

        snapshot = HazardSnapshot(
            edge_states={
                door_ab_id: HazardEdgeState(traversable=False),
                door_ac_id: HazardEdgeState(traversable=False),
            },
        )
        hazard_cost_model = HazardAwareCostModel(DefaultCostModel(), snapshot)
        hazard_engine = PathfindingEngine(self.engine.graph, cost_model=hazard_cost_model)

        route = hazard_engine.dijkstra(self.zones["A"].id, self.zones["D"].id)

        self.assertIsNone(route)

    def test_unblocked_baseline_prefers_neither_route_arbitrarily_but_is_finite(self):

        route = self.engine.dijkstra(self.zones["A"].id, self.zones["D"].id)

        self.assertIsNotNone(route)
        self.assertLess(route.total_cost, float("inf"))


class HazardAwareCapacityModelTests(unittest.TestCase):

    def setUp(self):

        (
            self.building, self.floor, self.room, self.corridor,
            self.door, self.exit_obj, self.engine,
        ) = build_two_zone_building(door_width=4.0)

        self.graph_edge = self.engine.graph.find_neighbors(
            self.engine.graph.find_node(self.room.id)
        )[0][1]

    def test_no_hazard_state_matches_the_base_model(self):

        snapshot = HazardSnapshot()
        model = HazardAwareCapacityModel(DefaultCapacityModel(), snapshot)

        self.assertEqual(
            model.capacity(self.graph_edge),
            DefaultCapacityModel().capacity(self.graph_edge),
        )

    def test_capacity_modifier_reduces_effective_capacity(self):

        base_capacity = DefaultCapacityModel().capacity(self.graph_edge)
        self.assertEqual(base_capacity, 6)  # width 4.0 * 1.5 people/m

        snapshot = HazardSnapshot(
            edge_states={self.graph_edge.id: HazardEdgeState(capacity_modifier=0.5)},
        )
        model = HazardAwareCapacityModel(DefaultCapacityModel(), snapshot)

        self.assertEqual(model.capacity(self.graph_edge), 3)

    def test_capacity_never_drops_below_the_minimum_floor(self):

        # A hazard driving capacity_modifier all the way to 0.0 must
        # still never deadlock MultiAgentSimulation's queue -- the
        # floor is preserved exactly like DefaultCapacityModel's own.
        snapshot = HazardSnapshot(
            edge_states={self.graph_edge.id: HazardEdgeState(capacity_modifier=0.0)},
        )
        model = HazardAwareCapacityModel(DefaultCapacityModel(), snapshot)

        self.assertEqual(model.capacity(self.graph_edge), DefaultCapacityModel.MINIMUM_CAPACITY)

    def test_reduced_capacity_changes_queueing_behavior_in_simulation(self):

        snapshot = HazardSnapshot(
            edge_states={self.graph_edge.id: HazardEdgeState(capacity_modifier=1 / 6)},
        )
        hazard_capacity_model = HazardAwareCapacityModel(DefaultCapacityModel(), snapshot)

        sim = MultiAgentSimulation(self.engine, capacity_model=hazard_capacity_model)

        for i in range(3):
            sim.add_occupant(self.room.id, occupant_id=f"p{i}")

        result = sim.run()

        # Capacity floored at 1 -- occupants after the first must queue.
        self.assertEqual(result.occupants["p0"].steps[0].queue_wait_time, 0.0)
        self.assertGreater(result.occupants["p1"].steps[0].queue_wait_time, 0.0)
        self.assertGreater(result.total_queue_events, 0)


class DecisionContextIntegrationTests(unittest.TestCase):

    def test_hazard_snapshot_defaults_to_none(self):

        context = DecisionContext(
            graph=None, engine=None, profile=BehaviorProfile(occupant_id="p1"),
            start_id="z1",
        )

        self.assertIsNone(context.hazard_snapshot)

    def test_existing_callers_that_omit_hazard_snapshot_still_work(self):

        # Backward compatibility: every existing construction of
        # DecisionContext (see test_behavior_layer.py) omits this
        # field entirely.
        context = DecisionContext(
            graph=None, engine=None, profile=BehaviorProfile(occupant_id="p1"),
            start_id="z1", decisions_so_far={}, prior_result=None,
        )

        self.assertIsNone(context.hazard_snapshot)

    def test_hazard_snapshot_can_be_supplied_and_queried_by_a_strategy(self):

        snapshot = HazardSnapshot(
            node_states={"z1": HazardNodeState(hazard_score=0.95)},
        )

        context = DecisionContext(
            graph=None, engine=None, profile=BehaviorProfile(occupant_id="p1"),
            start_id="z1", hazard_snapshot=snapshot,
        )

        severity = context.hazard_snapshot.node_state(context.start_id).severity

        self.assertEqual(severity, HazardSeverity.CRITICAL)


class HazardLayerIndependenceTests(unittest.TestCase):

    def test_hazard_package_never_touches_reference_or_designer(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "hazard"

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertNotIn(
                ".reference", text, f"{path.name} touches .reference directly"
            )
            self.assertIsNone(
                re.search(r"^\s*(from|import)\s+(models|designer)\b", text, re.MULTILINE),
                f"{path.name} imports models/designer directly -- the Building "
                f"Model must stay untouched by the Hazard Layer",
            )

    def test_frozen_subsystems_never_import_hazard(self):

        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parent.parent

        frozen_dirs = ("navigation", "pathfinding", "simulator", "analysis")

        for dir_name in frozen_dirs:

            for path in (root / dir_name).glob("*.py"):

                text = path.read_text()

                self.assertIsNone(
                    re.search(r"^\s*(from|import)\s+hazard\b", text, re.MULTILINE),
                    f"{dir_name}/{path.name} imports hazard/ -- reverses the "
                    f"dependency direction; frozen subsystems must stay "
                    f"unaware of the Hazard Layer",
                )

    def test_behavior_package_only_touches_hazard_through_the_snapshot_field(self):

        import pathlib

        text = (
            pathlib.Path(__file__).resolve().parent.parent
            / "behavior" / "context.py"
        ).read_text()

        self.assertIn("hazard_snapshot", text)

        other_behavior_files = [
            path
            for path in (
                pathlib.Path(__file__).resolve().parent.parent / "behavior"
            ).glob("*.py")
            if path.name != "context.py"
        ]

        for path in other_behavior_files:

            text = path.read_text()

            self.assertNotIn(
                "hazard", text.lower(),
                f"{path.name} references hazard/ -- only context.py should",
            )


if __name__ == "__main__":
    unittest.main()
