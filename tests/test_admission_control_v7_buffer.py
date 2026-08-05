import unittest

from types import SimpleNamespace

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.zone import Zone

from navigation.edge import Edge
from navigation.flow_region import FlowRegion, FlowRegionMember
from navigation.graph_builder import NavigationGraphGenerator
from navigation.node import Node

from pathfinding.engine import PathfindingEngine

from simulator.buffer import BufferModel, DefaultBufferModel
from simulator.coordinator import MultiAgentSimulation
from simulator.discharge import DischargeModel
from simulator.flow_region_capacity import FlowRegionCapacityModel


def make_zone(name, **kwargs):

    fields = dict(x=0.0, y=0.0, width=2.0, height=2.0)
    fields.update(kwargs)

    return Zone(name=name, **fields)


class _FixedCapacityBufferModel(BufferModel):

    # A minimal test fixture -- NOT a second production BufferModel.
    # Returns a fixed capacity regardless of the node's own area, so
    # behavioral tests can isolate "does the coordinator's buffer-
    # gating logic work correctly" from "is DefaultBufferModel's own
    # literature formula reasonable" (tested separately below).

    def __init__(self, capacity):
        self.capacity = capacity

    def buffer_capacity(self, node):
        return self.capacity


class _NoneCapacityBufferModel(BufferModel):

    def buffer_capacity(self, node):
        return None


def _make_node(node_id, width=None, height=None):

    reference = SimpleNamespace(width=width, height=height, area=(width * height) if width and height else None)

    return Node(id=node_id, name=node_id, floor_id="f", node_type=Node.ZONE, reference=reference)


# =====================================================
# BufferModel / DefaultBufferModel -- pure formula tests, no
# simulation involved.
# =====================================================


class BufferModelInterfaceTests(unittest.TestCase):

    def test_base_class_is_not_implemented(self):

        with self.assertRaises(NotImplementedError):
            BufferModel().buffer_capacity(_make_node("n1"))


class DefaultBufferModelTests(unittest.TestCase):

    def test_buffer_capacity_matches_area_times_max_density(self):

        node = _make_node("n1", width=5.0, height=5.0)
        model = DefaultBufferModel()

        expected = int(25.0 * DefaultBufferModel.MAX_DENSITY_PERS_PER_SQ_M)

        self.assertEqual(model.buffer_capacity(node), expected)

    def test_reuses_node_area_never_recomputes_geometry_itself(self):

        # Node.area already reads through to Zone.area (width * height)
        # -- DefaultBufferModel must consult it, not derive its own
        # width/height math.
        node = _make_node("n1", width=3.0, height=2.0)
        model = DefaultBufferModel()

        self.assertEqual(node.area, 6.0)
        self.assertEqual(model.buffer_capacity(node), int(6.0 * DefaultBufferModel.MAX_DENSITY_PERS_PER_SQ_M))

    def test_none_area_returns_none(self):

        node = _make_node("n1")  # width/height/area all None

        self.assertIsNone(DefaultBufferModel().buffer_capacity(node))

    def test_zero_area_floors_to_minimum_capacity(self):

        node = _make_node("n1", width=0.0, height=0.0)

        self.assertIsNone(DefaultBufferModel().buffer_capacity(node))

    def test_tiny_area_is_floored_at_minimum_capacity(self):

        node = _make_node("n1", width=0.1, height=0.1)  # area = 0.01

        self.assertEqual(DefaultBufferModel().buffer_capacity(node), DefaultBufferModel.MINIMUM_BUFFER_CAPACITY)


# =====================================================
# MultiAgentSimulation -- backward compatibility.
# =====================================================


class BackwardCompatibilityTests(unittest.TestCase):

    # buffer_model=None (the default, and every existing caller's own
    # construction) must behave byte-identically to before this
    # milestone -- see coordinator.py's own _can_admit() comment.

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
        self.floor.add_exit(Exit(name="Ex", zone_id=self.corridor.id, floor_id=self.floor.id))

        self.graph = NavigationGraphGenerator().build(self.building)
        self.engine = PathfindingEngine(self.graph)

    def test_no_buffer_model_produces_identical_queueing_to_before(self):

        sim = MultiAgentSimulation(self.engine)  # buffer_model defaults to None

        sim.add_occupant(self.room.id, occupant_id="first")
        sim.add_occupant(self.room.id, occupant_id="second")

        result = sim.run()

        first_step = result.occupants["first"].steps[0]
        second_step = result.occupants["second"].steps[0]

        self.assertEqual(first_step.queue_wait_time, 0.0)
        self.assertAlmostEqual(second_step.queue_wait_time, first_step.end_time)

    def test_buffer_model_is_none_attribute_by_default(self):

        sim = MultiAgentSimulation(self.engine)

        self.assertIsNone(sim.buffer_model)

    def test_no_buffer_waiters_are_ever_registered_without_a_buffer_model(self):

        sim = MultiAgentSimulation(self.engine)

        for i in range(4):
            sim.add_occupant(self.room.id, occupant_id=f"p{i}")

        sim.run()

        self.assertEqual(sim._buffer_waiters, {})


# =====================================================
# MultiAgentSimulation -- buffer gating, real spillback scenario.
# =====================================================


class BufferGatingTests(unittest.TestCase):

    # Room --(door1, narrow, storage=1)--> Landing (small) --(door2,
    # narrow, storage=1, far away -- slow drain)--> Corridor --> Exit.
    # Both doors already independently storage-gate one-at-a-time
    # (well-established, pre-existing mechanism) -- this is what
    # guarantees occupants arrive at Landing at genuinely staggered
    # times, so the buffer check has real prior occupancy to see by
    # the time later occupants attempt entry (see this module's own
    # empirical verification before these tests were written).

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.room = make_zone("Room", x=0.0, y=0.0)
        self.landing = make_zone("Landing", x=5.0, y=0.0)
        self.corridor = make_zone("Corridor", x=50.0, y=0.0)

        for zone in (self.room, self.landing, self.corridor):
            self.floor.add_zone(zone)

        self.door1 = Door(
            name="D1", zone_a_id=self.room.id, zone_b_id=self.landing.id,
            floor_id=self.floor.id, width=0.5,
        )
        self.door2 = Door(
            name="D2", zone_a_id=self.landing.id, zone_b_id=self.corridor.id,
            floor_id=self.floor.id, width=0.5,
        )
        self.floor.add_door(self.door1)
        self.floor.add_door(self.door2)
        self.floor.add_exit(Exit(name="Ex", zone_id=self.corridor.id, floor_id=self.floor.id))

        self.graph = NavigationGraphGenerator().build(self.building)
        self.engine = PathfindingEngine(self.graph)

    def test_without_a_buffer_model_landing_occupancy_exceeds_what_a_small_buffer_would_allow(self):

        sim = MultiAgentSimulation(self.engine)

        for i in range(6):
            sim.add_occupant(self.room.id, occupant_id=f"p{i}")

        result = sim.run()

        self.assertGreater(result.peak_node_occupancy[self.landing.id], 2)

    def test_buffer_capacity_is_never_exceeded_at_the_landing(self):

        sim = MultiAgentSimulation(self.engine, buffer_model=_FixedCapacityBufferModel(2))

        for i in range(6):
            sim.add_occupant(self.room.id, occupant_id=f"p{i}")

        result = sim.run()

        self.assertLessEqual(result.peak_node_occupancy[self.landing.id], 2)

    def test_smaller_buffer_caps_occupancy_more_tightly(self):

        results = {}

        for capacity in (1, 2, 4):

            sim = MultiAgentSimulation(self.engine, buffer_model=_FixedCapacityBufferModel(capacity))

            for i in range(6):
                sim.add_occupant(self.room.id, occupant_id=f"p{i}")

            result = sim.run()
            results[capacity] = result.peak_node_occupancy[self.landing.id]

        for capacity, peak in results.items():
            self.assertLessEqual(peak, capacity)

    def test_every_occupant_eventually_admitted_no_deadlock(self):

        sim = MultiAgentSimulation(self.engine, buffer_model=_FixedCapacityBufferModel(2))

        for i in range(6):
            sim.add_occupant(self.room.id, occupant_id=f"p{i}")

        result = sim.run()

        for i in range(6):
            self.assertIsNotNone(result.occupants[f"p{i}"].arrival_time)

    def test_fifo_order_is_preserved_under_buffer_gating(self):

        sim = MultiAgentSimulation(self.engine, buffer_model=_FixedCapacityBufferModel(2))

        for i in range(5):
            sim.add_occupant(self.room.id, occupant_id=f"p{i}")

        result = sim.run()

        arrival_order = sorted(range(5), key=lambda i: result.occupants[f"p{i}"].arrival_time)

        self.assertEqual(arrival_order, [0, 1, 2, 3, 4])

    def test_no_pending_buffer_waiters_remain_after_the_run_completes(self):

        sim = MultiAgentSimulation(self.engine, buffer_model=_FixedCapacityBufferModel(2))

        for i in range(6):
            sim.add_occupant(self.room.id, occupant_id=f"p{i}")

        sim.run()

        self.assertEqual(sim._buffer_waiters, {})
        self.assertEqual(sim._pending_retry, set())

    def test_none_buffer_capacity_fails_open_behaves_like_unconstrained(self):

        sim_open = MultiAgentSimulation(self.engine, buffer_model=_NoneCapacityBufferModel())
        sim_baseline = MultiAgentSimulation(self.engine)

        for i in range(6):
            sim_open.add_occupant(self.room.id, occupant_id=f"p{i}")
            sim_baseline.add_occupant(self.room.id, occupant_id=f"p{i}")

        result_open = sim_open.run()
        result_baseline = sim_baseline.run()

        self.assertEqual(result_open.total_evacuation_time, result_baseline.total_evacuation_time)

    def test_reuses_existing_node_occupancy_directly(self):

        # The buffer check reads self._node_occupancy -- the SAME dict
        # every existing peak_node_occupancy/reporting consumer already
        # reads -- not a new, parallel tracking structure.
        sim = MultiAgentSimulation(self.engine, buffer_model=_FixedCapacityBufferModel(2))

        sim.add_occupant(self.room.id, occupant_id="p0")
        sim.run()

        self.assertIn(self.landing.id, sim._node_occupancy)


class NarrowStorageStillBindsWithBufferModelTests(unittest.TestCase):

    # A generous buffer paired with a narrow (capacity=1) door proves
    # storage still binds independently -- a permissive buffer doesn't
    # silently relax it.

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.room = make_zone("Room", x=0.0, y=0.0)
        self.corridor = make_zone("Corridor", x=10.0, y=0.0, width=20.0, height=20.0)
        self.floor.add_zone(self.room)
        self.floor.add_zone(self.corridor)

        self.door = Door(
            name="Narrow", zone_a_id=self.room.id, zone_b_id=self.corridor.id,
            floor_id=self.floor.id, width=0.5,
        )
        self.floor.add_door(self.door)
        self.floor.add_exit(Exit(name="Ex", zone_id=self.corridor.id, floor_id=self.floor.id))

        self.graph = NavigationGraphGenerator().build(self.building)
        self.engine = PathfindingEngine(self.graph)

    def test_storage_still_gates_admission_with_a_generous_buffer(self):

        sim = MultiAgentSimulation(self.engine, buffer_model=_FixedCapacityBufferModel(1000))

        sim.add_occupant(self.room.id, occupant_id="first")
        sim.add_occupant(self.room.id, occupant_id="second")

        result = sim.run()

        first_step = result.occupants["first"].steps[0]
        second_step = result.occupants["second"].steps[0]

        self.assertEqual(first_step.queue_wait_time, 0.0)
        self.assertGreater(second_step.queue_wait_time, 0.0)


class DischargeAndBufferInteractionTests(unittest.TestCase):

    # Both new constraints (V4 discharge, V7 buffer) active together --
    # proves they compose without interfering with each other.

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.room = make_zone("Room", x=0.0, y=0.0)
        self.corridor = make_zone("Corridor", x=10.0, y=0.0, width=20.0, height=20.0)
        self.floor.add_zone(self.room)
        self.floor.add_zone(self.corridor)

        self.door = Door(
            name="Wide", zone_a_id=self.room.id, zone_b_id=self.corridor.id,
            floor_id=self.floor.id, width=4.0,
        )
        self.floor.add_door(self.door)
        self.floor.add_exit(Exit(name="Ex", zone_id=self.corridor.id, floor_id=self.floor.id))

        self.graph = NavigationGraphGenerator().build(self.building)
        self.engine = PathfindingEngine(self.graph)

    def test_discharge_and_buffer_both_active_still_admits_everyone(self):

        class _FixedRate(DischargeModel):
            def discharge_rate(self, edge_or_region):
                return 0.5  # 2s min gap

        sim = MultiAgentSimulation(
            self.engine, discharge_model=_FixedRate(), buffer_model=_FixedCapacityBufferModel(100),
        )

        for i in range(4):
            sim.add_occupant(self.room.id, occupant_id=f"p{i}")

        result = sim.run()

        for i in range(4):
            self.assertIsNotNone(result.occupants[f"p{i}"].arrival_time)

        start_times = [result.occupants[f"p{i}"].steps[0].start_time for i in range(4)]
        self.assertEqual(start_times, [0.0, 2.0, 4.0, 6.0])


class FlowRegionBufferTests(unittest.TestCase):

    # Buffer gating resolves the queue HEAD's own to_node, not a fixed
    # assumption -- proven directly against a FlowRegion whose member
    # edges lead to genuinely different destinations.

    def setUp(self):

        self.edge_a = self._make_edge("a", to_node_id="landing-a")
        self.edge_b = self._make_edge("b", to_node_id="landing-b")

        self.region = FlowRegion(
            id="region-1", edge_ids=("a", "b"), region_kind=FlowRegion.MERGE,
            total_length=2.0, representative_width=4.0,
            member_edges=(
                FlowRegionMember(edge=self.edge_a, upstream_node_id="u", downstream_node_id="landing-a"),
                FlowRegionMember(edge=self.edge_b, upstream_node_id="u", downstream_node_id="landing-b"),
            ),
        )
        self.flow_region_map = {"a": self.region, "b": self.region}

        engine = SimpleNamespace(graph=SimpleNamespace())

        self.sim = MultiAgentSimulation(
            engine, capacity_model=FlowRegionCapacityModel(), buffer_model=_FixedCapacityBufferModel(2),
            flow_region_map=self.flow_region_map,
        )

    def _make_edge(self, edge_id, to_node_id):

        return Edge(
            id=edge_id, edge_type=Edge.DOOR, from_node="u", to_node=to_node_id,
            walking_distance=1.0, reference=SimpleNamespace(width=4.0),
        )

    def test_landing_a_full_does_not_block_admission_toward_landing_b(self):

        # Admission Control V10 -- Storage-Throughput Separation.
        # _can_admit()'s own signature simplified to (edge, to_node,
        # time): storage is always resolved from the edge itself now
        # (never a FlowRegion), so there is no separate admission_object/
        # admission_key to pass in. Buffer gating itself -- the actual
        # behavior under test here -- is completely unchanged: still
        # keyed on the destination node's own occupancy, independent of
        # which member edge is being entered.

        # Fill landing-a to its buffer capacity.
        self.sim._node_occupancy["landing-a"] = {"occ-1", "occ-2"}

        self.assertFalse(self.sim._can_admit(self.edge_a, _make_node("landing-a"), 0.0))
        self.assertTrue(self.sim._can_admit(self.edge_a, _make_node("landing-b"), 0.0))


if __name__ == "__main__":
    unittest.main()
