import unittest

from types import SimpleNamespace

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.zone import Zone

from navigation.edge import Edge
from navigation.flow_region import FlowRegion, FlowRegionMember
from navigation.graph_builder import NavigationGraphGenerator

from pathfinding.engine import PathfindingEngine

from simulator.coordinator import MultiAgentSimulation
from simulator.discharge import DefaultDischargeModel, DischargeModel
from simulator.flow_region_capacity import FlowRegionCapacityModel, FlowRegionCapacityModelV2


def make_zone(name, **kwargs):

    fields = dict(x=0.0, y=0.0, width=2.0, height=2.0)
    fields.update(kwargs)

    return Zone(name=name, **fields)


class _FixedRateDischargeModel(DischargeModel):

    # A minimal test fixture -- NOT a second production DischargeModel.
    # Returns a fixed rate regardless of the edge/region's own width,
    # so behavioral tests can isolate "does the coordinator's gating
    # logic work correctly" from "is DefaultDischargeModel's own
    # literature formula reasonable" (tested separately below).

    def __init__(self, rate):
        self.rate = rate

    def discharge_rate(self, edge_or_region):
        return self.rate


class _NoneRateDischargeModel(DischargeModel):

    def discharge_rate(self, edge_or_region):
        return None


def _make_edge(edge_id, width=1.0, walking_distance=1.0):

    return Edge(
        id=edge_id, edge_type=Edge.DOOR, from_node="a", to_node="b",
        walking_distance=walking_distance, reference=SimpleNamespace(width=width),
    )


# =====================================================
# DischargeModel / DefaultDischargeModel -- pure formula tests, no
# simulation involved.
# =====================================================


class DischargeModelInterfaceTests(unittest.TestCase):

    def test_base_class_is_not_implemented(self):

        with self.assertRaises(NotImplementedError):
            DischargeModel().discharge_rate(_make_edge("d1"))


class DefaultDischargeModelTests(unittest.TestCase):

    def test_discharge_rate_matches_the_specific_flow_formula(self):

        edge = _make_edge("d1", width=1.4)
        model = DefaultDischargeModel()

        effective_width = 1.4 - DefaultDischargeModel.BOUNDARY_LAYER_M
        expected = DefaultDischargeModel.SPECIFIC_FLOW_PERS_PER_S_PER_M * effective_width

        self.assertAlmostEqual(model.discharge_rate(edge), expected)

    def test_effective_width_is_floored_rather_than_going_negative(self):

        edge = _make_edge("d1", width=0.05)
        model = DefaultDischargeModel()

        expected = DefaultDischargeModel.SPECIFIC_FLOW_PERS_PER_S_PER_M * DefaultDischargeModel.MINIMUM_EFFECTIVE_WIDTH_M

        self.assertAlmostEqual(model.discharge_rate(edge), expected)

    def test_width_none_returns_none(self):

        edge = _make_edge("d1")
        edge.reference.width = None

        self.assertIsNone(DefaultDischargeModel().discharge_rate(edge))

    def test_flow_region_uses_its_own_representative_width(self):

        region = FlowRegion(
            id="region-1", edge_ids=("d1",), region_kind=FlowRegion.CHAIN,
            total_length=10.0, representative_width=1.27, member_edges=(),
        )
        model = DefaultDischargeModel()

        effective_width = 1.27 - DefaultDischargeModel.BOUNDARY_LAYER_M
        expected = DefaultDischargeModel.SPECIFIC_FLOW_PERS_PER_S_PER_M * effective_width

        self.assertAlmostEqual(model.discharge_rate(region), expected)


# =====================================================
# MultiAgentSimulation -- backward compatibility.
# =====================================================


class BackwardCompatibilityTests(unittest.TestCase):

    # discharge_model=None (the default, and every existing caller's
    # own construction) must behave byte-identically to before this
    # milestone -- see coordinator.py's own _can_admit() comment. This
    # exercises the exact QueueFormationTests fixture shape from
    # tests/test_multi_agent_simulation.py.

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

    def test_no_discharge_model_produces_identical_queueing_to_before(self):

        sim = MultiAgentSimulation(self.engine)  # discharge_model defaults to None

        sim.add_occupant(self.room.id, occupant_id="first")
        sim.add_occupant(self.room.id, occupant_id="second")

        result = sim.run()

        first_step = result.occupants["first"].steps[0]
        second_step = result.occupants["second"].steps[0]

        self.assertEqual(first_step.queue_wait_time, 0.0)
        self.assertAlmostEqual(second_step.queue_wait_time, first_step.end_time)

    def test_discharge_model_is_none_attribute_by_default(self):

        sim = MultiAgentSimulation(self.engine)

        self.assertIsNone(sim.discharge_model)

    def test_no_retry_admission_events_are_ever_scheduled_without_a_discharge_model(self):

        sim = MultiAgentSimulation(self.engine)

        for i in range(4):
            sim.add_occupant(self.room.id, occupant_id=f"p{i}")

        sim.run()

        kinds = {entry[2] for entry in sim._event_heap}  # heap is empty after run(), but guard anyway
        self.assertNotIn(MultiAgentSimulation.RETRY_ADMISSION, kinds)
        self.assertEqual(sim._pending_retry, set())


# =====================================================
# MultiAgentSimulation -- discharge-rate gating, generous storage.
# =====================================================


class DischargeGatingTests(unittest.TestCase):

    # A WIDE door (generous storage capacity, ~6 under DefaultCapacityModel)
    # paired with a small, fixed discharge rate -- isolates the
    # discharge constraint as the sole binding one, proving storage and
    # throughput are genuinely independent (the whole point of this
    # milestone), not that a narrow door happens to also rate-limit.

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.room = make_zone("Room", x=0.0, y=0.0)
        self.corridor = make_zone("Corridor", x=10.0, y=0.0)
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

    def test_storage_alone_would_admit_everyone_instantly(self):

        # Sanity check on the fixture itself: without a discharge
        # model, 3 occupants through a wide door all start immediately.
        sim = MultiAgentSimulation(self.engine)

        for i in range(3):
            sim.add_occupant(self.room.id, occupant_id=f"p{i}")

        result = sim.run()

        for i in range(3):
            self.assertEqual(result.occupants[f"p{i}"].steps[0].queue_wait_time, 0.0)

    def test_discharge_rate_enforces_a_minimum_inter_admission_gap(self):

        discharge_model = _FixedRateDischargeModel(rate=0.5)  # min gap = 2.0s
        sim = MultiAgentSimulation(self.engine, discharge_model=discharge_model)

        for i in range(3):
            sim.add_occupant(self.room.id, occupant_id=f"p{i}")

        result = sim.run()

        start_times = [result.occupants[f"p{i}"].steps[0].start_time for i in range(3)]

        self.assertEqual(start_times, [0.0, 2.0, 4.0])

    def test_fifo_order_is_preserved_under_discharge_gating(self):

        discharge_model = _FixedRateDischargeModel(rate=0.5)
        sim = MultiAgentSimulation(self.engine, discharge_model=discharge_model)

        for i in range(4):
            sim.add_occupant(self.room.id, occupant_id=f"p{i}")

        result = sim.run()

        start_times = [(i, result.occupants[f"p{i}"].steps[0].start_time) for i in range(4)]
        ordered_by_time = sorted(start_times, key=lambda pair: pair[1])

        self.assertEqual([i for i, _ in ordered_by_time], [0, 1, 2, 3])

    def test_every_occupant_eventually_admitted_no_deadlock(self):

        discharge_model = _FixedRateDischargeModel(rate=0.5)
        sim = MultiAgentSimulation(self.engine, discharge_model=discharge_model)

        for i in range(5):
            sim.add_occupant(self.room.id, occupant_id=f"p{i}")

        result = sim.run()

        for i in range(5):
            self.assertIsNotNone(result.occupants[f"p{i}"].arrival_time)

    def test_discharge_wait_is_recorded_as_queue_wait_time(self):

        discharge_model = _FixedRateDischargeModel(rate=0.5)
        sim = MultiAgentSimulation(self.engine, discharge_model=discharge_model)

        sim.add_occupant(self.room.id, occupant_id="first")
        sim.add_occupant(self.room.id, occupant_id="second")

        result = sim.run()

        self.assertEqual(result.occupants["first"].steps[0].queue_wait_time, 0.0)
        self.assertAlmostEqual(result.occupants["second"].steps[0].queue_wait_time, 2.0)

    def test_none_discharge_rate_fails_open_behaves_like_storage_only(self):

        sim = MultiAgentSimulation(self.engine, discharge_model=_NoneRateDischargeModel())

        for i in range(3):
            sim.add_occupant(self.room.id, occupant_id=f"p{i}")

        result = sim.run()

        for i in range(3):
            self.assertEqual(result.occupants[f"p{i}"].steps[0].queue_wait_time, 0.0)

    def test_no_pending_retries_remain_after_the_run_completes(self):

        discharge_model = _FixedRateDischargeModel(rate=0.5)
        sim = MultiAgentSimulation(self.engine, discharge_model=discharge_model)

        for i in range(3):
            sim.add_occupant(self.room.id, occupant_id=f"p{i}")

        sim.run()

        self.assertEqual(sim._pending_retry, set())


class NarrowStorageStillBindsWithDischargeModelTests(unittest.TestCase):

    # The inverse of DischargeGatingTests -- a narrow (capacity=1) door
    # with a GENEROUS discharge rate, proving storage still binds
    # independently and a permissive discharge model doesn't silently
    # relax it.

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

    def test_storage_still_gates_admission_even_with_a_generous_discharge_rate(self):

        discharge_model = _FixedRateDischargeModel(rate=100.0)  # min gap = 0.01s, effectively no constraint
        sim = MultiAgentSimulation(self.engine, discharge_model=discharge_model)

        sim.add_occupant(self.room.id, occupant_id="first")
        sim.add_occupant(self.room.id, occupant_id="second")

        result = sim.run()

        first_step = result.occupants["first"].steps[0]
        second_step = result.occupants["second"].steps[0]

        self.assertEqual(first_step.queue_wait_time, 0.0)
        self.assertGreater(second_step.queue_wait_time, 0.0)
        self.assertAlmostEqual(second_step.start_time, first_step.end_time)


class FlowRegionDischargeTests(unittest.TestCase):

    # Admission Control V10 -- Storage-Throughput Separation supersedes
    # this class's own pre-V10 assumption (discharge gating shared,
    # unconditionally, across EVERY member edge of a region). The
    # Design Review that produced V10 found that assumption was exactly
    # the mechanism behind Admission Control V9's own regression: an
    # occupant's internal continuation from one member edge to the next
    # re-triggered the SAME shared discharge clock they had themselves
    # just satisfied moments earlier, self-throttling against their own
    # prior admission. V10's fix: throughput applies only when crossing
    # a region's own IDENTIFIED bottleneck edge (FlowRegionCapacityModelV2.
    # bottleneck_edges()) -- an internal continuation across any OTHER
    # member edge never touches the shared clock at all. These tests
    # verify that distinction directly, using two edges of genuinely
    # different width so the bottleneck is unambiguous.

    def setUp(self):

        self.wide_edge = _make_edge("wide", width=4.0, walking_distance=1.0)     # not the bottleneck
        self.narrow_edge = _make_edge("narrow", width=0.91, walking_distance=1.0)  # the true bottleneck

        self.region = FlowRegion(
            id="region-1", edge_ids=("wide", "narrow"), region_kind=FlowRegion.CHAIN,
            total_length=2.0, representative_width=0.91,
            member_edges=(
                FlowRegionMember(edge=self.wide_edge, upstream_node_id="u", downstream_node_id="v"),
                FlowRegionMember(edge=self.narrow_edge, upstream_node_id="v", downstream_node_id="w"),
            ),
        )
        self.flow_region_map = {"wide": self.region, "narrow": self.region}

        self.discharge_model = _FixedRateDischargeModel(rate=0.5)  # min gap = 2.0s

        # A minimal fake engine -- only .graph is read by
        # MultiAgentSimulation's own __init__ (never used by the
        # methods under test here); no real pathfinding is exercised.
        engine = SimpleNamespace(graph=SimpleNamespace())

        self.sim = MultiAgentSimulation(
            engine, capacity_model=FlowRegionCapacityModelV2(), discharge_model=self.discharge_model,
            flow_region_map=self.flow_region_map,
        )

    def test_storage_resolution_is_always_the_edge_itself_never_the_region(self):

        # Design Review correction #1 -- storage is always local.
        admission_object_wide, key_wide = self.sim._resolve_admission(self.wide_edge)
        admission_object_narrow, key_narrow = self.sim._resolve_admission(self.narrow_edge)

        self.assertIs(admission_object_wide, self.wide_edge)
        self.assertEqual(key_wide, self.wide_edge.id)
        self.assertIs(admission_object_narrow, self.narrow_edge)
        self.assertEqual(key_narrow, self.narrow_edge.id)
        self.assertNotEqual(key_wide, key_narrow)

    def test_only_the_narrow_edge_is_identified_as_the_bottleneck(self):

        throughput_object_wide, applies_wide = self.sim._resolve_throughput(self.wide_edge)
        throughput_object_narrow, applies_narrow = self.sim._resolve_throughput(self.narrow_edge)

        self.assertFalse(applies_wide)
        self.assertTrue(applies_narrow)
        self.assertIs(throughput_object_narrow, self.region)

    def test_admission_via_the_non_bottleneck_edge_never_touches_the_shared_clock(self):

        wide_edge_only_sim_route_time = 5.0

        self.assertTrue(self.sim._can_admit(self.wide_edge, None, wide_edge_only_sim_route_time))

        # _can_admit() alone never mutates state -- exercise the actual
        # admission-recording path (_admit_onto_edge()'s own throughput
        # bookkeeping) instead of hand-writing to _last_admission_time.
        _admit_stub_occupant(self.sim, self.wide_edge, wide_edge_only_sim_route_time)

        self.assertNotIn(self.region.id, self.sim._last_admission_time)

    def test_admission_via_the_bottleneck_edge_updates_the_shared_clock(self):

        _admit_stub_occupant(self.sim, self.narrow_edge, 5.0)

        self.assertEqual(self.sim._last_admission_time.get(self.region.id), 5.0)

    def test_an_admission_via_the_bottleneck_blocks_a_subsequent_bottleneck_attempt(self):

        _admit_stub_occupant(self.sim, self.narrow_edge, 10.0)

        # 1s later -- still within the 2.0s gap -- a second attempt
        # through the SAME bottleneck edge must be blocked.
        # to_node=None is safe here -- this sim has no buffer_model, so
        # _can_admit() never dereferences it (Admission Control V7's
        # own buffer check is gated on self.buffer_model is not None).
        self.assertFalse(self.sim._can_admit(self.narrow_edge, None, 11.0))
        self.assertTrue(self.sim._can_admit(self.narrow_edge, None, 12.0))

    def test_an_admission_via_the_bottleneck_never_blocks_the_non_bottleneck_edge(self):

        _admit_stub_occupant(self.sim, self.narrow_edge, 10.0)

        # The wide edge has its own, independent, always-ample storage
        # (StairCapacityModel's own default base_model on a 4.0m-wide
        # door) and is never throughput-gated at all -- the bottleneck's
        # own recent admission must have zero effect on it.
        self.assertTrue(self.sim._can_admit(self.wide_edge, None, 10.5))


class FlowRegionDischargeWithoutBottleneckIdentificationTests(unittest.TestCase):

    # Design Review correction #4's own fail-safe: FlowRegionCapacityModel
    # (V1) exposes no bottleneck_edges() method at all. Pairing it with
    # a discharge_model must never crash and must never guess -- every
    # member edge of a CHAIN/MERGE region simply never has throughput
    # applied to it, reducing to storage-only (per edge) + buffer,
    # exactly the legacy per-edge behavior.

    def test_no_member_edge_is_ever_treated_as_a_bottleneck_without_v2(self):

        edge_a = _make_edge("a", width=4.0, walking_distance=1.0)
        edge_b = _make_edge("b", width=0.91, walking_distance=1.0)

        region = FlowRegion(
            id="region-1", edge_ids=("a", "b"), region_kind=FlowRegion.CHAIN,
            total_length=2.0, representative_width=0.91,
            member_edges=(
                FlowRegionMember(edge=edge_a, upstream_node_id="u", downstream_node_id="v"),
                FlowRegionMember(edge=edge_b, upstream_node_id="v", downstream_node_id="w"),
            ),
        )

        engine = SimpleNamespace(graph=SimpleNamespace())

        sim = MultiAgentSimulation(
            engine, capacity_model=FlowRegionCapacityModel(),
            discharge_model=_FixedRateDischargeModel(rate=0.5),
            flow_region_map={"a": region, "b": region},
        )

        _, applies_a = sim._resolve_throughput(edge_a)
        _, applies_b = sim._resolve_throughput(edge_b)

        self.assertFalse(applies_a)
        self.assertFalse(applies_b)


def _admit_stub_occupant(sim, edge, time):

    # Minimal stand-in for a real registered Occupant, sufficient for
    # _admit_onto_edge()'s own bookkeeping (edge/admission occupancy,
    # throughput clock, congestion-driven duration/timeline entry) --
    # these tests only need the SIDE EFFECTS on shared coordinator
    # state, not a full route/timeline.
    from simulator.occupant import Occupant

    occupant = Occupant(occupant_id=f"stub-{edge.id}-{time}", walking_speed=1.2, route=None, depart_time=time)
    occupant.current_edge_index = 0
    occupant.route = SimpleNamespace(
        nodes=[SimpleNamespace(id="from"), SimpleNamespace(id="to")], edges=[edge],
    )
    sim._occupants[occupant.occupant_id] = occupant
    sim._timelines[occupant.occupant_id] = []
    sim._generation[occupant.occupant_id] = 0

    sim._admit_onto_edge(time, occupant, edge)

    # Immediately vacate the edge's own STORAGE pool (as if this stub
    # occupant had already finished crossing) -- these tests isolate
    # the THROUGHPUT clock's own side effect (_last_admission_time),
    # which _admit_onto_edge() sets regardless of how long the occupant
    # subsequently stays; leaving them "on" the edge would make a
    # second admission attempt fail on STORAGE grounds instead of the
    # discharge-rate gap this test actually exercises.
    sim._admission_occupancy.get(edge.id, set()).discard(occupant.occupant_id)
    sim._edge_occupancy.get(edge.id, set()).discard(occupant.occupant_id)


if __name__ == "__main__":
    unittest.main()
