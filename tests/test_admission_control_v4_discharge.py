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
from simulator.flow_region_capacity import FlowRegionCapacityModel


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

    # Discharge gating applies at the REGION level (shared across
    # member edges) when a flow_region_map is supplied, exactly like
    # capacity_model/congestion_model already do. _resolve_admission()
    # itself is entirely unmodified by this milestone (already
    # exhaustively covered by the 102 pre-existing, still-passing
    # admission-control/flow-region tests) -- what these tests verify
    # directly is that discharge state (_last_admission_time,
    # _pending_retry) lives on that SAME admission_key a region
    # resolves to, so two different member edges of one region
    # genuinely share one discharge gate rather than each getting their
    # own.

    def setUp(self):

        self.edge_a = _make_edge("a", width=4.0, walking_distance=1.0)
        self.edge_b = _make_edge("b", width=4.0, walking_distance=1.0)

        self.region = FlowRegion(
            id="region-1", edge_ids=("a", "b"), region_kind=FlowRegion.CHAIN,
            total_length=2.0, representative_width=4.0,
            member_edges=(
                FlowRegionMember(edge=self.edge_a, upstream_node_id="u", downstream_node_id="v"),
                FlowRegionMember(edge=self.edge_b, upstream_node_id="v", downstream_node_id="w"),
            ),
        )
        flow_region_map = {"a": self.region, "b": self.region}

        self.discharge_model = _FixedRateDischargeModel(rate=0.5)  # min gap = 2.0s, region-wide

        # A minimal fake engine -- only .graph is read by
        # MultiAgentSimulation's own __init__ (never used by the
        # methods under test here); no real pathfinding is exercised.
        engine = SimpleNamespace(graph=SimpleNamespace())

        self.sim = MultiAgentSimulation(
            engine, capacity_model=FlowRegionCapacityModel(), discharge_model=self.discharge_model,
            flow_region_map=flow_region_map,
        )

    def test_both_member_edges_resolve_to_the_same_admission_key(self):

        _, key_a = self.sim._resolve_admission(self.edge_a)
        _, key_b = self.sim._resolve_admission(self.edge_b)

        self.assertEqual(key_a, key_b)
        self.assertEqual(key_a, self.region.id)

    def test_admission_object_handed_to_discharge_model_is_the_region_not_the_edge(self):

        admission_object, _ = self.sim._resolve_admission(self.edge_a)

        self.assertIs(admission_object, self.region)

    def test_an_admission_recorded_via_edge_a_blocks_a_subsequent_attempt_via_edge_b(self):

        admission_object_a, admission_key = self.sim._resolve_admission(self.edge_a)
        admission_object_b, _ = self.sim._resolve_admission(self.edge_b)

        # Simulate an admission having just happened through edge_a's
        # own member of the region.
        self.sim._last_admission_time[admission_key] = 10.0

        # An attempt through edge_b, the OTHER member edge, 1s later --
        # still within the 2.0s region-wide gap -- must be blocked,
        # proving the two edges share one discharge gate rather than
        # each tracking their own.
        self.assertFalse(self.sim._can_admit(admission_object_b, admission_key, 11.0))
        self.assertTrue(self.sim._can_admit(admission_object_b, admission_key, 12.0))


if __name__ == "__main__":
    unittest.main()
