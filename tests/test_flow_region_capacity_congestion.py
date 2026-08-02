import unittest

from types import SimpleNamespace

from navigation.edge import Edge
from navigation.flow_region import FlowRegion

from simulator.capacity import CapacityModel, DefaultCapacityModel, StairCapacityModel
from simulator.congestion import (
    CongestionModel,
    DefaultCongestionModel,
    StairAwareCongestionModel,
)
from simulator.flow_region_capacity import FlowRegionCapacityModel
from simulator.flow_region_congestion import FlowRegionCongestionModel


def make_region(total_length=10.0, representative_width=2.0, region_kind=FlowRegion.CHAIN):

    return FlowRegion(
        id="flow-region-test",
        edge_ids=("e1", "e2"),
        region_kind=region_kind,
        total_length=total_length,
        representative_width=representative_width,
    )


class FlowRegionCapacityModelInterfaceTests(unittest.TestCase):

    def test_is_a_capacity_model(self):

        self.assertIsInstance(FlowRegionCapacityModel(), CapacityModel)

    def test_satisfies_the_capacity_model_contract_by_duck_typing(self):

        model = FlowRegionCapacityModel()

        # The one method MultiAgentSimulation actually calls -- must be
        # present and callable with a single positional argument, just
        # like every other CapacityModel.
        self.assertTrue(callable(model.capacity))
        self.assertEqual(model.capacity(make_region()), model.capacity(make_region()))


class FlowRegionCapacityModelRegionFormulaTests(unittest.TestCase):

    def test_capacity_is_area_times_jam_density(self):

        model = FlowRegionCapacityModel()
        region = make_region(total_length=10.0, representative_width=2.0)

        expected = int(10.0 * 2.0 * FlowRegionCapacityModel.JAM_DENSITY_PEOPLE_PER_SQUARE_METER)

        self.assertEqual(model.capacity(region), expected)
        self.assertEqual(model.capacity(region), 40)

    def test_capacity_floored_at_minimum_for_a_tiny_region(self):

        model = FlowRegionCapacityModel()
        region = make_region(total_length=0.1, representative_width=0.1)

        self.assertEqual(model.capacity(region), FlowRegionCapacityModel.MINIMUM_CAPACITY)

    def test_capacity_falls_back_to_minimum_when_length_is_not_derivable(self):

        model = FlowRegionCapacityModel()
        region = make_region(total_length=None, representative_width=2.0)

        self.assertEqual(model.capacity(region), FlowRegionCapacityModel.MINIMUM_CAPACITY)

    def test_capacity_falls_back_to_minimum_when_width_is_not_derivable(self):

        model = FlowRegionCapacityModel()
        region = make_region(total_length=10.0, representative_width=None)

        self.assertEqual(model.capacity(region), FlowRegionCapacityModel.MINIMUM_CAPACITY)

    def test_larger_region_footprint_yields_larger_capacity(self):

        model = FlowRegionCapacityModel()

        small = make_region(total_length=5.0, representative_width=1.0)
        large = make_region(total_length=20.0, representative_width=2.0)

        self.assertLess(model.capacity(small), model.capacity(large))

    def test_region_kind_does_not_affect_the_capacity_formula(self):

        # region_kind is explicitly diagnostic-only (see
        # navigation/flow_region.py) -- it must never change the
        # computed number, only chain vs. merge vs. single bookkeeping.
        model = FlowRegionCapacityModel()

        chain_region = make_region(region_kind=FlowRegion.CHAIN)
        merge_region = make_region(region_kind=FlowRegion.MERGE)
        single_region = make_region(region_kind=FlowRegion.SINGLE)

        self.assertEqual(model.capacity(chain_region), model.capacity(merge_region))
        self.assertEqual(model.capacity(chain_region), model.capacity(single_region))


class FlowRegionCapacityModelEdgeDelegationTests(unittest.TestCase):

    # When given a plain Edge (not a FlowRegion), FlowRegionCapacityModel
    # must behave exactly like today's production capacity model --
    # this is what makes it a safe drop-in candidate for later
    # milestones without changing any existing edge-only behavior.

    def test_door_edge_delegates_to_the_default_base_model_behavior(self):

        edge = Edge(id="d1", edge_type=Edge.DOOR, from_node="a", to_node="b", reference=SimpleNamespace(width=1.4))

        model = FlowRegionCapacityModel()
        base = StairCapacityModel()

        self.assertEqual(model.capacity(edge), base.capacity(edge))

    def test_stair_edge_delegates_to_the_default_base_model_behavior(self):

        edge = Edge(
            id="s1",
            edge_type=Edge.STAIR,
            from_node="a",
            to_node="b",
            walking_distance=8.0,
            reference=SimpleNamespace(width=1.2),
        )

        model = FlowRegionCapacityModel()
        base = StairCapacityModel()

        self.assertEqual(model.capacity(edge), base.capacity(edge))

    def test_custom_base_model_is_used_for_edges_but_never_for_regions(self):

        sentinel = object()

        class FakeBaseModel(CapacityModel):

            def capacity(self, edge):
                return sentinel

        edge = Edge(id="d1", edge_type=Edge.DOOR, from_node="a", to_node="b")

        model = FlowRegionCapacityModel(base_model=FakeBaseModel())

        self.assertIs(model.capacity(edge), sentinel)
        self.assertNotEqual(model.capacity(make_region()), sentinel)

    def test_default_base_model_is_stair_capacity_model(self):

        model = FlowRegionCapacityModel()

        self.assertIsInstance(model.base_model, StairCapacityModel)


class FlowRegionCongestionModelInterfaceTests(unittest.TestCase):

    def test_is_a_congestion_model(self):

        self.assertIsInstance(FlowRegionCongestionModel(), CongestionModel)

    def test_satisfies_the_congestion_model_contract_by_duck_typing(self):

        model = FlowRegionCongestionModel()

        self.assertTrue(callable(model.speed_factor))
        self.assertEqual(
            model.speed_factor(make_region(), 0, 10),
            model.speed_factor(make_region(), 0, 10),
        )


class FlowRegionCongestionModelRegionFormulaTests(unittest.TestCase):

    def test_no_other_occupants_means_full_speed(self):

        model = FlowRegionCongestionModel()
        region = make_region()

        self.assertEqual(model.speed_factor(region, 0, 10), 1.0)

    def test_at_capacity_hits_the_minimum_speed_factor(self):

        model = FlowRegionCongestionModel()
        region = make_region()

        self.assertAlmostEqual(
            model.speed_factor(region, 10, 10),
            FlowRegionCongestionModel.MINIMUM_SPEED_FACTOR,
        )

    def test_partial_occupancy_degrades_linearly(self):

        model = FlowRegionCongestionModel()
        region = make_region()

        expected = 1.0 - 0.5 * (1.0 - FlowRegionCongestionModel.MINIMUM_SPEED_FACTOR)

        self.assertAlmostEqual(model.speed_factor(region, 5, 10), expected)

    def test_zero_capacity_region_hits_the_minimum_speed_factor(self):

        model = FlowRegionCongestionModel()
        region = make_region()

        self.assertEqual(
            model.speed_factor(region, 0, 0),
            FlowRegionCongestionModel.MINIMUM_SPEED_FACTOR,
        )

    def test_opposing_occupants_has_no_effect_on_a_region(self):

        # FlowRegion carries no per-member edge-type breakdown to gate
        # a stair-only counterflow penalty on -- the region formula is
        # deliberately the same simple shape as DefaultCongestionModel,
        # which never reads opposing_occupants either.
        model = FlowRegionCongestionModel()
        region = make_region()

        without_opposing = model.speed_factor(region, 4, 10, opposing_occupants=0)
        with_opposing = model.speed_factor(region, 4, 10, opposing_occupants=5)

        self.assertEqual(without_opposing, with_opposing)


class FlowRegionCongestionModelEdgeDelegationTests(unittest.TestCase):

    def test_door_edge_delegates_to_the_default_base_model_behavior(self):

        edge = Edge(id="d1", edge_type=Edge.DOOR, from_node="a", to_node="b")

        model = FlowRegionCongestionModel()
        base = StairAwareCongestionModel()

        self.assertEqual(
            model.speed_factor(edge, 3, 10),
            base.speed_factor(edge, 3, 10),
        )

    def test_stair_edge_with_counterflow_delegates_to_the_default_base_model_behavior(self):

        edge = Edge(id="s1", edge_type=Edge.STAIR, from_node="a", to_node="b")

        model = FlowRegionCongestionModel()
        base = StairAwareCongestionModel()

        self.assertEqual(
            model.speed_factor(edge, 3, 10, opposing_occupants=2),
            base.speed_factor(edge, 3, 10, opposing_occupants=2),
        )

    def test_stair_edge_counterflow_penalty_still_applies_on_the_edge_path(self):

        # Sanity check that delegation is real, not coincidental: the
        # same stair edge with vs. without counterflow must differ on
        # the Edge path, confirming FlowRegionCongestionModel did not
        # silently swallow the counterflow behavior it delegates to.
        edge = Edge(id="s1", edge_type=Edge.STAIR, from_node="a", to_node="b")

        model = FlowRegionCongestionModel()

        without_opposing = model.speed_factor(edge, 3, 10, opposing_occupants=0)
        with_opposing = model.speed_factor(edge, 3, 10, opposing_occupants=2)

        self.assertGreater(without_opposing, with_opposing)

    def test_custom_base_model_is_used_for_edges_but_never_for_regions(self):

        sentinel = 0.42

        class FakeBaseModel(CongestionModel):

            def speed_factor(self, edge, other_occupants, capacity, opposing_occupants=0):
                return sentinel

        edge = Edge(id="d1", edge_type=Edge.DOOR, from_node="a", to_node="b")

        model = FlowRegionCongestionModel(base_model=FakeBaseModel())

        self.assertEqual(model.speed_factor(edge, 3, 10), sentinel)
        self.assertNotEqual(model.speed_factor(make_region(), 3, 10), sentinel)

    def test_default_base_model_is_stair_aware_congestion_model(self):

        model = FlowRegionCongestionModel()

        self.assertIsInstance(model.base_model, StairAwareCongestionModel)


if __name__ == "__main__":
    unittest.main()
