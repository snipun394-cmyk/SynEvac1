import math
import unittest

from types import SimpleNamespace

from navigation.edge import Edge
from navigation.flow_region import FlowRegion, FlowRegionMember

from simulator.capacity import CapacityModel
from simulator.capacity_candidate_c import CapacityModelCandidateC


def make_edge(edge_id, edge_type=Edge.DOOR, width=1.4, walking_distance=2.0, capacity=None):

    reference = SimpleNamespace(width=width) if capacity is None else SimpleNamespace(width=width, capacity=capacity)

    return Edge(
        id=edge_id, edge_type=edge_type, from_node="a", to_node="b",
        walking_distance=walking_distance, reference=reference,
    )


def member(edge, upstream, downstream):

    return FlowRegionMember(edge=edge, upstream_node_id=upstream, downstream_node_id=downstream)


def make_region(member_edges, region_kind=FlowRegion.CHAIN, total_length=None, representative_width=None):

    edge_ids = tuple(sorted(m.edge.id for m in member_edges))

    return FlowRegion(
        id="flow-region-test",
        edge_ids=edge_ids,
        region_kind=region_kind,
        total_length=total_length,
        representative_width=representative_width,
        member_edges=tuple(member_edges),
    )


def expected_capacity(width, service_time):

    effective_width = max(width - CapacityModelCandidateC.BOUNDARY_LAYER_M, CapacityModelCandidateC.MINIMUM_EFFECTIVE_WIDTH_M)
    discharge_rate = CapacityModelCandidateC.SPECIFIC_FLOW_PERS_PER_S_PER_M * effective_width

    return max(math.ceil(discharge_rate * service_time), CapacityModelCandidateC.MINIMUM_CAPACITY)


class InterfaceTests(unittest.TestCase):

    def test_is_a_capacity_model(self):

        self.assertIsInstance(CapacityModelCandidateC(), CapacityModel)

    def test_explicit_capacity_always_wins_over_the_derived_formula(self):

        edge = make_edge("exit-a", edge_type=Edge.EXIT, width=0.91, walking_distance=3.5, capacity=7)

        self.assertEqual(CapacityModelCandidateC().capacity(edge), 7)


class EdgeFormulaTests(unittest.TestCase):

    # Verifies capacity = ceil(discharge_rate * service_time), where
    # discharge_rate uses the SFPE specific-flow figure on effective
    # (boundary-layer-adjusted) width, and service_time reuses Edge's
    # own already-computed traversal_time -- never re-derived here.

    def test_capacity_matches_the_discharge_rate_times_service_time_formula(self):

        edge = make_edge("d1", width=1.4, walking_distance=2.0)

        expected = expected_capacity(width=1.4, service_time=2.0 / Edge.ASSUMED_WALK_SPEED_M_PER_S)

        self.assertEqual(CapacityModelCandidateC().capacity(edge), expected)
        self.assertEqual(expected, 3)  # hand-computed: ceil(1.645 * 1.6667) = 3

    def test_wider_edge_yields_strictly_higher_capacity_than_a_narrower_one(self):

        narrow = make_edge("d-narrow", width=0.91, walking_distance=3.5)
        wide = make_edge("d-wide", width=4.0, walking_distance=3.5)

        model = CapacityModelCandidateC()

        self.assertLess(model.capacity(narrow), model.capacity(wide))

    def test_longer_edge_yields_strictly_higher_capacity_than_a_shorter_one_of_the_same_width(self):

        # A longer service time means more people can be "in service"
        # (concurrently occupying the edge) for the same discharge
        # rate -- captured directly by the derived-server-count
        # formula, something the old area/mincut formulas had no
        # equivalent notion of.
        short = make_edge("d-short", width=1.4, walking_distance=1.0)
        long_edge = make_edge("d-long", width=1.4, walking_distance=20.0)

        model = CapacityModelCandidateC()

        self.assertLess(model.capacity(short), model.capacity(long_edge))

    def test_width_none_falls_back_to_minimum_capacity(self):

        edge = make_edge("d1", width=None, walking_distance=2.0)

        self.assertEqual(CapacityModelCandidateC().capacity(edge), CapacityModelCandidateC.MINIMUM_CAPACITY)

    def test_walking_distance_none_falls_back_to_minimum_capacity(self):

        edge = make_edge("d1", width=1.4, walking_distance=None)

        self.assertEqual(CapacityModelCandidateC().capacity(edge), CapacityModelCandidateC.MINIMUM_CAPACITY)

    def test_effective_width_is_floored_rather_than_going_negative(self):

        # width (0.05) is narrower than the boundary-layer subtraction
        # (0.15) alone -- effective_width must clamp to
        # MINIMUM_EFFECTIVE_WIDTH_M rather than produce a negative
        # (and therefore nonsensical) discharge_rate.
        edge = make_edge("d1", width=0.05, walking_distance=100.0)

        expected = expected_capacity(width=0.05, service_time=100.0 / Edge.ASSUMED_WALK_SPEED_M_PER_S)

        self.assertEqual(CapacityModelCandidateC().capacity(edge), expected)
        self.assertGreater(expected, CapacityModelCandidateC.MINIMUM_CAPACITY)

    def test_capacity_is_never_below_the_minimum_floor(self):

        edge = make_edge("d1", width=0.05, walking_distance=0.1)

        self.assertGreaterEqual(CapacityModelCandidateC().capacity(edge), CapacityModelCandidateC.MINIMUM_CAPACITY)


class RegionFormulaTests(unittest.TestCase):

    # Unlike V1 (area x jam-density) and V2 (min-cut/max-flow), this
    # model reduces a region to exactly the same two-quantity formula
    # as a plain edge, using the region's own aggregate
    # representative_width/total_length -- CHAIN and MERGE are
    # deliberately NOT special-cased.

    def test_region_capacity_matches_the_same_formula_using_aggregate_fields(self):

        region = make_region([], region_kind=FlowRegion.CHAIN, total_length=10.0, representative_width=1.27)

        expected = expected_capacity(width=1.27, service_time=10.0 / Edge.ASSUMED_WALK_SPEED_M_PER_S)

        self.assertEqual(CapacityModelCandidateC().capacity(region), expected)

    def test_chain_and_merge_regions_with_identical_aggregate_fields_give_identical_capacity(self):

        chain_region = make_region([], region_kind=FlowRegion.CHAIN, total_length=15.0, representative_width=1.0)
        merge_region = make_region([], region_kind=FlowRegion.MERGE, total_length=15.0, representative_width=1.0)

        model = CapacityModelCandidateC()

        self.assertEqual(model.capacity(chain_region), model.capacity(merge_region))

    def test_region_with_no_total_length_falls_back_to_minimum(self):

        region = make_region([], region_kind=FlowRegion.SINGLE, total_length=None, representative_width=1.27)

        self.assertEqual(CapacityModelCandidateC().capacity(region), CapacityModelCandidateC.MINIMUM_CAPACITY)

    def test_region_with_no_representative_width_falls_back_to_minimum(self):

        region = make_region([], region_kind=FlowRegion.SINGLE, total_length=10.0, representative_width=None)

        self.assertEqual(CapacityModelCandidateC().capacity(region), CapacityModelCandidateC.MINIMUM_CAPACITY)

    def test_door_edge_never_delegates_to_a_base_model_unlike_v1_v2(self):

        # Candidate C has no base_model at all -- it replaces the
        # formula everywhere, not just inside Flow Regions.
        self.assertFalse(hasattr(CapacityModelCandidateC(), "base_model"))


class RealBuildingRegressionTests(unittest.TestCase):

    # End-to-end against the real, already-committed NIST recreations
    # -- not synthetic fixtures -- confirming the formula is actually
    # reachable through a real FlowRegion's own aggregate fields.

    def test_10story_chain_region_capacity_matches_the_formula_on_its_own_aggregate_fields(self):

        import sys
        sys.path.insert(0, "scripts")
        from run_nist_10story_validation import build_nist_10story_building

        from navigation.graph_builder import NavigationGraphGenerator

        building = build_nist_10story_building()
        graph = NavigationGraphGenerator().build(building)

        region = graph.flow_regions["stair-a-2"]
        self.assertEqual(region.region_kind, FlowRegion.CHAIN)

        result = CapacityModelCandidateC().capacity(region)

        expected = expected_capacity(
            width=region.representative_width,
            service_time=region.total_length / Edge.ASSUMED_WALK_SPEED_M_PER_S,
        )

        self.assertEqual(result, expected)
        self.assertGreaterEqual(result, CapacityModelCandidateC.MINIMUM_CAPACITY)

    def test_18story_merge_region_capacity_matches_the_formula_on_its_own_aggregate_fields(self):

        import sys
        sys.path.insert(0, "scripts")
        from run_nist_18story_complete_merge_validation import build_nist_18story_complete_building

        from navigation.graph_builder import NavigationGraphGenerator

        building = build_nist_18story_complete_building()
        graph = NavigationGraphGenerator().build(building)

        region = graph.flow_regions["door-7-lobby"]
        self.assertEqual(region.region_kind, FlowRegion.MERGE)

        result = CapacityModelCandidateC().capacity(region)

        expected = expected_capacity(
            width=region.representative_width,
            service_time=region.total_length / Edge.ASSUMED_WALK_SPEED_M_PER_S,
        )

        self.assertEqual(result, expected)
        self.assertGreaterEqual(result, CapacityModelCandidateC.MINIMUM_CAPACITY)


if __name__ == "__main__":
    unittest.main()
