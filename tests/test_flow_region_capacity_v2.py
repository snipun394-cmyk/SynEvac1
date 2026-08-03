import unittest

from types import SimpleNamespace

from navigation.edge import Edge
from navigation.flow_region import FlowRegion, FlowRegionMember

from simulator.capacity import CapacityModel, StairCapacityModel
from simulator.flow_region_capacity import FlowRegionCapacityModelV2


def make_edge(edge_id, edge_type=Edge.STAIR, width=1.0, walking_distance=1.0, capacity=None):

    reference = SimpleNamespace(width=width) if capacity is None else SimpleNamespace(width=width, capacity=capacity)

    return Edge(
        id=edge_id, edge_type=edge_type, from_node="a", to_node="b",
        walking_distance=walking_distance, reference=reference,
    )


def member(edge, upstream, downstream):

    return FlowRegionMember(edge=edge, upstream_node_id=upstream, downstream_node_id=downstream)


def make_region(member_edges, region_kind=FlowRegion.CHAIN):

    edge_ids = tuple(sorted(m.edge.id for m in member_edges))

    return FlowRegion(
        id="flow-region-test",
        edge_ids=edge_ids,
        region_kind=region_kind,
        total_length=None,
        representative_width=None,
        member_edges=tuple(member_edges),
    )


class InterfaceTests(unittest.TestCase):

    def test_is_a_capacity_model(self):

        self.assertIsInstance(FlowRegionCapacityModelV2(), CapacityModel)

    def test_default_base_model_is_stair_capacity_model(self):

        self.assertIsInstance(FlowRegionCapacityModelV2().base_model, StairCapacityModel)

    def test_door_edge_delegates_to_the_base_model(self):

        edge = make_edge("d1", edge_type=Edge.DOOR, width=1.4)

        model = FlowRegionCapacityModelV2()
        base = StairCapacityModel()

        self.assertEqual(model.capacity(edge), base.capacity(edge))

    def test_custom_base_model_feeds_every_member_edge_of_a_region(self):

        # A multi-member region's own capacity is a COMBINATION (min-
        # cut) of each member's own base_model-derived capacity, not a
        # single passthrough -- for a two-edge chain both fed a flat 5
        # by the fake base model, min-cut must equal that flat 5 (the
        # narrowest, and only, value), proving the fake model was
        # actually consulted per member rather than bypassed.
        class FakeBaseModel(CapacityModel):

            def capacity(self, edge):
                return 5

        edge = make_edge("d1", edge_type=Edge.DOOR)
        region = make_region(
            [member(make_edge("s1"), "u", "v"), member(make_edge("s2"), "v", "w")],
        )

        model = FlowRegionCapacityModelV2(base_model=FakeBaseModel())

        self.assertEqual(model.capacity(edge), 5)
        self.assertEqual(model.capacity(region), 5)


class EmptyAndSingleMemberTests(unittest.TestCase):

    def test_region_with_no_orientable_members_falls_back_to_minimum(self):

        region = make_region([], region_kind=FlowRegion.SINGLE)

        self.assertEqual(FlowRegionCapacityModelV2().capacity(region), FlowRegionCapacityModelV2.MINIMUM_CAPACITY)

    def test_single_member_region_equals_that_edges_own_capacity(self):

        edge = make_edge("s1", width=4.0, walking_distance=6.8)
        region = make_region([member(edge, "top", "bottom")], region_kind=FlowRegion.SINGLE)

        model = FlowRegionCapacityModelV2()

        self.assertEqual(model.capacity(region), model.base_model.capacity(edge))


class ChainMinCutTests(unittest.TestCase):

    # A pure serial chain: z1 -> e1 -> z2 -> e2 -> z3 -> e3 -> z4.
    # Deliberately different per-edge widths so the three capacities
    # differ -- min-cut must equal the single narrowest edge, never a
    # sum or an average.

    def test_capacity_equals_the_narrowest_single_edge(self):

        e1 = make_edge("e1", width=4.0, walking_distance=6.8)   # wide
        e2 = make_edge("e2", width=1.27, walking_distance=6.8)  # narrow -- the real bottleneck
        e3 = make_edge("e3", width=4.0, walking_distance=6.8)   # wide

        members = [
            member(e1, "z1", "z2"),
            member(e2, "z2", "z3"),
            member(e3, "z3", "z4"),
        ]
        region = make_region(members, region_kind=FlowRegion.CHAIN)

        model = FlowRegionCapacityModelV2()

        expected = model.base_model.capacity(e2)

        self.assertEqual(model.capacity(region), expected)
        self.assertLess(model.capacity(region), model.base_model.capacity(e1))

    def test_nine_edge_chain_matches_the_real_synevac_bug_scenario(self):

        # Reproduces the exact Milestone 5 finding: 8 stair flights
        # (capacity 1 each under StairCapacityModel, width 1.27,
        # walking_distance ~6.8-8.4) plus a wide exit -- V1 gave this
        # exact shape a capacity of 121; V2 must give it 1.

        chain = []
        previous_node = "z10"

        for floor in range(9, 1, -1):

            edge = make_edge(f"stair-{floor}", width=1.27, walking_distance=6.8)
            node = f"z{floor}"
            chain.append(member(edge, previous_node, node))
            previous_node = node

        exit_edge = make_edge("exit-a", edge_type=Edge.EXIT, width=0.91, walking_distance=3.5)
        chain.append(member(exit_edge, previous_node, "outside"))

        region = make_region(chain, region_kind=FlowRegion.CHAIN)
        model = FlowRegionCapacityModelV2()

        self.assertEqual(model.capacity(region), 1)


class MergeMinCutTests(unittest.TestCase):

    # Three independent chains (a, b, c) converging on one shared node,
    # discharging through one shared exit edge.

    def _build(self, branch_widths, discharge_width):

        members = []

        for name, width in branch_widths.items():
            edge = make_edge(f"stair-{name}", width=width, walking_distance=6.8)
            members.append(member(edge, f"top-{name}", "lobby"))

        exit_edge = make_edge("exit-lobby", edge_type=Edge.EXIT, width=discharge_width, walking_distance=3.5)
        members.append(member(exit_edge, "lobby", "outside"))

        return make_region(members, region_kind=FlowRegion.MERGE), members

    def test_capacity_capped_by_the_shared_discharge_when_branches_are_wider(self):

        # Three wide branches (ample capacity each), one narrow shared
        # exit -- the exit's own capacity is the true limiting cut.
        region, members = self._build({"a": 4.0, "b": 4.0, "c": 4.0}, discharge_width=0.91)
        model = FlowRegionCapacityModelV2()

        exit_edge = next(m.edge for m in members if m.edge.id == "exit-lobby")
        expected = model.base_model.capacity(exit_edge)

        self.assertEqual(model.capacity(region), expected)

    def test_capacity_capped_by_the_sum_of_branches_when_discharge_is_wider(self):

        # Three narrow branches, one very wide shared exit -- the exit
        # can never receive more than what the branches can each
        # deliver, so the sum of the (narrow) branches governs, not the
        # (wide) exit.
        region, members = self._build({"a": 1.27, "b": 1.27, "c": 1.27}, discharge_width=8.0)
        model = FlowRegionCapacityModelV2()

        branch_edges = [m.edge for m in members if m.edge.id != "exit-lobby"]
        expected_sum = sum(model.base_model.capacity(e) for e in branch_edges)
        exit_edge = next(m.edge for m in members if m.edge.id == "exit-lobby")
        expected_discharge = model.base_model.capacity(exit_edge)

        self.assertLess(expected_sum, expected_discharge)
        self.assertEqual(model.capacity(region), expected_sum)

    def test_merge_capacity_is_never_less_than_a_pure_area_sum_would_be_wrong_in_the_other_direction(self):

        # Sanity check on the theorem itself: capacity must never exceed
        # EITHER the discharge capacity or the branch sum individually.
        region, members = self._build({"a": 1.27, "b": 4.0, "c": 1.27}, discharge_width=2.0)
        model = FlowRegionCapacityModelV2()

        exit_edge = next(m.edge for m in members if m.edge.id == "exit-lobby")
        branch_edges = [m.edge for m in members if m.edge.id != "exit-lobby"]

        discharge_capacity = model.base_model.capacity(exit_edge)
        branch_sum = sum(model.base_model.capacity(e) for e in branch_edges)

        result = model.capacity(region)

        self.assertLessEqual(result, discharge_capacity)
        self.assertLessEqual(result, branch_sum)
        self.assertEqual(result, min(discharge_capacity, branch_sum))


class RealBuildingRegressionTests(unittest.TestCase):

    # End-to-end against the actual completed 18-story merge building
    # (Stairs 3, 7, 12 -> shared lobby -> exit-lobby) and the actual
    # 10-story chain building -- not synthetic fixtures, the real
    # NIST recreations already committed to this repo.

    def test_18story_merge_region_capacity_is_capped_correctly(self):

        import sys
        sys.path.insert(0, "scripts")
        from run_nist_18story_complete_merge_validation import build_nist_18story_complete_building

        from navigation.graph_builder import NavigationGraphGenerator

        building = build_nist_18story_complete_building()
        graph = NavigationGraphGenerator().build(building)

        region = graph.flow_regions["door-7-lobby"]
        self.assertEqual(region.region_kind, FlowRegion.MERGE)

        model = FlowRegionCapacityModelV2()
        result = model.capacity(region)

        # Independently compute the expected min-cut by hand from the
        # same real edges, to avoid the test just re-running the
        # production code under test.
        edges_by_id = {m.edge.id: m.edge for m in region.member_edges}
        exit_edge = edges_by_id["exit-lobby"]
        discharge_capacity = model.base_model.capacity(exit_edge)

        branch_capacity_sum = 0
        for prefix in ("3", "7", "12"):
            door_edge = edges_by_id[f"door-{prefix}-lobby"]
            branch_capacity_sum += model.base_model.capacity(door_edge)

        expected = min(discharge_capacity, branch_capacity_sum)

        self.assertEqual(result, expected)
        self.assertGreater(result, 0)

    def test_10story_chain_region_capacity_equals_the_narrowest_stair(self):

        import sys
        sys.path.insert(0, "scripts")
        from run_nist_10story_validation import build_nist_10story_building

        from navigation.graph_builder import NavigationGraphGenerator

        building = build_nist_10story_building()
        graph = NavigationGraphGenerator().build(building)

        region = graph.flow_regions["stair-a-2"]
        self.assertEqual(region.region_kind, FlowRegion.CHAIN)

        model = FlowRegionCapacityModelV2()
        result = model.capacity(region)

        per_edge_minimum = min(
            model.base_model.capacity(m.edge) for m in region.member_edges
        )

        self.assertEqual(result, per_edge_minimum)
        self.assertEqual(result, 1)  # the exact Milestone 5 bottleneck value


if __name__ == "__main__":
    unittest.main()
