import unittest

from types import SimpleNamespace

from navigation.edge import Edge
from navigation.flow_region import FlowRegion
from navigation.flow_region_inference import FlowRegionInferencer
from navigation.graph import NavigationGraph
from navigation.node import Node


def make_graph(nodes, edges):

    graph = NavigationGraph()

    for node in nodes:
        graph.add_node(node)

    for edge in edges:
        graph.add_edge(edge)

    return graph


def outside_node():

    return Node(
        id=Node.OUTSIDE_NODE_ID,
        name="Outside",
        floor_id="",
        node_type=Node.OUTSIDE,
    )


def zone_node(node_id, name=None):

    return Node(
        id=node_id,
        name=name or node_id,
        floor_id="floor-1",
        node_type=Node.ZONE,
    )


class ChainInferenceTests(unittest.TestCase):

    # A pure chain: z1 -> z2 -> z3 -> Outside, every intermediate zone
    # offering exactly one way to continue toward Outside. All three
    # edges (mixed Stair/Exit types) should group into one region.

    def setUp(self):

        self.graph = make_graph(
            nodes=[
                outside_node(),
                zone_node("z1"),
                zone_node("z2"),
                zone_node("z3"),
            ],
            edges=[
                Edge(
                    id="stair-1",
                    edge_type=Edge.STAIR,
                    from_node="z1",
                    to_node="z2",
                    walking_distance=5.0,
                    reference=SimpleNamespace(width=1.2),
                ),
                Edge(
                    id="stair-2",
                    edge_type=Edge.STAIR,
                    from_node="z2",
                    to_node="z3",
                    walking_distance=4.0,
                    reference=SimpleNamespace(width=1.2),
                ),
                Edge(
                    id="exit-1",
                    edge_type=Edge.EXIT,
                    from_node="z3",
                    to_node=Node.OUTSIDE_NODE_ID,
                    walking_distance=2.0,
                    reference=SimpleNamespace(width=1.0),
                ),
            ],
        )

        self.mapping = FlowRegionInferencer.infer(self.graph)

    def test_all_three_edges_share_one_region(self):

        region_ids = {
            self.mapping[edge_id].id
            for edge_id in ("stair-1", "stair-2", "exit-1")
        }

        self.assertEqual(len(region_ids), 1)

    def test_region_kind_is_chain_not_merge(self):

        self.assertEqual(
            self.mapping["stair-1"].region_kind,
            FlowRegion.CHAIN,
        )

    def test_region_edge_ids_are_sorted_and_complete(self):

        self.assertEqual(
            self.mapping["stair-1"].edge_ids,
            tuple(sorted(["stair-1", "stair-2", "exit-1"])),
        )

    def test_aggregate_length_and_width(self):

        region = self.mapping["stair-1"]

        self.assertEqual(region.total_length, 5.0 + 4.0 + 2.0)
        self.assertEqual(region.representative_width, 1.0)

    def test_region_id_derived_from_lowest_sorted_edge_id(self):

        expected_first = sorted(["stair-1", "stair-2", "exit-1"])[0]

        self.assertEqual(
            self.mapping["stair-1"].id,
            f"flow-region-{expected_first}",
        )

    def test_same_region_object_identity_across_member_edges(self):

        self.assertIs(self.mapping["stair-1"], self.mapping["stair-2"])
        self.assertIs(self.mapping["stair-2"], self.mapping["exit-1"])


class MergeInferenceTests(unittest.TestCase):

    # Two stairs (from separate zones/floors) converging into one lobby
    # that has exactly one way out -- the classic "stair chain plus the
    # door/exit it discharges into" shape the whole Option D campaign is
    # aimed at. All three edges should group, and the grouping must be
    # tagged "merge" (more than one edge fed the same node), not "chain".

    def setUp(self):

        self.graph = make_graph(
            nodes=[
                outside_node(),
                zone_node("lobby"),
                zone_node("a"),
                zone_node("b"),
            ],
            edges=[
                Edge(
                    id="stair-a",
                    edge_type=Edge.STAIR,
                    from_node="a",
                    to_node="lobby",
                    walking_distance=6.0,
                ),
                Edge(
                    id="stair-b",
                    edge_type=Edge.STAIR,
                    from_node="b",
                    to_node="lobby",
                    walking_distance=6.0,
                ),
                Edge(
                    id="exit-lobby",
                    edge_type=Edge.EXIT,
                    from_node="lobby",
                    to_node=Node.OUTSIDE_NODE_ID,
                    walking_distance=3.0,
                ),
            ],
        )

        self.mapping = FlowRegionInferencer.infer(self.graph)

    def test_both_stairs_and_the_shared_exit_are_one_region(self):

        region_ids = {
            self.mapping[edge_id].id
            for edge_id in ("stair-a", "stair-b", "exit-lobby")
        }

        self.assertEqual(len(region_ids), 1)

    def test_region_kind_is_merge(self):

        self.assertEqual(
            self.mapping["exit-lobby"].region_kind,
            FlowRegion.MERGE,
        )

    def test_region_contains_exactly_the_three_edges(self):

        self.assertEqual(
            self.mapping["exit-lobby"].edge_ids,
            tuple(sorted(["stair-a", "stair-b", "exit-lobby"])),
        )


class ForkInferenceTests(unittest.TestCase):

    # A stair chain (top -> mid -> bottom) that then forks into two
    # separate exits at "bottom". The chain above the fork must still
    # group together; grouping must not cross the fork in either
    # direction, and each branch past the fork stays its own singleton.

    def setUp(self):

        self.graph = make_graph(
            nodes=[
                outside_node(),
                zone_node("top"),
                zone_node("mid"),
                zone_node("bottom"),
            ],
            edges=[
                Edge(
                    id="stair-top",
                    edge_type=Edge.STAIR,
                    from_node="top",
                    to_node="mid",
                    walking_distance=5.0,
                ),
                Edge(
                    id="stair-mid",
                    edge_type=Edge.STAIR,
                    from_node="mid",
                    to_node="bottom",
                    walking_distance=5.0,
                ),
                Edge(
                    id="exit-1",
                    edge_type=Edge.EXIT,
                    from_node="bottom",
                    to_node=Node.OUTSIDE_NODE_ID,
                    walking_distance=1.0,
                ),
                Edge(
                    id="exit-2",
                    edge_type=Edge.EXIT,
                    from_node="bottom",
                    to_node=Node.OUTSIDE_NODE_ID,
                    walking_distance=1.0,
                ),
            ],
        )

        self.mapping = FlowRegionInferencer.infer(self.graph)

    def test_stair_chain_groups_up_to_the_fork(self):

        self.assertEqual(
            self.mapping["stair-top"].id,
            self.mapping["stair-mid"].id,
        )

        self.assertEqual(
            self.mapping["stair-top"].region_kind,
            FlowRegion.CHAIN,
        )

        self.assertEqual(
            self.mapping["stair-top"].edge_ids,
            tuple(sorted(["stair-top", "stair-mid"])),
        )

    def test_grouping_does_not_cross_the_fork(self):

        self.assertNotEqual(
            self.mapping["stair-mid"].id,
            self.mapping["exit-1"].id,
        )

        self.assertNotEqual(
            self.mapping["stair-mid"].id,
            self.mapping["exit-2"].id,
        )

    def test_both_branches_past_the_fork_are_their_own_singleton_regions(self):

        self.assertEqual(self.mapping["exit-1"].region_kind, FlowRegion.SINGLE)
        self.assertEqual(self.mapping["exit-2"].region_kind, FlowRegion.SINGLE)

        self.assertNotEqual(
            self.mapping["exit-1"].id,
            self.mapping["exit-2"].id,
        )


class IsolatedSingleEdgeInferenceTests(unittest.TestCase):

    def test_lone_edge_becomes_its_own_trivial_region(self):

        graph = make_graph(
            nodes=[outside_node(), zone_node("z")],
            edges=[
                Edge(
                    id="exit-only",
                    edge_type=Edge.EXIT,
                    from_node="z",
                    to_node=Node.OUTSIDE_NODE_ID,
                    walking_distance=3.0,
                    reference=SimpleNamespace(width=0.9),
                ),
            ],
        )

        mapping = FlowRegionInferencer.infer(graph)
        region = mapping["exit-only"]

        self.assertEqual(region.edge_ids, ("exit-only",))
        self.assertEqual(region.region_kind, FlowRegion.SINGLE)
        self.assertEqual(region.id, "flow-region-exit-only")
        self.assertEqual(region.total_length, 3.0)
        self.assertEqual(region.representative_width, 0.9)


class NonTraversableAndAmbiguousEdgeTests(unittest.TestCase):

    def test_locked_door_is_excluded_from_grouping_and_becomes_singleton(self):

        # A locked Door cannot carry any real evacuation flow, so it
        # must not be used to justify grouping anything -- it becomes
        # its own trivial region rather than joining the chain around
        # it, and the still-traversable stair+exit chain groups
        # normally, unaffected by the excluded door upstream.

        graph = make_graph(
            nodes=[
                outside_node(),
                zone_node("z1"),
                zone_node("z2"),
                zone_node("z3"),
            ],
            edges=[
                Edge(
                    id="door-locked",
                    edge_type=Edge.DOOR,
                    from_node="z1",
                    to_node="z2",
                    walking_distance=2.0,
                    reference=SimpleNamespace(locked=True),
                ),
                Edge(
                    id="stair-1",
                    edge_type=Edge.STAIR,
                    from_node="z2",
                    to_node="z3",
                    walking_distance=5.0,
                ),
                Edge(
                    id="exit-1",
                    edge_type=Edge.EXIT,
                    from_node="z3",
                    to_node=Node.OUTSIDE_NODE_ID,
                    walking_distance=1.0,
                ),
            ],
        )

        mapping = FlowRegionInferencer.infer(graph)

        self.assertEqual(mapping["door-locked"].edge_ids, ("door-locked",))
        self.assertEqual(mapping["door-locked"].region_kind, FlowRegion.SINGLE)

        self.assertEqual(mapping["stair-1"].id, mapping["exit-1"].id)

    def test_equal_distance_tie_leaves_edge_ungrouped(self):

        # A "ring corridor": z1 can reach Outside via either of two
        # equally-short doors through z2 or z3. The door directly
        # between z2 and z3 has no well-defined "toward Outside"
        # direction (both ends are equidistant) and must not be forced
        # into a group.

        graph = make_graph(
            nodes=[
                outside_node(),
                zone_node("z1"),
                zone_node("z2"),
                zone_node("z3"),
            ],
            edges=[
                Edge(
                    id="exit-z1",
                    edge_type=Edge.EXIT,
                    from_node="z1",
                    to_node=Node.OUTSIDE_NODE_ID,
                    walking_distance=1.0,
                ),
                Edge(
                    id="door-z1-z2",
                    edge_type=Edge.DOOR,
                    from_node="z1",
                    to_node="z2",
                    walking_distance=1.0,
                ),
                Edge(
                    id="door-z1-z3",
                    edge_type=Edge.DOOR,
                    from_node="z1",
                    to_node="z3",
                    walking_distance=1.0,
                ),
                Edge(
                    id="door-z2-z3",
                    edge_type=Edge.DOOR,
                    from_node="z2",
                    to_node="z3",
                    walking_distance=1.0,
                ),
            ],
        )

        mapping = FlowRegionInferencer.infer(graph)

        self.assertEqual(mapping["door-z2-z3"].edge_ids, ("door-z2-z3",))
        self.assertEqual(mapping["door-z2-z3"].region_kind, FlowRegion.SINGLE)


class MappingCompletenessTests(unittest.TestCase):

    def test_every_edge_gets_a_mapping_and_regions_partition_the_edge_set(self):

        graph = make_graph(
            nodes=[
                outside_node(),
                zone_node("a"),
                zone_node("b"),
                zone_node("lobby"),
            ],
            edges=[
                Edge(
                    id="stair-a",
                    edge_type=Edge.STAIR,
                    from_node="a",
                    to_node="lobby",
                    walking_distance=6.0,
                ),
                Edge(
                    id="stair-b",
                    edge_type=Edge.STAIR,
                    from_node="b",
                    to_node="lobby",
                    walking_distance=6.0,
                ),
                Edge(
                    id="exit-lobby",
                    edge_type=Edge.EXIT,
                    from_node="lobby",
                    to_node=Node.OUTSIDE_NODE_ID,
                    walking_distance=3.0,
                ),
            ],
        )

        mapping = FlowRegionInferencer.infer(graph)
        all_edge_ids = {edge.id for edge in graph.edges}

        self.assertEqual(set(mapping.keys()), all_edge_ids)

        seen = set()

        for region in mapping.values():
            seen.update(region.edge_ids)

        self.assertEqual(seen, all_edge_ids)

    def test_empty_graph_produces_empty_mapping(self):

        graph = NavigationGraph()

        self.assertEqual(FlowRegionInferencer.infer(graph), {})


class NavigationGraphGeneratorIntegrationTests(unittest.TestCase):

    # Confirms FlowRegionInferencer is actually wired into
    # NavigationGraphGenerator.build() (not just correct in isolation),
    # and that doing so left node/edge construction itself completely
    # unchanged -- the regression guarantee Milestone 1 requires.

    def setUp(self):

        from models.building import Building
        from models.door import Door
        from models.exit import Exit
        from models.staircase import Staircase
        from models.zone import Zone

        from navigation.graph_builder import NavigationGraphGenerator

        self.Building = Building
        self.Door = Door
        self.Exit = Exit
        self.Staircase = Staircase
        self.Zone = Zone
        self.NavigationGraphGenerator = NavigationGraphGenerator

    def make_zone(self, name, **kwargs):

        fields = dict(x=0.0, y=0.0, width=2.0, height=2.0)
        fields.update(kwargs)

        return self.Zone(name=name, **fields)

    def test_build_still_produces_the_same_nodes_and_edges_as_before(self):

        building = self.Building(name="B")
        ground = building.create_floor(name="Ground Floor")
        floor1 = building.create_floor(name="Floor 1")

        lobby = self.make_zone("Lobby")
        office = self.make_zone("Office")
        ground.add_zone(lobby)
        ground.add_zone(office)

        upstairs_office = self.make_zone("Upstairs Office")
        floor1.add_zone(upstairs_office)

        door = self.Door(
            name="Lobby-Office Door",
            zone_a_id=lobby.id,
            zone_b_id=office.id,
            floor_id=ground.id,
        )
        ground.add_door(door)

        exit_obj = self.Exit(
            name="Front Exit",
            zone_id=lobby.id,
            floor_id=ground.id,
        )
        ground.add_exit(exit_obj)

        stair = self.Staircase(
            name="Main Stair",
            from_floor_id=ground.id,
            to_floor_id=floor1.id,
            from_zone_id=lobby.id,
            to_zone_id=upstairs_office.id,
        )
        ground.add_stair(stair)

        graph = self.NavigationGraphGenerator().build(building)

        # Unchanged from before this milestone: 4 real nodes (Lobby,
        # Office, Upstairs Office, Outside) and 3 edges (door, exit,
        # stair).
        self.assertEqual(len(graph.nodes), 4)
        self.assertEqual(len(graph.edges), 3)

        # New, additive: every edge has a Flow Region, and the stair
        # groups together with the Front Exit -- Lobby has exactly one
        # way out.
        self.assertEqual(
            set(graph.flow_regions.keys()),
            {door.id, exit_obj.id, stair.id},
        )

        self.assertEqual(
            graph.flow_regions[stair.id].id,
            graph.flow_regions[exit_obj.id].id,
        )

    def test_hand_built_navigation_graph_has_no_flow_regions_by_default(self):

        # NavigationGraph() constructed directly (as many existing
        # tests do, never going through the generator) must not
        # fabricate Flow Region data it never computed.
        self.assertEqual(NavigationGraph().flow_regions, {})


if __name__ == "__main__":
    unittest.main()
