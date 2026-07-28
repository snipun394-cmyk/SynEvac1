import unittest

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.staircase import Staircase
from models.zone import Zone

from predictive_dataset.candidate import edges_by_candidate_id, enumerate_candidates
from predictive_dataset.graph_context_v4 import compute_graph_context_for_building
from predictive_dataset.live_extractor_v4 import extract_live_v4_candidate_features
from predictive_dataset.simulation_extractor_v2_1 import build_alternative_route_counts
from predictive_dataset.simulation_extractor_v4 import extract_v4_candidate_features


# =====================================================
# Predictive Dataset V4 milestone, Phase 4 -- semantic correctness of
# the PROMOTED graph-context descriptors on small, deterministic,
# hand-built graphs whose expected values are known by inspection, not
# just "networkx ran without crashing". This test file caught a real
# bug during development: a dict-key-ordering mismatch between
# networkx.edge_betweenness_centrality's own (u, v) order and this
# module's sorted lookup key silently zeroed out betweenness for 30 of
# 92 candidate edges across all 16 Dataset V3 structural variants --
# fixed in predictive_dataset/graph_context_v4.py before promotion, and
# every betweenness assertion below is a numeric value checked against
# hand-computed ground truth specifically so a regression cannot hide
# behind "it's just some float >= 0" again.
# =====================================================


class LinearChainTests(unittest.TestCase):
    """hall --door1-- mid --door2-- lobby --exit1-- OUTSIDE. A pure
    3-edge chain: every edge is a bridge, and both betweenness and
    catchment are hand-computable exactly."""

    def _building(self) -> Building:

        floor = Floor(
            id="f1", name="Ground", display_order=0,
            zones=[
                Zone(id="z-hall", name="Hall", floor_id="f1", x=0, y=0, width=10, height=10),
                Zone(id="z-mid", name="Mid", floor_id="f1", x=12, y=0, width=10, height=10),
                Zone(id="z-lobby", name="Lobby", floor_id="f1", x=24, y=0, width=10, height=10),
            ],
            doors=[
                Door(id="door-1", normally_open=True, zone_a_id="z-hall", zone_b_id="z-mid"),
                Door(id="door-2", normally_open=True, zone_a_id="z-mid", zone_b_id="z-lobby"),
            ],
            exits=[Exit(id="exit-1", zone_id="z-lobby")],
        )
        return Building(id="chain", name="Chain", floors=[floor])

    def test_every_edge_is_a_bridge(self):

        ctx = compute_graph_context_for_building(self._building())
        self.assertTrue(all(c.is_bridge for c in ctx.values()))

    def test_betweenness_matches_hand_computed_values(self):
        """3 nodes reachable from OUTSIDE (hall, mid, lobby) -> C(4,2)=6
        node pairs total (hall,mid,lobby,outside). door-1 (hall-mid) lies
        on shortest paths for pairs: (hall,mid),(hall,lobby),(hall,outside)
        = 3/6 = 0.5. door-2 (mid-lobby) lies on: (hall,lobby),(mid,lobby),
        (hall,outside),(mid,outside) = 4/6 = 0.667. exit-1 (lobby-outside)
        lies on: (hall,outside),(mid,outside),(lobby,outside) = 3/6 = 0.5."""

        ctx = compute_graph_context_for_building(self._building())

        self.assertAlmostEqual(ctx["door-1"].betweenness_centrality, 3 / 6, places=6)
        self.assertAlmostEqual(ctx["door-2"].betweenness_centrality, 4 / 6, places=6)
        self.assertAlmostEqual(ctx["exit-1"].betweenness_centrality, 3 / 6, places=6)

    def test_catchment_increases_toward_outside(self):

        ctx = compute_graph_context_for_building(self._building())

        self.assertEqual(ctx["door-1"].upstream_catchment_count, 1)   # only hall's own path
        self.assertEqual(ctx["door-2"].upstream_catchment_count, 2)   # hall's and mid's
        self.assertEqual(ctx["exit-1"].upstream_catchment_count, 3)   # hall's, mid's, AND lobby's


class BranchingGraphTests(unittest.TestCase):
    """A hub zone with 3 spokes (each a dead-end zone with no exit of
    its own) plus one exit ON the hub. Every spoke door is a bridge
    (its zone has no other way out); the hub's own exit is also a
    bridge (only path to OUTSIDE)."""

    def _building(self) -> Building:

        floor = Floor(
            id="f1", name="Ground", display_order=0,
            zones=[
                Zone(id="z-hub", name="Hub", floor_id="f1", x=0, y=0, width=10, height=10),
                Zone(id="z-spoke-a", name="SpokeA", floor_id="f1", x=-20, y=0, width=10, height=10),
                Zone(id="z-spoke-b", name="SpokeB", floor_id="f1", x=20, y=0, width=10, height=10),
                Zone(id="z-spoke-c", name="SpokeC", floor_id="f1", x=0, y=20, width=10, height=10),
            ],
            doors=[
                Door(id="door-a", normally_open=True, zone_a_id="z-hub", zone_b_id="z-spoke-a"),
                Door(id="door-b", normally_open=True, zone_a_id="z-hub", zone_b_id="z-spoke-b"),
                Door(id="door-c", normally_open=True, zone_a_id="z-hub", zone_b_id="z-spoke-c"),
            ],
            exits=[Exit(id="exit-1", zone_id="z-hub")],
        )
        return Building(id="branch", name="Branch", floors=[floor])

    def test_every_edge_is_a_bridge_in_a_pure_tree(self):

        ctx = compute_graph_context_for_building(self._building())
        self.assertTrue(all(c.is_bridge for c in ctx.values()))

    def test_each_spoke_catchment_is_exactly_one(self):
        """Each spoke door only carries its own dead-end zone's path to
        OUTSIDE -- none of the other spokes route through it."""

        ctx = compute_graph_context_for_building(self._building())

        for door_id in ("door-a", "door-b", "door-c"):
            self.assertEqual(ctx[door_id].upstream_catchment_count, 1)

        # exit-1 carries ALL 4 zones' paths (hub's own + all 3 spokes').
        self.assertEqual(ctx["exit-1"].upstream_catchment_count, 4)

    def test_symmetric_star_gives_equal_betweenness_to_every_spoke_and_the_exit(self):
        """5 nodes (hub + 3 spokes + outside) -> C(5,2)=10 pairs. Each
        spoke door lies on exactly 4 pairs (its own zone paired with
        each of the other 4 nodes) = 0.4; the hub's exit is symmetric
        with a spoke in this construction (outside plays the same role
        a 4th spoke would) and also lies on exactly 4 pairs = 0.4. A
        genuinely symmetric star SHOULD give equal betweenness to every
        arm -- this is the correct, hand-computed value, not a
        coincidence to explain away."""

        ctx = compute_graph_context_for_building(self._building())

        for candidate_id in ("door-a", "door-b", "door-c", "exit-1"):
            with self.subTest(candidate=candidate_id):
                self.assertAlmostEqual(ctx[candidate_id].betweenness_centrality, 0.4, places=6)


class RingRedundancyTests(unittest.TestCase):
    """A 3-zone RING (A-B, B-C, C-A, all doors -- a cycle) plus one exit
    hanging off zone A. Ring edges are NOT bridges (redundant); the exit
    edge IS a bridge (sole connection to OUTSIDE) -- the fundamentally
    different route-redundancy structure Dataset V3's linear/branching/
    hub families never captured."""

    def _building(self) -> Building:

        floor = Floor(
            id="f1", name="Ground", display_order=0,
            zones=[
                Zone(id="z-a", name="A", floor_id="f1", x=0, y=0, width=10, height=10),
                Zone(id="z-b", name="B", floor_id="f1", x=20, y=0, width=10, height=10),
                Zone(id="z-c", name="C", floor_id="f1", x=10, y=20, width=10, height=10),
            ],
            doors=[
                Door(id="door-ab", normally_open=True, zone_a_id="z-a", zone_b_id="z-b"),
                Door(id="door-bc", normally_open=True, zone_a_id="z-b", zone_b_id="z-c"),
                Door(id="door-ca", normally_open=True, zone_a_id="z-c", zone_b_id="z-a"),
            ],
            exits=[Exit(id="exit-1", zone_id="z-a")],
        )
        return Building(id="ring", name="Ring", floors=[floor])

    def test_ring_edges_are_not_bridges(self):

        ctx = compute_graph_context_for_building(self._building())

        self.assertFalse(ctx["door-ab"].is_bridge)
        self.assertFalse(ctx["door-bc"].is_bridge)
        self.assertFalse(ctx["door-ca"].is_bridge)

    def test_sole_exit_edge_is_a_bridge(self):

        ctx = compute_graph_context_for_building(self._building())
        self.assertTrue(ctx["exit-1"].is_bridge)

    def test_ring_gives_every_zone_an_alternative_route_unlike_a_chain(self):
        """In this ring, z-b and z-c can each reach z-a (and therefore
        OUTSIDE) two structurally different ways -- door-bc/door-ca vs.
        door-ab -- the defining alternative-route-structure gap the
        Cross-Topology investigation's coverage-gap analysis flagged as
        absent from Dataset V3's linear/branching/hub-only families."""

        candidates = enumerate_candidates(self._building())
        alt_counts = build_alternative_route_counts(candidates)

        # Every ring edge shares a zone with both of the other two ring
        # edges (and, for door-ab/door-ca, with exit-1 too, since they
        # touch z-a) -- genuinely higher local alternative-route density
        # than the chain/branching fixtures above, where each edge only
        # ever shares a zone with its immediate neighbor(s).
        self.assertGreaterEqual(alt_counts["door-bc"], 2)
        self.assertGreaterEqual(alt_counts["door-ca"], 2)


class MultiExitCatchmentTests(unittest.TestCase):
    """Two zones, each with its OWN exit, connected by one door. Catchment
    must correctly split: each zone's shortest path prefers ITS OWN
    (zero-distance-to-outside) exit over routing through the other zone."""

    def _building(self) -> Building:

        floor = Floor(
            id="f1", name="Ground", display_order=0,
            zones=[
                Zone(id="z-left", name="Left", floor_id="f1", x=0, y=0, width=10, height=10),
                Zone(id="z-right", name="Right", floor_id="f1", x=20, y=0, width=10, height=10),
            ],
            doors=[Door(id="door-1", normally_open=True, zone_a_id="z-left", zone_b_id="z-right")],
            exits=[
                Exit(id="exit-left", zone_id="z-left"),
                Exit(id="exit-right", zone_id="z-right"),
            ],
        )
        return Building(id="multi-exit", name="MultiExit", floors=[floor])

    def test_each_exit_only_carries_its_own_zone(self):
        """z-left's shortest path to OUTSIDE is directly via exit-left
        (not through z-right); symmetric for z-right/exit-right. door-1
        carries NEITHER zone's shortest path (each zone has its own,
        strictly shorter, direct exit) -- catchment 0, a genuine,
        honest zero, not a bug."""

        ctx = compute_graph_context_for_building(self._building())

        self.assertEqual(ctx["exit-left"].upstream_catchment_count, 1)
        self.assertEqual(ctx["exit-right"].upstream_catchment_count, 1)
        self.assertEqual(ctx["door-1"].upstream_catchment_count, 0)

    def test_two_independently_exited_zones_form_a_triangle_with_no_bridges(self):
        """z-left--door-1--z-right--exit-right--OUTSIDE--exit-left--z-left
        is a CYCLE: OUTSIDE is a single shared node, so two independently-
        exited zones connected by a door form a triangle through it, not
        two separate pendants. Removing ANY ONE of the 3 edges still
        leaves the graph connected via the other two -- NONE of the 3
        edges are bridges. A real, disclosed structural property of this
        codebase's graph model (every Exit shares the same OUTSIDE node)
        worth documenting precisely because it is easy to assume
        otherwise (a naive "each zone's exit is its only way out"
        intuition would wrongly expect door-1 to be a bridge)."""

        ctx = compute_graph_context_for_building(self._building())
        self.assertFalse(ctx["door-1"].is_bridge)
        self.assertFalse(ctx["exit-left"].is_bridge)
        self.assertFalse(ctx["exit-right"].is_bridge)


class StairConnectedMultiFloorTests(unittest.TestCase):
    """2 floors, one Stair as the ONLY vertical connector. The upper
    floor's zone has no exit of its own -- everything must route down
    through the Stair, exactly like the single-exit chain case, but
    across a floor boundary (Stair, not Door)."""

    def _building(self) -> Building:

        ground = Floor(
            id="floor-ground", name="Ground", display_order=0,
            zones=[Zone(id="z-ground", name="Ground Zone", floor_id="floor-ground", x=0, y=0, width=10, height=10)],
            exits=[Exit(id="exit-1", zone_id="z-ground")],
        )
        upper = Floor(
            id="floor-upper", name="Upper", display_order=1,
            zones=[Zone(id="z-upper", name="Upper Zone", floor_id="floor-upper", x=0, y=0, width=10, height=10)],
            # A Staircase is registered on its OWN (from_floor) floor --
            # from_zone_id must resolve against THIS floor's zones
            # (navigation/graph_builder.py's _add_stair_edges resolves
            # from_zone against the floor iterating floor.stairs, not
            # against from_floor_id/to_floor_id directly).
            stairs=[Staircase(
                id="stair-1", from_zone_id="z-upper", to_zone_id="z-ground",
                from_floor_id="floor-upper", to_floor_id="floor-ground",
            )],
        )
        return Building(id="stair-building", name="StairBuilding", floors=[ground, upper])

    def test_stair_and_exit_are_both_bridges(self):

        ctx = compute_graph_context_for_building(self._building())
        self.assertTrue(ctx["stair-1"].is_bridge)
        self.assertTrue(ctx["exit-1"].is_bridge)

    def test_stair_catchment_includes_upper_floor_zone(self):
        """The upper zone's ONLY path to OUTSIDE crosses the Stair --
        catchment 1 (just its own zone; the ground zone doesn't route
        through the stair to reach its own, already-adjacent exit)."""

        ctx = compute_graph_context_for_building(self._building())
        self.assertEqual(ctx["stair-1"].upstream_catchment_count, 1)
        self.assertEqual(ctx["exit-1"].upstream_catchment_count, 2)  # ground's own path + upper's (via stair)


class SimLiveEquivalenceTests(unittest.TestCase):
    """Phase 3 -- proves simulation and live extraction produce IDENTICAL
    graph-context values for the same Building, since both
    predictive_dataset/simulation_extractor_v4.py and
    predictive_dataset/live_extractor_v4.py delegate to the SAME
    predictive_dataset.graph_context_v4.compute_graph_context_for_building
    function -- not two independently-written implementations that
    could silently drift."""

    def _building(self) -> Building:

        floor = Floor(
            id="f1", name="Ground", display_order=0,
            zones=[
                Zone(id="z-a", name="A", floor_id="f1", x=0, y=0, width=10, height=10),
                Zone(id="z-b", name="B", floor_id="f1", x=20, y=0, width=10, height=10),
            ],
            doors=[Door(id="door-1", normally_open=True, zone_a_id="z-a", zone_b_id="z-b")],
            exits=[Exit(id="exit-1", zone_id="z-b")],
        )
        return Building(id="sim-live", name="SimLive", floors=[floor])

    def test_sim_and_live_extractors_produce_identical_graph_context_values(self):

        building = self._building()
        candidates = enumerate_candidates(building)
        edges = edges_by_candidate_id(building)
        graph_context = compute_graph_context_for_building(building)
        alt_counts = build_alternative_route_counts(candidates)

        class FakeOccupantResult:
            occupants = {}

        class FakeOccupancySnapshot:
            def observation_at(self, node_id):
                class Observation:
                    occupant_count = 0
                return Observation()

        for candidate in candidates:

            edge = edges[candidate.candidate_id]

            sim_features = extract_v4_candidate_features(
                candidate, edge, 10.0,
                building=building, movement_result=FakeOccupantResult(), occupancy_snapshot=FakeOccupancySnapshot(),
                alternative_route_counts=alt_counts, graph_context=graph_context,
            )

            live_features = extract_live_v4_candidate_features(
                candidate, edge, 10.0,
                building=building, crowd_snapshot=None, graph_context=graph_context,
                alternative_route_counts=alt_counts,
            )

            for field in ("candidate_betweenness_centrality", "candidate_is_bridge", "candidate_upstream_catchment_count"):
                with self.subTest(candidate=candidate.candidate_id, field=field):
                    self.assertEqual(sim_features[field], live_features[field])

    def test_both_extractors_import_the_same_underlying_computation_function(self):
        """A static, import-level guard against a future edit
        accidentally forking the two extractors onto separately-written
        graph algorithms."""

        import predictive_dataset.live_extractor_v4 as live_mod
        import predictive_dataset.simulation_extractor_v4 as sim_mod

        self.assertIs(live_mod.CandidateGraphContext, sim_mod.CandidateGraphContext)

        with open(sim_mod.__file__, encoding="utf-8") as f:
            sim_source = f.read()
        with open(live_mod.__file__, encoding="utf-8") as f:
            live_source = f.read()

        self.assertIn("graph_context_v4", sim_source)
        self.assertIn("graph_context_v4", live_source)
        # Neither module defines its own edge_betweenness_centrality/
        # bridges/shortest_path call -- both must delegate, never
        # reimplement.
        self.assertNotIn("edge_betweenness_centrality", sim_source)
        self.assertNotIn("edge_betweenness_centrality", live_source)


if __name__ == "__main__":
    unittest.main()
