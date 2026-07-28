import unittest

import networkx as nx
import pandas as pd

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from predictive_dataset.experimental_features_v4 import (
    GRAPH_CONTEXT_FEATURE_NAMES,
    NORMALIZED_FEATURE_NAMES,
    add_graph_context_features,
    add_normalized_features,
    build_graph_context_table,
    compute_graph_context_for_variant,
    variant_structural_constants,
)
from predictive_dataset.topologies_v2 import TopologySpec, build_single_exit_lowrise
from predictive_dataset.topologies_v3 import StructuralVariant, all_structural_variants_v3


# =====================================================
# Cross-Topology Generalization Investigation, Phase 18 -- mechanical
# guards for the new experimental extractor. This module must NEVER
# import LiveRuntime, predictive_dataset/schema.py (the frozen canonical
# schema), Recommendation, or Guidance -- it is a read-only, additive,
# investigation-only extractor (module docstring's own charter).
# =====================================================


class ModuleIsolationGuardTests(unittest.TestCase):
    """Static import-boundary guards -- this experimental module must
    never be wired into production code paths."""

    def test_module_does_not_import_live_runtime(self):

        import predictive_dataset.experimental_features_v4 as mod

        with open(mod.__file__, encoding="utf-8") as f:
            source = f.read()
        self.assertNotIn("live_runtime", source.lower())
        self.assertNotIn("LiveRuntime", source)

    def test_module_does_not_modify_canonical_schema(self):

        import predictive_dataset.experimental_features_v4 as mod

        with open(mod.__file__, encoding="utf-8") as f:
            source = f.read()
        # It may READ topology_signature (already-canonical dataset
        # metadata) but must never import predictive_dataset.schema's
        # CANDIDATE_FEATURE_SCHEMA to mutate/extend it in place.
        self.assertNotIn("CANDIDATE_FEATURE_SCHEMA", source)

    def test_family_id_is_not_among_the_experimental_feature_names(self):
        """The Phase 9 family-ID diagnostic must remain something the
        INVESTIGATION SCRIPT assembles ad hoc for one diagnostic-only
        experiment -- it must never leak into this module's own reusable
        NORMALIZED/GRAPH_CONTEXT feature-name tuples, which is exactly
        what an accidental future production wiring would reuse."""

        for name in NORMALIZED_FEATURE_NAMES + GRAPH_CONTEXT_FEATURE_NAMES:
            self.assertNotIn("family", name.lower())
            self.assertNotIn("variant", name.lower())


class GraphContextHandBuiltGraphTests(unittest.TestCase):
    """Correctness of the graph descriptors on small, hand-built
    buildings whose graph-theoretic properties are known by inspection,
    not just "runs without crashing"."""

    def _linear_chain_building(self) -> Building:
        """lobby -- door1 -- mid -- door2 -- hall -- exit -- OUTSIDE.
        A pure chain: every edge is a bridge (cut-edge), and catchment
        count is asymmetric -- the edge closest to OUTSIDE carries every
        upstream zone's shortest path, the edge closest to the dead-end
        zone carries only that one zone's."""

        floor = Floor(
            id="f1", name="Ground", display_order=0,
            zones=[
                Zone(id="z-lobby", name="Lobby", floor_id="f1", x=0, y=0, width=10, height=10),
                Zone(id="z-mid", name="Mid", floor_id="f1", x=12, y=0, width=10, height=10),
                Zone(id="z-hall", name="Hall", floor_id="f1", x=24, y=0, width=10, height=10),
            ],
            doors=[
                Door(id="door-1", normally_open=True, zone_a_id="z-lobby", zone_b_id="z-mid"),
                Door(id="door-2", normally_open=True, zone_a_id="z-mid", zone_b_id="z-hall"),
            ],
            exits=[Exit(id="exit-1", zone_id="z-hall")],
        )
        return Building(id="chain-building", name="Chain", floors=[floor])

    def test_every_edge_in_a_pure_chain_is_a_bridge(self):

        variant = StructuralVariant("test_family", "test_chain", "test", TopologySpec(
            name="chain", description="", building=self._linear_chain_building(),
            definition=build_single_exit_lowrise().definition, scenario_count=1,
        ))

        rows = compute_graph_context_for_variant(variant)
        self.assertEqual(len(rows), 3)  # door-1, door-2, exit-1
        self.assertTrue(all(r.graph_is_bridge for r in rows))

    def test_catchment_count_reflects_upstream_demand_dependency(self):
        """In a pure chain lobby--door1--mid--door2--hall--exit1--OUTSIDE,
        the edge NEAREST outside is the one every upstream zone's
        shortest path must funnel through: exit-1 carries lobby's, mid's,
        AND hall's own path (count 3); door-2 carries lobby's and mid's
        (count 2, hall doesn't use door-2 -- hall is past it already);
        door-1 carries only lobby's own path (count 1, since it is the
        first edge out of lobby and no other zone's shortest path to
        OUTSIDE revisits it). This is the intended "how much downstream
        demand structurally depends on this candidate" semantic -- a
        bottleneck near the exit is structurally more load-bearing than
        one near a single dead-end zone."""

        variant = StructuralVariant("test_family", "test_chain", "test", TopologySpec(
            name="chain", description="", building=self._linear_chain_building(),
            definition=build_single_exit_lowrise().definition, scenario_count=1,
        ))

        rows = {r.candidate_id: r for r in compute_graph_context_for_variant(variant)}

        self.assertEqual(rows["door-1"].graph_upstream_catchment_count, 1)
        self.assertEqual(rows["door-2"].graph_upstream_catchment_count, 2)
        self.assertEqual(rows["exit-1"].graph_upstream_catchment_count, 3)
        self.assertGreaterEqual(rows["exit-1"].graph_upstream_catchment_count, rows["door-1"].graph_upstream_catchment_count)

    def test_parallel_doors_collapse_onto_one_graph_edge(self):
        """Two doors directly connecting the SAME two zones are a
        documented simplification: they collapse onto one nx graph edge
        and inherit identical centrality/bridge/catchment values."""

        floor = Floor(
            id="f1", name="Ground", display_order=0,
            zones=[
                Zone(id="z-a", name="A", floor_id="f1", x=0, y=0, width=10, height=10),
                Zone(id="z-b", name="B", floor_id="f1", x=12, y=0, width=10, height=10),
            ],
            doors=[
                Door(id="door-1", normally_open=True, zone_a_id="z-a", zone_b_id="z-b"),
                Door(id="door-2", normally_open=True, zone_a_id="z-a", zone_b_id="z-b"),
            ],
            exits=[Exit(id="exit-1", zone_id="z-b")],
        )
        building = Building(id="parallel-building", name="Parallel", floors=[floor])
        variant = StructuralVariant("test_family", "test_parallel", "test", TopologySpec(
            name="parallel", description="", building=building,
            definition=build_single_exit_lowrise().definition, scenario_count=1,
        ))

        rows = {r.candidate_id: r for r in compute_graph_context_for_variant(variant)}

        self.assertEqual(rows["door-1"].graph_is_bridge, rows["door-2"].graph_is_bridge)
        self.assertEqual(rows["door-1"].graph_upstream_catchment_count, rows["door-2"].graph_upstream_catchment_count)

    def test_empty_building_returns_no_rows(self):

        empty_building = Building(id="empty", name="Empty", floors=[
            Floor(id="empty-floor", name="Empty", display_order=0, zones=[]),
        ])
        variant = StructuralVariant("test_family", "test_empty", "test", TopologySpec(
            name="empty", description="", building=empty_building,
            definition=build_single_exit_lowrise().definition, scenario_count=1,
        ))

        self.assertEqual(compute_graph_context_for_variant(variant), ())


class GraphContextAllVariantsTests(unittest.TestCase):
    """The real Dataset V3 registry -- sanity checks, not full recomputation
    of every value (those are validated on hand-built graphs above)."""

    def test_returns_one_row_per_candidate_across_all_sixteen_variants(self):

        variants = all_structural_variants_v3()
        table = build_graph_context_table()

        expected_candidate_count = sum(
            len(compute_graph_context_for_variant(v)) for v in variants
        )
        self.assertEqual(len(table), expected_candidate_count)
        self.assertEqual(set(table.columns), {"structural_variant_id", "candidate_id"} | set(GRAPH_CONTEXT_FEATURE_NAMES))

    def test_betweenness_centrality_is_bounded_zero_to_one(self):

        table = build_graph_context_table()
        self.assertTrue((table["graph_edge_betweenness_centrality"] >= 0.0).all())
        self.assertTrue((table["graph_edge_betweenness_centrality"] <= 1.0).all())

    def test_no_two_variants_are_computed_from_the_same_networkx_object(self):
        """Regression guard: a shared mutable nx.Graph across variants
        would silently corrupt one variant's descriptors with another's
        edges -- each call must build its own fresh graph."""

        variants = all_structural_variants_v3()
        single_exit = next(v for v in variants if v.variant_id == "single_exit_lowrise")
        multi_exit = next(v for v in variants if v.variant_id == "multi_exit_wide")

        rows_a = compute_graph_context_for_variant(single_exit)
        rows_b = compute_graph_context_for_variant(multi_exit)

        ids_a = {r.candidate_id for r in rows_a}
        ids_b = {r.candidate_id for r in rows_b}
        self.assertEqual(ids_a & ids_b, set())


class NormalizedFeaturesTests(unittest.TestCase):

    def test_variant_structural_constants_has_one_row_per_variant(self):

        constants = variant_structural_constants()
        variants = all_structural_variants_v3()
        self.assertEqual(len(constants), len(variants))
        self.assertEqual(set(constants["structural_variant_id"]), {v.variant_id for v in variants})

    def test_add_normalized_features_adds_all_expected_columns(self):

        frame = pd.DataFrame({
            "structural_variant_id": ["single_exit_lowrise", "multi_exit_wide"],
            "candidate_capacity": [10.0, 5.0],
            "candidate_queue_length": [2.0, 1.0],
            "candidate_recent_flow_rate": [1.0, 0.5],
            "candidate_walking_distance": [20.0, 10.0],
            "candidate_alternative_route_count": [1, 2],
            "candidate_adjacent_zone_occupancy": [3.0, 2.0],
            "total_active_occupant_count": [10, 8],
        })

        result = add_normalized_features(frame)

        for name in NORMALIZED_FEATURE_NAMES:
            self.assertIn(name, result.columns)
        self.assertEqual(len(result), len(frame))

    def test_zero_capacity_does_not_raise_divide_by_zero(self):

        frame = pd.DataFrame({
            "structural_variant_id": ["single_exit_lowrise"],
            "candidate_capacity": [0.0],
            "candidate_queue_length": [3.0],
            "candidate_recent_flow_rate": [1.0],
            "candidate_walking_distance": [20.0],
            "candidate_alternative_route_count": [1],
            "candidate_adjacent_zone_occupancy": [2.0],
            "total_active_occupant_count": [5],
        })

        result = add_normalized_features(frame)

        for name in NORMALIZED_FEATURE_NAMES:
            self.assertFalse(pd.isna(result[name].iloc[0]))
            self.assertNotEqual(result[name].iloc[0], float("inf"))

    def test_add_graph_context_features_joins_by_variant_and_candidate(self):

        table = build_graph_context_table()
        sample = table.iloc[0]

        frame = pd.DataFrame({
            "structural_variant_id": [sample["structural_variant_id"]],
            "candidate_id": [sample["candidate_id"]],
        })

        result = add_graph_context_features(frame, table)

        self.assertEqual(result["graph_edge_betweenness_centrality"].iloc[0], sample["graph_edge_betweenness_centrality"])
        self.assertEqual(result["graph_is_bridge"].iloc[0], sample["graph_is_bridge"])

    def test_add_graph_context_features_handles_unknown_candidate_gracefully(self):
        """A row whose (variant, candidate) pair isn't in the graph-
        context table (should never happen in real Dataset V3 data, but
        must not crash) gets safe defaults, not a raised KeyError."""

        frame = pd.DataFrame({
            "structural_variant_id": ["nonexistent_variant"],
            "candidate_id": ["nonexistent_candidate"],
        })

        result = add_graph_context_features(frame, build_graph_context_table())

        self.assertEqual(result["graph_is_bridge"].iloc[0], False)
        self.assertEqual(result["graph_edge_betweenness_centrality"].iloc[0], 0.0)
        self.assertEqual(result["graph_upstream_catchment_count"].iloc[0], 0)


if __name__ == "__main__":
    unittest.main()
