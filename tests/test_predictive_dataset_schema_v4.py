import unittest

from predictive_dataset.schema import CANDIDATE_FEATURE_NAMES, CANDIDATE_FEATURE_SCHEMA, SCHEMA_VERSION
from predictive_dataset.schema_v4 import (
    CANDIDATE_FEATURE_NAMES_V4,
    CANDIDATE_FEATURE_SCHEMA_V4,
    SCHEMA_VERSION_V4,
    field_by_name_v4,
)


# =====================================================
# Predictive Dataset V4 milestone, Phase 2 -- mechanical schema-order/
# version guards. The central invariant: schema.py (Dataset V1/V2/V3's
# frozen schema) must remain byte-for-byte identical to before this
# milestone -- old datasets/models stay loadable and identifiable by
# their original schema version regardless of anything V4 adds.
# =====================================================


class OldSchemaUntouchedTests(unittest.TestCase):

    def test_old_schema_version_is_unchanged(self):
        self.assertEqual(SCHEMA_VERSION, "1.0")

    def test_old_schema_still_has_exactly_nine_fields(self):
        self.assertEqual(len(CANDIDATE_FEATURE_SCHEMA), 9)
        self.assertEqual(len(CANDIDATE_FEATURE_NAMES), 9)

    def test_v4_schema_is_a_strict_superset_starting_with_the_old_schema_in_order(self):
        """The old 9 fields must appear FIRST, in the SAME order, inside
        the new V4 schema -- a V1-schema-aware reader can still read the
        first 9 columns of a V4 dataset correctly."""

        self.assertEqual(CANDIDATE_FEATURE_NAMES_V4[:9], CANDIDATE_FEATURE_NAMES)


class V4SchemaShapeTests(unittest.TestCase):

    def test_v4_schema_version_is_distinct_and_traceable(self):
        self.assertEqual(SCHEMA_VERSION_V4, "4.0")
        self.assertNotEqual(SCHEMA_VERSION_V4, SCHEMA_VERSION)

    def test_v4_schema_has_fifteen_fields(self):
        """9 frozen + 3 promoted V2.1 experimental + 3 promoted graph-context."""
        self.assertEqual(len(CANDIDATE_FEATURE_SCHEMA_V4), 15)
        self.assertEqual(len(CANDIDATE_FEATURE_NAMES_V4), 15)

    def test_v4_schema_names_are_unique(self):
        self.assertEqual(len(CANDIDATE_FEATURE_NAMES_V4), len(set(CANDIDATE_FEATURE_NAMES_V4)))

    def test_promoted_fields_present_in_expected_order(self):

        expected_tail = (
            "candidate_recent_flow_rate", "candidate_congestion_trend", "candidate_alternative_route_count",
            "candidate_betweenness_centrality", "candidate_is_bridge", "candidate_upstream_catchment_count",
        )
        self.assertEqual(CANDIDATE_FEATURE_NAMES_V4[9:], expected_tail)

    def test_field_by_name_v4_resolves_every_promoted_field(self):

        for name in (
            "candidate_recent_flow_rate", "candidate_congestion_trend", "candidate_alternative_route_count",
            "candidate_betweenness_centrality", "candidate_is_bridge", "candidate_upstream_catchment_count",
        ):
            with self.subTest(field=name):
                self.assertEqual(field_by_name_v4(name).name, name)

    def test_field_by_name_v4_raises_for_unknown_field(self):
        with self.assertRaises(KeyError):
            field_by_name_v4("not_a_real_field")

    def test_graph_context_fields_are_never_nullable(self):
        """Unlike candidate_capacity/candidate_walking_distance, all
        three graph-context fields are structurally guaranteed for
        every well-formed candidate (Phase 1's own documented contract)."""

        for name in ("candidate_betweenness_centrality", "candidate_is_bridge", "candidate_upstream_catchment_count"):
            with self.subTest(field=name):
                self.assertFalse(field_by_name_v4(name).nullable)

    def test_topology_family_and_variant_id_never_enter_the_v4_schema(self):
        """Per the Cross-Topology Generalization Investigation's own
        Phase 14 audit: dataset/campaign bookkeeping (which generator
        built a scenario) must never be promoted into a production
        feature schema, regardless of any diagnostic result."""

        for name in CANDIDATE_FEATURE_NAMES_V4:
            self.assertNotIn("family", name.lower())
            self.assertNotIn("variant", name.lower())

    def test_every_field_has_a_sim_live_source_documented(self):
        for field in CANDIDATE_FEATURE_SCHEMA_V4:
            with self.subTest(field=field.name):
                self.assertTrue(field.source)
                self.assertTrue(field.semantic_meaning)


if __name__ == "__main__":
    unittest.main()
