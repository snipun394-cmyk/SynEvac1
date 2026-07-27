import unittest

from predictive_dataset.topologies_v2 import build_single_exit_lowrise
from predictive_dataset.topologies_v3 import StructuralVariant, all_structural_variants_v3
from predictive_dataset.topology_diversity_v3 import compare_signature_sets, structural_diversity_report
from predictive_dataset.topology_signature import compute_all_signatures


class StructuralDiversityReportTests(unittest.TestCase):

    def test_all_sixteen_variants_are_genuinely_distinct(self):
        """The Phase 4 gate this milestone requires BEFORE launching any
        campaign: mechanically proven, not asserted by description."""

        report = structural_diversity_report(all_structural_variants_v3())

        self.assertEqual(report["requested_variant_count"], 16)
        self.assertEqual(report["distinct_structural_signature_count"], 16)
        self.assertEqual(report["duplicate_structural_signature_groups"], [])
        self.assertTrue(report["all_genuinely_distinct"])

    def test_family_distribution_is_four_per_family(self):

        report = structural_diversity_report(all_structural_variants_v3())
        self.assertEqual(
            report["family_distribution"],
            {"single_exit_lowrise": 4, "twin_stair_highrise": 4, "multi_exit_wide": 4, "v1_topology_fixed": 4},
        )

    def test_detects_a_deliberately_injected_duplicate(self):
        """Two StructuralVariants wrapping the SAME underlying Building
        object (only family/variant_id/name differ, not the graph
        itself) MUST be flagged as a duplicate structural signature --
        this is the exact "renamed ids, not real diversity" failure
        mode Phase 4 exists to catch."""

        base = build_single_exit_lowrise()
        renamed = type(base)(
            name="single_exit_lowrise_renamed_clone", description=base.description,
            building=base.building, definition=base.definition, scenario_count=base.scenario_count,
        )

        variants = (
            StructuralVariant("single_exit_lowrise", "single_exit_lowrise", "base", base),
            StructuralVariant("single_exit_lowrise", "single_exit_lowrise_renamed_clone", "clone", renamed),
        )

        report = structural_diversity_report(variants)

        self.assertEqual(report["requested_variant_count"], 2)
        self.assertEqual(report["distinct_structural_signature_count"], 1)
        self.assertFalse(report["all_genuinely_distinct"])
        self.assertEqual(len(report["duplicate_structural_signature_groups"]), 1)
        self.assertEqual(
            set(report["duplicate_structural_signature_groups"][0]["variant_ids"]),
            {"single_exit_lowrise", "single_exit_lowrise_renamed_clone"},
        )

    def test_route_redundancy_distribution_has_min_max_mean(self):

        report = structural_diversity_report(all_structural_variants_v3())
        rr = report["route_redundancy_distribution"]

        self.assertIsNotNone(rr["min"])
        self.assertIsNotNone(rr["max"])
        self.assertIsNotNone(rr["mean"])
        self.assertLessEqual(rr["min"], rr["mean"])
        self.assertLessEqual(rr["mean"], rr["max"])


class CompareSignatureSetsTests(unittest.TestCase):

    def test_v3_spans_a_broader_floor_count_range_than_v2(self):
        """Dataset V2's 4 fixed graphs vs V3's 16 structural variants --
        V3 must span AT LEAST as broad a floor-count range (Phase 14's
        own success criterion: structural distributions materially
        broader, never narrower)."""

        from predictive_dataset.topologies_v2 import all_topology_specs

        v2_signatures = compute_all_signatures(
            StructuralVariant(spec.name, spec.name, "base", spec) for spec in all_topology_specs()
        )
        v3_signatures = compute_all_signatures(all_structural_variants_v3())

        comparison = compare_signature_sets(v2_signatures, v3_signatures)

        floor_spread = comparison["field_spreads"]["floor_count"]
        self.assertGreaterEqual(floor_spread["new"]["max"], floor_spread["baseline"]["max"])
        self.assertGreaterEqual(floor_spread["new"]["distinct_count"], floor_spread["baseline"]["distinct_count"])

        self.assertEqual(comparison["baseline_variant_count"], 4)
        self.assertEqual(comparison["new_variant_count"], 16)


if __name__ == "__main__":
    unittest.main()
