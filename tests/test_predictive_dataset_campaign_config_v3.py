import unittest

from predictive_dataset.campaign_config_v3 import (
    CAMPAIGN_VERSION_V3,
    COVERAGE_TARGETS_V3,
    FULLSCALE_SCENARIOS_PER_VARIANT,
    MASTER_SEED_V3,
    PILOT_SCENARIOS_PER_VARIANT,
    build_campaign_config_v3,
)
from predictive_dataset.target_generator_v2 import TARGET_VERSION_V2
from predictive_dataset.topologies_v3 import all_structural_variants_v3, with_scenario_count


class CampaignConfigV3Tests(unittest.TestCase):

    def test_build_campaign_config_v3_uses_target_v2_by_default(self):

        config = build_campaign_config_v3(all_structural_variants_v3())
        self.assertEqual(config.target_version, TARGET_VERSION_V2)

    def test_variant_ids_match_registry(self):

        variants = all_structural_variants_v3()
        config = build_campaign_config_v3(variants)

        self.assertEqual(set(config.variant_ids), {v.variant_id for v in variants})
        self.assertEqual(len(config.variant_ids), 16)

    def test_scenario_counts_by_variant_reflects_overrides(self):

        variants = all_structural_variants_v3()
        pilot_variants = tuple(
            type(v)(v.family, v.variant_id, v.variant_label, with_scenario_count(v.topology, PILOT_SCENARIOS_PER_VARIANT))
            for v in variants
        )
        config = build_campaign_config_v3(pilot_variants)

        self.assertTrue(all(c == PILOT_SCENARIOS_PER_VARIANT for c in config.scenario_counts_by_variant.values()))

    def test_to_dict_contains_expected_keys(self):

        config = build_campaign_config_v3(all_structural_variants_v3())
        as_dict = config.to_dict()

        for key in (
            "campaign_version", "schema_version", "target_version", "structural_variant_version",
            "master_seed", "tick_dt_seconds", "horizon_seconds", "minimum_end_time_seconds",
            "variant_ids", "scenario_counts_by_variant", "coverage_targets",
        ):
            with self.subTest(key=key):
                self.assertIn(key, as_dict)

    def test_coverage_targets_have_positive_minimum_counts(self):

        for name, target in COVERAGE_TARGETS_V3.items():
            with self.subTest(target=name):
                self.assertGreater(target.minimum_count, 0)
                self.assertTrue(target.description)

    def test_pilot_and_fullscale_constants_are_distinct_and_ordered(self):

        self.assertLess(PILOT_SCENARIOS_PER_VARIANT, FULLSCALE_SCENARIOS_PER_VARIANT)

    def test_default_campaign_version_and_seed(self):

        config = build_campaign_config_v3(all_structural_variants_v3())
        self.assertEqual(config.campaign_version, CAMPAIGN_VERSION_V3)
        self.assertEqual(config.master_seed, MASTER_SEED_V3)


if __name__ == "__main__":
    unittest.main()
