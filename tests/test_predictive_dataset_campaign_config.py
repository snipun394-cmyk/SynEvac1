import json
import unittest

from predictive_dataset.campaign_config import build_campaign_config
from predictive_dataset.versioning import dataset_version


class CampaignConfigTests(unittest.TestCase):

    def test_config_is_reproducible_for_the_same_inputs(self):

        first = build_campaign_config(scenario_count=10, master_seed=42)
        second = build_campaign_config(scenario_count=10, master_seed=42)

        self.assertEqual(first.to_dict(), second.to_dict())

    def test_config_captures_scenario_count_and_seed(self):

        config = build_campaign_config(scenario_count=250, master_seed=7)

        self.assertEqual(config.scenario_count, 250)
        self.assertEqual(config.master_seed, 7)

    def test_config_is_json_serializable(self):

        config = build_campaign_config(scenario_count=10, master_seed=1)

        serialized = json.dumps(config.to_dict())
        self.assertIsInstance(serialized, str)

    def test_distributions_include_occupant_fire_and_engineering_state_diversity(self):

        config = build_campaign_config(scenario_count=10, master_seed=1)
        distributions = config.to_dict()["distributions"]

        for key in (
            "occupant_ranges", "fire_ignition_zone_preference", "fire_growth_parameter_seconds",
            "blocked_door_states", "blocked_exit_states", "stair_availability_states",
        ):
            self.assertIn(key, distributions)
            self.assertTrue(distributions[key])


class DatasetVersionTests(unittest.TestCase):

    def test_dataset_version_carries_the_chosen_horizon(self):

        version = dataset_version(20.0)

        self.assertEqual(version.prediction_horizon_seconds, 20.0)
        self.assertTrue(version.schema_version)
        self.assertTrue(version.campaign_version)
        self.assertTrue(version.target_version)

    def test_dataset_version_is_serializable(self):

        version = dataset_version(30.0)
        serialized = json.dumps(version.to_dict())

        self.assertIsInstance(serialized, str)


if __name__ == "__main__":
    unittest.main()
