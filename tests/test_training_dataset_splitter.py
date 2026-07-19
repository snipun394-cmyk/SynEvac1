import shutil
import tempfile
import unittest

from training_dataset.loader import load_campaign
from training_dataset.splitter import split_dataset, stratify_by_fire_profile

from tests.training_dataset_fixtures import make_campaign


class _TempOutputDir:

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="training_dataset_splitter_test_")
        return self.path

    def __exit__(self, *args):
        shutil.rmtree(self.path, ignore_errors=True)


class DeterministicSplitTests(unittest.TestCase):

    def test_same_master_seed_gives_the_same_split(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=10, master_seed=5)
            dataset = load_campaign(output_dir)

            first = split_dataset(dataset, master_seed=99)
            second = split_dataset(dataset, master_seed=99)

            self.assertEqual(first, second)

    def test_a_different_master_seed_can_give_a_different_split(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=10, master_seed=5)
            dataset = load_campaign(output_dir)

            split_a = split_dataset(dataset, master_seed=1)
            split_b = split_dataset(dataset, master_seed=2)

            self.assertNotEqual(split_a.to_dict(), split_b.to_dict())

    def test_split_is_stable_regardless_of_input_sample_order(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=8, master_seed=5)
            dataset = load_campaign(output_dir)

            forward = split_dataset(dataset, master_seed=42)
            reversed_samples = list(dataset)[::-1]
            backward = split_dataset(reversed_samples, master_seed=42)

            self.assertEqual(forward, backward)


class NoOverlapTests(unittest.TestCase):

    def test_every_scenario_id_appears_in_exactly_one_split(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=12, master_seed=3)
            dataset = load_campaign(output_dir)

            split = split_dataset(dataset, master_seed=17)

            train = set(split.train_ids)
            validation = set(split.validation_ids)
            test = set(split.test_ids)

            self.assertEqual(train & validation, set())
            self.assertEqual(train & test, set())
            self.assertEqual(validation & test, set())

            self.assertEqual(
                train | validation | test, set(dataset.scenario_ids),
            )

    def test_split_for_reports_the_correct_bucket(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=6, master_seed=3)
            dataset = load_campaign(output_dir)

            split = split_dataset(dataset, master_seed=17)

            for scenario_id in split.train_ids:
                self.assertEqual(split.split_for(scenario_id), "train")

            for scenario_id in split.validation_ids:
                self.assertEqual(split.split_for(scenario_id), "validation")

            for scenario_id in split.test_ids:
                self.assertEqual(split.split_for(scenario_id), "test")

            self.assertIsNone(split.split_for("scn-not-in-this-campaign"))


class SplitSizeTests(unittest.TestCase):

    def test_fractions_must_sum_to_one(self):

        with self.assertRaises(ValueError):
            split_dataset(
                [], master_seed=1, train_fraction=0.5, validation_fraction=0.4, test_fraction=0.2,
            )

    def test_custom_fractions_are_respected_at_scale(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=20, master_seed=9)
            dataset = load_campaign(output_dir)

            split = split_dataset(
                dataset, master_seed=1,
                train_fraction=0.5, validation_fraction=0.25, test_fraction=0.25,
            )

            self.assertEqual(
                len(split.train_ids) + len(split.validation_ids) + len(split.test_ids),
                len(dataset),
            )
            # Roughly 10/5/5 -- allow rounding slack.
            self.assertAlmostEqual(len(split.train_ids), 10, delta=1)
            self.assertAlmostEqual(len(split.validation_ids), 5, delta=1)
            self.assertAlmostEqual(len(split.test_ids), 5, delta=1)

    def test_all_train_fraction_puts_everything_in_train(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=5, master_seed=9)
            dataset = load_campaign(output_dir)

            split = split_dataset(
                dataset, master_seed=1, train_fraction=1.0, validation_fraction=0.0, test_fraction=0.0,
            )

            self.assertEqual(set(split.train_ids), set(dataset.scenario_ids))
            self.assertEqual(split.validation_ids, ())
            self.assertEqual(split.test_ids, ())


class StratificationTests(unittest.TestCase):

    def test_stratified_split_still_covers_every_scenario_exactly_once(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=15, master_seed=11)
            dataset = load_campaign(output_dir)

            split = split_dataset(
                dataset, master_seed=4, stratify_by=stratify_by_fire_profile,
            )

            all_ids = set(split.train_ids) | set(split.validation_ids) | set(split.test_ids)
            self.assertEqual(all_ids, set(dataset.scenario_ids))
            self.assertEqual(
                len(split.train_ids) + len(split.validation_ids) + len(split.test_ids),
                len(dataset),
            )


if __name__ == "__main__":
    unittest.main()
