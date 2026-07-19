import shutil
import tempfile
import unittest

from tests.training_dataset_fixtures import make_campaign

import ai_training as at


class RealCampaignTestCase(unittest.TestCase):

    # Shared base for every ai_training test that needs a real,
    # already-exported campaign to consume -- ai_training is a pure
    # consumer, so its own tests must run against a real Campaign
    # Studio export (same real-pipeline fixture training_dataset's own
    # tests use, tests/training_dataset_fixtures.py::make_campaign),
    # never a hand-authored CSV/JSON stand-in that could silently drift
    # from the real artifact shape. Generated ONCE per test class
    # (setUpClass) and shared read-only across every test method in
    # that class, since nothing in ai_training can mutate a loaded
    # TrainingDataset.

    CAMPAIGN_COUNT = 30
    CAMPAIGN_SEED = 7

    @classmethod
    def setUpClass(cls):

        super().setUpClass()

        cls._tmp_dir = tempfile.mkdtemp(prefix="ai_training_test_")
        cls.campaign_summary = make_campaign(
            cls._tmp_dir, count=cls.CAMPAIGN_COUNT, master_seed=cls.CAMPAIGN_SEED,
        )
        cls.dataset = at.load_campaign_dataset(cls._tmp_dir)

    @classmethod
    def tearDownClass(cls):

        shutil.rmtree(cls._tmp_dir, ignore_errors=True)
        super().tearDownClass()
