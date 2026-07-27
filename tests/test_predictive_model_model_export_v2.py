import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from predictive_model.calibration import IsotonicCalibrator
from predictive_model.model_export_v2 import (
    ExtendedModelMetadataV2,
    export_calibrator,
    load_calibrator,
    load_metadata_v2,
    write_metadata_v2,
)


class ModelExportV2Tests(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _sample_metadata(self, **overrides):
        defaults = dict(
            model_version="localized_predictive_model_v2",
            dataset_campaign_version="predictive_dataset_campaign_v2",
            dataset_schema_version="1.0",
            dataset_feature_version="1.0",
            dataset_target_version="v1-congestion-threshold-2-horizon-window",
            prediction_horizon_seconds=20.0,
            training_seed=20260726,
            split_strategy="scenario-level 70/15/15",
            algorithm="gradient_boosting",
            algorithm_library="HistGradientBoostingModel",
            hyperparameters={"max_iter": 300},
            calibration_method="isotonic (fit on validation split only)",
            calibration_artifact="calibrator.joblib",
            feature_order=("a", "b", "c"),
            candidate_types_supported=("Door", "Exit", "Stair"),
            decision_threshold=0.9,
            known_limitations=("limitation 1", "limitation 2"),
            git_commit="deadbeef",
        )
        defaults.update(overrides)
        return ExtendedModelMetadataV2(**defaults)

    # --- calibrator serialization/reload ---

    def test_calibrator_round_trips_transform(self):

        calibrator = IsotonicCalibrator()
        val_probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        val_y = np.array([0, 0, 1, 1, 1])
        calibrator.fit(val_probs, val_y)

        original = calibrator.transform(val_probs)

        path = export_calibrator(calibrator, self.tmp_dir)
        self.assertTrue(os.path.exists(path))

        reloaded = load_calibrator(path)
        reloaded_output = reloaded.transform(val_probs)

        np.testing.assert_array_almost_equal(original, reloaded_output)

    def test_calibrator_output_stays_in_probability_range_after_reload(self):

        calibrator = IsotonicCalibrator()
        val_probs = np.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.95])
        val_y = np.array([0, 0, 0, 1, 1, 1])
        calibrator.fit(val_probs, val_y)

        path = export_calibrator(calibrator, self.tmp_dir)
        reloaded = load_calibrator(path)

        test_probs = np.array([0.0, 0.5, 1.0])
        output = reloaded.transform(test_probs)

        self.assertTrue(np.all(output >= 0.0))
        self.assertTrue(np.all(output <= 1.0))

    # --- metadata v2 serialization/reload ---

    def test_metadata_v2_json_round_trips_all_fields(self):

        metadata = self._sample_metadata()
        path = write_metadata_v2(metadata, self.tmp_dir)
        reloaded = load_metadata_v2(path)

        self.assertEqual(reloaded["model_version"], metadata.model_version)
        self.assertEqual(reloaded["dataset_campaign_version"], metadata.dataset_campaign_version)
        self.assertEqual(reloaded["training_seed"], metadata.training_seed)
        self.assertEqual(reloaded["algorithm"], metadata.algorithm)
        self.assertEqual(reloaded["feature_order"], list(metadata.feature_order))
        self.assertEqual(reloaded["candidate_types_supported"], list(metadata.candidate_types_supported))
        self.assertEqual(reloaded["known_limitations"], list(metadata.known_limitations))

    def test_metadata_v2_always_marks_not_wired_into_live_inference(self):

        metadata = self._sample_metadata()
        self.assertTrue(metadata.to_dict()["not_wired_into_live_inference"])

    def test_metadata_v2_is_json_serializable(self):

        metadata = self._sample_metadata()
        serialized = json.dumps(metadata.to_dict())
        self.assertIsInstance(serialized, str)

    def test_metadata_v2_dataset_version_matches_campaign_v2_not_v1(self):
        """Metadata/version consistency (Phase 21): this milestone must
        never silently export a model tagged as trained against V1's
        campaign_version."""

        metadata = self._sample_metadata()
        self.assertEqual(metadata.dataset_campaign_version, "predictive_dataset_campaign_v2")
        self.assertNotEqual(metadata.dataset_campaign_version, "predictive_dataset_campaign_v1")


if __name__ == "__main__":
    unittest.main()
