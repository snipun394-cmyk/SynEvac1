import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from predictive_model.baselines import DecisionTreeBaseline
from predictive_model.model_export import ModelMetadata, export_model, load_model, load_model_metadata


class ModelExportTests(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _sample_metadata(self, **overrides):
        defaults = dict(
            model_name="decision_tree",
            model_library="DecisionTreeBaseline",
            dataset_schema_version="1.0",
            dataset_campaign_version="predictive_dataset_campaign_v1",
            dataset_feature_version="1.0",
            dataset_target_version="v1-congestion-threshold-2-horizon-window",
            prediction_horizon_seconds=20.0,
            feature_names=("a", "b", "c"),
            train_scenario_count=1400,
            val_scenario_count=300,
            test_scenario_count=300,
            decision_threshold=0.42,
            class_weight_strategy="balanced",
            validation_metrics={"roc_auc": 0.9},
            test_metrics={"roc_auc": 0.89},
            production_readiness="PROMISING_BUT_NEEDS_MORE_DATA",
            production_readiness_rationale="test rationale",
        )
        defaults.update(overrides)
        return ModelMetadata(**defaults)

    def test_export_writes_model_and_metadata_files(self):

        model = DecisionTreeBaseline(seed=1)
        X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=float)
        y = np.array([0, 1, 0, 1])
        model.fit(X, y)

        paths = export_model(model, self._sample_metadata(feature_names=("f0", "f1")), self.tmp_dir)

        self.assertTrue(os.path.exists(paths["model_path"]))
        self.assertTrue(os.path.exists(paths["metadata_path"]))

    def test_exported_model_round_trips_predictions(self):

        model = DecisionTreeBaseline(seed=1)
        X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=float)
        y = np.array([0, 1, 0, 1])
        model.fit(X, y)

        original_predictions = model.predict_proba(X)

        paths = export_model(model, self._sample_metadata(feature_names=("f0", "f1")), self.tmp_dir)
        reloaded_model = load_model(paths["model_path"])
        reloaded_predictions = reloaded_model.predict_proba(X)

        np.testing.assert_array_almost_equal(original_predictions, reloaded_predictions)

    def test_metadata_json_round_trips_all_fields(self):

        metadata = self._sample_metadata()
        model = DecisionTreeBaseline(seed=1)
        model.fit(np.array([[0.0], [1.0]]), np.array([0, 1]))

        paths = export_model(model, metadata, self.tmp_dir)
        reloaded = load_model_metadata(paths["metadata_path"])

        self.assertEqual(reloaded["model_name"], metadata.model_name)
        self.assertEqual(reloaded["dataset_schema_version"], metadata.dataset_schema_version)
        self.assertEqual(reloaded["prediction_horizon_seconds"], metadata.prediction_horizon_seconds)
        self.assertEqual(reloaded["feature_names"], list(metadata.feature_names))
        self.assertEqual(reloaded["production_readiness"], metadata.production_readiness)

    def test_metadata_always_marks_not_wired_into_live_inference(self):
        """Phase 12's own charter: this milestone must never wire its
        model into live inference. The exported metadata must always say
        so explicitly, regardless of what production_readiness ends up
        being classified as."""

        for readiness in ("READY", "PROMISING_BUT_NEEDS_MORE_DATA", "NOT_READY"):

            metadata = self._sample_metadata(production_readiness=readiness)
            self.assertTrue(metadata.to_dict()["not_wired_into_live_inference"])

    def test_metadata_is_json_serializable(self):

        metadata = self._sample_metadata()
        serialized = json.dumps(metadata.to_dict())
        self.assertIsInstance(serialized, str)


if __name__ == "__main__":
    unittest.main()
