import json
import os
import shutil
import unittest

from ai_inference.loader import IncompatibleModelError, LoadedModel, ModelProvenance, load_model

from tests.ai_inference_fixtures import RealSavedModelsTestCase


class LoadModelTests(RealSavedModelsTestCase):

    def test_returns_a_loaded_model_with_a_fitted_estimator(self):

        loaded = load_model(self.evac_rf_dir)

        self.assertIsInstance(loaded, LoadedModel)
        self.assertIsNotNone(loaded.model.preprocessor)
        self.assertIsNotNone(loaded.model.feature_schema)
        self.assertEqual(loaded.directory, self.evac_rf_dir)

    def test_provenance_matches_the_saved_manifest(self):

        loaded = load_model(self.evac_rf_dir)

        self.assertIsInstance(loaded.provenance, ModelProvenance)
        self.assertEqual(loaded.provenance.experiment_name, "evac-random_forest")
        self.assertEqual(loaded.provenance.model_name, "evacuation_time")
        self.assertEqual(loaded.provenance.algorithm, "random_forest")
        self.assertEqual(set(loaded.provenance.metrics.keys()), {"mae", "rmse", "r2"})
        self.assertGreater(loaded.provenance.train_size, 0)
        self.assertGreater(loaded.provenance.test_size, 0)

    def test_dataset_version_defaults_to_none_and_is_never_fabricated(self):

        loaded = load_model(self.evac_rf_dir)

        self.assertIsNone(loaded.provenance.dataset_version)

    def test_dataset_version_is_carried_through_verbatim_when_supplied(self):

        loaded = load_model(self.evac_rf_dir, dataset_version="campaign-2026-07-14")

        self.assertEqual(loaded.provenance.dataset_version, "campaign-2026-07-14")

    def test_model_version_is_deterministic_for_the_same_artifact(self):

        first = load_model(self.evac_rf_dir)
        second = load_model(self.evac_rf_dir)

        self.assertEqual(first.provenance.model_version, second.provenance.model_version)

    def test_different_models_have_different_model_versions(self):

        rf = load_model(self.evac_dirs["random_forest"])
        gb = load_model(self.evac_dirs["gradient_boosting"])

        self.assertNotEqual(rf.provenance.model_version, gb.provenance.model_version)

    def test_bottleneck_model_restores_its_target(self):

        loaded_location = load_model(self.bottleneck_location_dir)
        loaded_occurrence = load_model(self.bottleneck_occurrence_dir)

        self.assertEqual(loaded_location.model.target, "location")
        self.assertEqual(loaded_occurrence.model.target, "occurrence")

    def test_exit_usage_model_restores_its_output_names(self):

        loaded = load_model(self.exit_usage_dir)

        self.assertIsNotNone(loaded.model.output_names)
        self.assertGreater(len(loaded.model.output_names), 0)

    def test_smoke_prediction_model_loads_successfully(self):

        loaded = load_model(self.smoke_dir)

        self.assertEqual(loaded.provenance.model_name, "smoke_prediction")


class LoadModelRejectionTests(RealSavedModelsTestCase):

    def test_rejects_a_directory_with_no_manifest_or_model(self):

        missing_dir = os.path.join(self._tmp_dir, "does_not_exist")

        with self.assertRaises(IncompatibleModelError):
            load_model(missing_dir)

    def test_rejects_a_directory_missing_only_the_model_file(self):

        directory = os.path.join(self._tmp_dir, "missing_model_only")
        os.makedirs(directory, exist_ok=True)
        shutil.copy(
            os.path.join(self.evac_rf_dir, "manifest.json"), os.path.join(directory, "manifest.json"),
        )

        with self.assertRaises(IncompatibleModelError):
            load_model(directory)

    def test_rejects_invalid_json_manifest(self):

        directory = os.path.join(self._tmp_dir, "broken_json")
        os.makedirs(directory, exist_ok=True)
        shutil.copy(
            os.path.join(self.evac_rf_dir, "model.joblib"), os.path.join(directory, "model.joblib"),
        )

        with open(os.path.join(directory, "manifest.json"), "w", encoding="utf-8") as handle:
            handle.write("{not valid json")

        with self.assertRaises(IncompatibleModelError):
            load_model(directory)

    def test_rejects_a_manifest_missing_required_top_level_keys(self):

        directory = os.path.join(self._tmp_dir, "incomplete_manifest")
        os.makedirs(directory, exist_ok=True)
        shutil.copy(
            os.path.join(self.evac_rf_dir, "model.joblib"), os.path.join(directory, "model.joblib"),
        )

        with open(os.path.join(directory, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump({"config": {"name": "x", "model_name": "evacuation_time"}}, handle)

        with self.assertRaises(IncompatibleModelError):
            load_model(directory)

    def test_rejects_an_unknown_model_name(self):

        directory = os.path.join(self._tmp_dir, "unknown_model_name")
        os.makedirs(directory, exist_ok=True)
        shutil.copy(
            os.path.join(self.evac_rf_dir, "model.joblib"), os.path.join(directory, "model.joblib"),
        )

        with open(os.path.join(self.evac_rf_dir, "manifest.json"), "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        manifest["config"]["model_name"] = "not_a_real_model"

        with open(os.path.join(directory, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)

        with self.assertRaises(IncompatibleModelError):
            load_model(directory)

    def test_rejects_a_corrupted_model_joblib(self):

        directory = os.path.join(self._tmp_dir, "corrupted_joblib")
        os.makedirs(directory, exist_ok=True)
        shutil.copy(
            os.path.join(self.evac_rf_dir, "manifest.json"), os.path.join(directory, "manifest.json"),
        )

        with open(os.path.join(directory, "model.joblib"), "wb") as handle:
            handle.write(b"this is not a valid joblib payload")

        with self.assertRaises(IncompatibleModelError):
            load_model(directory)


if __name__ == "__main__":
    unittest.main()
