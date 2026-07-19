import unittest

import ai_training as at
from ai_inference.confidence import (
    ConfidenceReport,
    build_confidence_report,
    build_ensemble_confidence_report,
    estimate_confidence,
)
from ai_inference.ensemble import Ensemble, EnsembleMember
from ai_inference.loader import load_model

from tests.ai_inference_fixtures import RealSavedModelsTestCase


class EstimateConfidenceRegressionTests(RealSavedModelsTestCase):

    def setUp(self):

        X, _y, _extra = at.EvacuationTimeModel.build_table(self.dataset)
        self.row = X[0]

    def test_random_forest_reports_a_dispersion_based_confidence(self):

        loaded = load_model(self.evac_dirs["random_forest"])

        confidence, basis = estimate_confidence(loaded, self.row)

        self.assertEqual(basis, "random_forest_tree_dispersion")
        self.assertIsNotNone(confidence)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_gradient_boosting_honestly_reports_no_confidence(self):

        loaded = load_model(self.evac_dirs["gradient_boosting"])

        confidence, basis = estimate_confidence(loaded, self.row)

        self.assertIsNone(confidence)
        self.assertEqual(basis, "unavailable_for_algorithm")

    def test_xgboost_honestly_reports_no_confidence(self):

        loaded = load_model(self.evac_dirs["xgboost"])

        confidence, basis = estimate_confidence(loaded, self.row)

        self.assertIsNone(confidence)
        self.assertEqual(basis, "unavailable_for_algorithm")

    def test_is_deterministic_for_the_same_row(self):

        loaded = load_model(self.evac_dirs["random_forest"])

        first, _basis = estimate_confidence(loaded, self.row)
        second, _basis = estimate_confidence(loaded, self.row)

        self.assertEqual(first, second)


class EstimateConfidenceClassificationTests(RealSavedModelsTestCase):

    def test_uses_predict_proba(self):

        loaded = load_model(self.bottleneck_location_dir)

        Xb, _yb, _extra = at.BottleneckModel.build_table(self.dataset, target="location")

        confidence, basis = estimate_confidence(loaded, Xb[0])

        self.assertEqual(basis, "predict_proba")
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


class ExitUsageConfidenceTests(RealSavedModelsTestCase):

    def test_multioutput_random_forest_reports_averaged_dispersion(self):

        loaded = load_model(self.exit_usage_dir)

        Xe, _ye, _extra = at.ExitUsageModel.build_table(self.dataset)

        confidence, basis = estimate_confidence(loaded, Xe[0])

        self.assertEqual(basis, "random_forest_tree_dispersion")
        self.assertIsNotNone(confidence)


class BuildConfidenceReportTests(RealSavedModelsTestCase):

    def test_report_contains_full_provenance(self):

        loaded = load_model(self.evac_dirs["random_forest"], dataset_version="campaign-v9")

        X, _y, _extra = at.EvacuationTimeModel.build_table(self.dataset)

        report = build_confidence_report(loaded, X[0])

        self.assertIsInstance(report, ConfidenceReport)
        self.assertEqual(report.experiment_id, "evac-random_forest")
        self.assertEqual(report.model_version, loaded.provenance.model_version)
        self.assertEqual(report.training_dataset_version, "campaign-v9")
        self.assertEqual(report.model_used, "evacuation_time (random_forest)")
        self.assertEqual(report.prediction, loaded.model.predict([X[0]])[0])

    def test_to_dict_contains_every_field(self):

        loaded = load_model(self.evac_dirs["random_forest"])
        X, _y, _extra = at.EvacuationTimeModel.build_table(self.dataset)

        report = build_confidence_report(loaded, X[0])
        payload = report.to_dict()

        self.assertEqual(
            set(payload.keys()),
            {
                "prediction", "confidence", "confidence_basis", "model_used",
                "training_dataset_version", "experiment_id", "model_version",
            },
        )


class BuildEnsembleConfidenceReportTests(RealSavedModelsTestCase):

    def test_regression_ensemble_reports_dispersion_based_confidence(self):

        members = [
            EnsembleMember(load_model(self.evac_dirs["random_forest"]))
            for _ in range(1)
        ] + [
            EnsembleMember(load_model(self.evac_dirs["gradient_boosting"])),
            EnsembleMember(load_model(self.evac_dirs["xgboost"])),
        ]
        ensemble = Ensemble(members)

        X, _y, _extra = at.EvacuationTimeModel.build_table(self.dataset)
        report = build_ensemble_confidence_report(ensemble, X[0])

        self.assertEqual(report.confidence_basis, "ensemble_prediction_dispersion")
        self.assertIsNotNone(report.confidence)
        self.assertGreaterEqual(report.confidence, 0.0)
        self.assertLessEqual(report.confidence, 1.0)
        self.assertIn(",", report.experiment_id)
        self.assertIn(",", report.model_version)

    def test_classification_ensemble_reports_vote_agreement(self):

        ensemble = Ensemble([EnsembleMember(load_model(self.bottleneck_location_dir))])

        Xb, _yb, _extra = at.BottleneckModel.build_table(self.dataset, target="location")
        report = build_ensemble_confidence_report(ensemble, Xb[0])

        self.assertEqual(report.confidence_basis, "ensemble_vote_agreement")
        # A single-member ensemble always agrees with itself.
        self.assertEqual(report.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
