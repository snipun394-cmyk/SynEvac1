import json
import unittest

from ai_training.models.evacuation_time_model import EvacuationTimeModel

from ai_explainability.prediction_report import PredictionReport, explain_prediction

from tests.ai_explainability_fixtures import RealTrainedModelsTestCase


class ExplainPredictionRegressionTests(RealTrainedModelsTestCase):

    def test_report_shape_for_a_regression_model(self):

        report = explain_prediction(
            self.evac_model, self.evac_X_test[0], scenario_id="scn-test", y_true=self.evac_y_test[0],
        )

        self.assertIsInstance(report, PredictionReport)
        self.assertEqual(report.scenario_id, "scn-test")
        self.assertIsInstance(report.prediction, float)
        self.assertIsNone(report.confidence)
        self.assertGreater(len(report.top_contributing_features), 0)
        self.assertEqual(report.input_summary, self.evac_X_test[0])
        self.assertEqual(report.target_summary["target_name"], "total_evacuation_time")
        self.assertEqual(report.target_summary["task"], "regression")
        self.assertEqual(report.target_summary["true_value"], self.evac_y_test[0])

    def test_prediction_matches_model_predict(self):

        report = explain_prediction(self.evac_model, self.evac_X_test[0])
        expected = self.evac_model.predict([self.evac_X_test[0]])[0]

        self.assertEqual(report.prediction, expected)

    def test_top_contributing_features_respects_top_n(self):

        report = explain_prediction(self.evac_model, self.evac_X_test[0], top_n=2)

        self.assertEqual(len(report.top_contributing_features), 2)

    def test_top_contributing_features_sorted_descending(self):

        report = explain_prediction(self.evac_model, self.evac_X_test[0], top_n=10)

        contributions = [value for _name, value in report.top_contributing_features]
        self.assertEqual(contributions, sorted(contributions, reverse=True))

    def test_no_true_value_means_no_true_value_key(self):

        report = explain_prediction(self.evac_model, self.evac_X_test[0])

        self.assertNotIn("true_value", report.target_summary)

    def test_raises_when_model_is_not_fit(self):

        model = EvacuationTimeModel()

        with self.assertRaises(RuntimeError):
            explain_prediction(model, self.evac_X_test[0])


class ExplainPredictionClassificationTests(RealTrainedModelsTestCase):

    def test_confidence_is_a_probability_between_zero_and_one(self):

        report = explain_prediction(self.bottleneck_model, self.bottleneck_X_test[0])

        self.assertIsNotNone(report.confidence)
        self.assertGreaterEqual(report.confidence, 0.0)
        self.assertLessEqual(report.confidence, 1.0)

    def test_target_summary_reports_classification_task(self):

        report = explain_prediction(self.bottleneck_model, self.bottleneck_X_test[0])

        self.assertEqual(report.target_summary["task"], "classification")
        self.assertEqual(report.target_summary["target_name"], "bottleneck_location")


class PredictionReportToDictTests(RealTrainedModelsTestCase):

    def test_to_dict_is_json_serializable(self):

        report = explain_prediction(
            self.evac_model, self.evac_X_test[0], scenario_id="scn-json", y_true=self.evac_y_test[0],
        )

        payload = report.to_dict()

        # default=str absorbs numpy scalar types (np.float64 etc.) the
        # same way a real Command Center JSON encoder would need to --
        # this only proves the *shape* round-trips, not that every
        # value is already a bare Python primitive.
        serialized = json.dumps(payload, default=str)
        reloaded = json.loads(serialized)

        self.assertEqual(reloaded["scenario_id"], "scn-json")
        self.assertIn("prediction", reloaded)
        self.assertIn("top_contributing_features", reloaded)
        self.assertEqual(
            set(reloaded["top_contributing_features"][0].keys()), {"feature", "contribution"},
        )

    def test_to_dict_contains_every_top_level_field(self):

        report = explain_prediction(self.evac_model, self.evac_X_test[0])
        payload = report.to_dict()

        self.assertEqual(
            set(payload.keys()),
            {
                "scenario_id", "prediction", "confidence",
                "top_contributing_features", "input_summary", "target_summary",
            },
        )


if __name__ == "__main__":
    unittest.main()
