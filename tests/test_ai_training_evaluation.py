import unittest

from ai_training.evaluation import (
    evaluate_classification,
    evaluate_multioutput_regression,
    evaluate_regression,
)
from ai_training.models.bottleneck_model import BottleneckModel
from ai_training.models.evacuation_time_model import EvacuationTimeModel
from ai_training.models.exit_usage_model import ExitUsageModel
from ai_training.split import apply_split, make_split

from tests.ai_training_fixtures import RealCampaignTestCase


class EvaluateRegressionTests(RealCampaignTestCase):

    def test_returns_mae_rmse_r2_keys(self):

        X_rows, y, _extra = EvacuationTimeModel.build_table(self.dataset)
        split = make_split(len(X_rows), test_size=0.3, random_state=0)
        X_train, _val, X_test = apply_split(X_rows, split)
        y_train, _val, y_test = apply_split(y, split)

        model = EvacuationTimeModel()
        model.fit(X_train, y_train)

        metrics = evaluate_regression(model, X_test, y_test)

        self.assertEqual(set(metrics.keys()), {"mae", "rmse", "r2"})
        self.assertGreaterEqual(metrics["mae"], 0.0)
        self.assertGreaterEqual(metrics["rmse"], 0.0)


class EvaluateClassificationTests(RealCampaignTestCase):

    def test_returns_full_classification_metric_set(self):

        X_rows, y, _extra = BottleneckModel.build_table(self.dataset, target="occurrence")
        split = make_split(len(X_rows), test_size=0.3, random_state=0)
        X_train, _val, X_test = apply_split(X_rows, split)
        y_train, _val, y_test = apply_split(y, split)

        model = BottleneckModel(target="occurrence")
        model.fit(X_train, y_train)

        metrics = evaluate_classification(model, X_test, y_test)

        self.assertEqual(
            set(metrics.keys()), {"accuracy", "precision", "recall", "f1", "roc_auc"},
        )
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)


class EvaluateMultioutputRegressionTests(RealCampaignTestCase):

    def test_returns_per_output_and_macro_average(self):

        X_rows, y, extra = ExitUsageModel.build_table(self.dataset)
        output_names = extra["output_names"]

        split = make_split(len(X_rows), test_size=0.3, random_state=0)
        X_train, _val, X_test = apply_split(X_rows, split)
        y_train, _val, y_test = apply_split(y, split)

        model = ExitUsageModel()
        model.fit(X_train, y_train, output_names=output_names)

        metrics = evaluate_multioutput_regression(model, X_test, y_test, output_names)

        self.assertEqual(set(metrics.keys()), {"per_output", "macro_average"})
        self.assertEqual(set(metrics["per_output"].keys()), set(output_names))
        self.assertEqual(set(metrics["macro_average"].keys()), {"mae", "rmse", "r2"})


if __name__ == "__main__":
    unittest.main()
