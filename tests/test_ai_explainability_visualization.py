import os
import shutil
import tempfile
import unittest

from ai_explainability.benchmark import benchmark_algorithms, to_table_rows
from ai_explainability.feature_importance import rank_feature_importances
from ai_explainability.visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    plot_prediction_comparison,
    plot_residuals,
    render_benchmark_table_image,
)

from tests.ai_explainability_fixtures import RealTrainedModelsTestCase


class VisualizationTests(RealTrainedModelsTestCase):

    def setUp(self):

        self.out_dir = tempfile.mkdtemp(prefix="ai_explainability_viz_")

    def tearDown(self):

        shutil.rmtree(self.out_dir, ignore_errors=True)

    def _path(self, name):

        return os.path.join(self.out_dir, name)

    def test_plot_feature_importance_creates_a_nonempty_file(self):

        ranked = rank_feature_importances(self.evac_model, top_n=5)
        path = plot_feature_importance(ranked, self._path("fi.png"))

        self.assertEqual(path, self._path("fi.png"))
        self.assertTrue(os.path.isfile(path))
        self.assertGreater(os.path.getsize(path), 0)

    def test_plot_prediction_comparison_creates_a_nonempty_file(self):

        predictions = self.evac_model.predict(self.evac_X_test)
        path = plot_prediction_comparison(self.evac_y_test, predictions, self._path("pred.png"))

        self.assertTrue(os.path.isfile(path))
        self.assertGreater(os.path.getsize(path), 0)

    def test_plot_residuals_creates_a_nonempty_file(self):

        predictions = self.evac_model.predict(self.evac_X_test)
        path = plot_residuals(self.evac_y_test, predictions, self._path("resid.png"))

        self.assertTrue(os.path.isfile(path))
        self.assertGreater(os.path.getsize(path), 0)

    def test_plot_confusion_matrix_creates_a_nonempty_file(self):

        predictions = self.bottleneck_model.predict(self.bottleneck_X_test)
        path = plot_confusion_matrix(self.bottleneck_y_test, predictions, self._path("cm.png"))

        self.assertTrue(os.path.isfile(path))
        self.assertGreater(os.path.getsize(path), 0)

    def test_plot_model_comparison_and_benchmark_table_create_nonempty_files(self):

        results = benchmark_algorithms(self.dataset, "evacuation_time", ["random_forest", "gradient_boosting"])

        comparison_path = plot_model_comparison(results, "mae", self._path("cmp.png"))
        table_path = render_benchmark_table_image(to_table_rows(results), self._path("table.png"))

        for path in (comparison_path, table_path):
            self.assertTrue(os.path.isfile(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_creates_missing_parent_directories(self):

        nested_path = os.path.join(self.out_dir, "nested", "deeper", "fi.png")
        ranked = rank_feature_importances(self.evac_model, top_n=3)

        path = plot_feature_importance(ranked, nested_path)

        self.assertTrue(os.path.isfile(path))


if __name__ == "__main__":
    unittest.main()
