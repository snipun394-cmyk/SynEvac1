import os
import unittest

import numpy as np

from ai_training.models.bottleneck_model import NO_BOTTLENECK_LABEL, BottleneckModel
from ai_training.models.evacuation_time_model import EvacuationTimeModel
from ai_training.models.exit_usage_model import ExitUsageModel
from ai_training.models.smoke_prediction_model import SmokePredictionModel
from ai_training.split import apply_split, make_split

from tests.ai_training_fixtures import RealCampaignTestCase


class EvacuationTimeModelTests(RealCampaignTestCase):

    def test_build_table_excludes_identifier_columns_and_aligns_with_outcomes(self):

        X_rows, y, extra = EvacuationTimeModel.build_table(self.dataset)

        self.assertEqual(extra, {})
        self.assertEqual(len(X_rows), len(self.dataset))
        self.assertEqual(len(y), len(self.dataset))

        for row in X_rows:
            for identifier in ("scenario_id", "definition_id", "seed"):
                self.assertNotIn(identifier, row)

        for value, outcome in zip(y, self.dataset.outcome_rows()):
            self.assertEqual(value, outcome["total_evacuation_time"])

    def test_fit_predict_produces_one_finite_prediction_per_row(self):

        X_rows, y, _extra = EvacuationTimeModel.build_table(self.dataset)
        split = make_split(len(X_rows), test_size=0.3, random_state=0)
        X_train, _val, X_test = apply_split(X_rows, split)
        y_train, _val, _y_test = apply_split(y, split)

        model = EvacuationTimeModel()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        self.assertEqual(len(predictions), len(X_test))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_predict_before_fit_raises(self):

        model = EvacuationTimeModel()

        with self.assertRaises(RuntimeError):
            model.predict([{"total_occupants": 3}])

    def test_save_load_roundtrip_reproduces_identical_predictions(self):

        X_rows, y, _extra = EvacuationTimeModel.build_table(self.dataset)
        model = EvacuationTimeModel()
        model.fit(X_rows, y)

        path = os.path.join(self._tmp_dir, "evac_model.joblib")
        model.save(path)
        reloaded = EvacuationTimeModel.load(path)

        np.testing.assert_allclose(model.predict(X_rows), reloaded.predict(X_rows))
        self.assertEqual(reloaded.feature_schema, model.feature_schema)


class BottleneckModelTests(RealCampaignTestCase):

    def test_build_table_occurrence_produces_boolean_labels(self):

        _X_rows, y, extra = BottleneckModel.build_table(self.dataset, target="occurrence")

        self.assertEqual(extra, {})

        for value in y:
            self.assertIsInstance(value, bool)

    def test_build_table_location_uses_none_label_when_ground_truth_has_no_peak(self):

        _X_rows, y, _extra = BottleneckModel.build_table(self.dataset, target="location")

        for value, ground_truth in zip(y, self.dataset.ground_truth_rows()):

            if ground_truth.get("peak_congestion_location_id") is None:
                self.assertEqual(value, NO_BOTTLENECK_LABEL)
            else:
                self.assertEqual(value, ground_truth["peak_congestion_location_id"])

    def test_invalid_target_raises_on_build_table_and_constructor(self):

        with self.assertRaises(ValueError):
            BottleneckModel.build_table(self.dataset, target="not_a_real_target")

        with self.assertRaises(ValueError):
            BottleneckModel(target="not_a_real_target")

    def test_fit_predict_classification(self):

        X_rows, y, _extra = BottleneckModel.build_table(self.dataset, target="occurrence")
        split = make_split(len(X_rows), test_size=0.3, random_state=0)
        X_train, _val, X_test = apply_split(X_rows, split)
        y_train, _val, _y_test = apply_split(y, split)

        model = BottleneckModel(target="occurrence")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        self.assertEqual(len(predictions), len(X_test))
        for prediction in predictions:
            self.assertIn(prediction, (True, False))

    def test_save_load_roundtrip_preserves_target_and_predictions(self):

        X_rows, y, _extra = BottleneckModel.build_table(self.dataset, target="location")
        model = BottleneckModel(target="location")
        model.fit(X_rows, y)

        path = os.path.join(self._tmp_dir, "bottleneck_model.joblib")
        model.save(path)
        reloaded = BottleneckModel.load(path)

        self.assertEqual(reloaded.target, "location")
        self.assertEqual(list(model.predict(X_rows)), list(reloaded.predict(X_rows)))


class ExitUsageModelTests(RealCampaignTestCase):

    def test_build_table_percentages_sum_to_100_or_zero_per_scenario(self):

        _X_rows, y, extra = ExitUsageModel.build_table(self.dataset)

        self.assertIn("output_names", extra)
        self.assertGreater(len(extra["output_names"]), 0)

        for row in y:

            total = sum(row)
            self.assertTrue(
                abs(total - 100.0) < 1e-6 or abs(total) < 1e-6,
                f"expected exit percentages to sum to ~100 or 0, got {total}",
            )

    def test_fit_predict_returns_one_dict_per_scenario_keyed_by_exit_id(self):

        X_rows, y, extra = ExitUsageModel.build_table(self.dataset)
        output_names = extra["output_names"]

        model = ExitUsageModel()
        model.fit(X_rows, y, output_names=output_names)
        predictions = model.predict(X_rows)

        self.assertEqual(len(predictions), len(X_rows))

        for row in predictions:
            self.assertEqual(set(row.keys()), set(output_names))

    def test_predict_requires_fit_with_output_names_first(self):

        model = ExitUsageModel()

        with self.assertRaises(RuntimeError):
            model.predict([{"total_occupants": 1}])

    def test_save_load_roundtrip_preserves_output_names_and_predictions(self):

        X_rows, y, extra = ExitUsageModel.build_table(self.dataset)
        model = ExitUsageModel()
        model.fit(X_rows, y, output_names=extra["output_names"])

        path = os.path.join(self._tmp_dir, "exit_usage_model.joblib")
        model.save(path)
        reloaded = ExitUsageModel.load(path)

        self.assertEqual(reloaded.output_names, model.output_names)
        self.assertEqual(model.predict(X_rows), reloaded.predict(X_rows))


class SmokePredictionModelTests(RealCampaignTestCase):

    def test_build_table_groups_are_aligned_with_rows_and_labels_are_zone_ids(self):

        X_rows, y, extra = SmokePredictionModel.build_table(self.dataset)

        self.assertIn("groups", extra)
        self.assertEqual(len(extra["groups"]), len(X_rows))
        self.assertEqual(len(y), len(X_rows))

        scenario_ids = set(self.dataset.scenario_ids)

        for group, label in zip(extra["groups"], y):
            self.assertIn(group, scenario_ids)
            self.assertRegex(label, r"^Zone_\d+$")

    def test_feature_rows_exclude_scenario_id_but_keep_simulation_time(self):

        X_rows, _y, _extra = SmokePredictionModel.build_table(self.dataset)

        for row in X_rows:
            self.assertNotIn("scenario_id", row)
            self.assertIn("simulation_time", row)

    def test_fit_predict_with_group_aware_split_never_leaks_a_scenario_across_sides(self):

        X_rows, y, extra = SmokePredictionModel.build_table(self.dataset)
        groups = extra["groups"]

        split = make_split(len(X_rows), groups=groups, test_size=0.3, random_state=0)

        train_groups = {groups[i] for i in split.train_indices}
        test_groups = {groups[i] for i in split.test_indices}
        self.assertEqual(train_groups & test_groups, set())

        X_train, _val, X_test = apply_split(X_rows, split)
        y_train, _val, _y_test = apply_split(y, split)

        model = SmokePredictionModel()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        self.assertEqual(len(predictions), len(X_test))
        for prediction in predictions:
            self.assertRegex(prediction, r"^Zone_\d+$")


if __name__ == "__main__":
    unittest.main()
