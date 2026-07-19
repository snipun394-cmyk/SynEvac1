import unittest

import numpy as np

import ai_training as at
from ai_inference.cache import PredictionCache
from ai_inference.loader import load_model
from ai_inference.predictor import Prediction, Predictor, prediction_type_key

from tests.ai_inference_fixtures import RealSavedModelsTestCase


class PredictionTypeKeyTests(RealSavedModelsTestCase):

    def test_evacuation_time_key_is_its_own_model_name(self):

        loaded = load_model(self.evac_rf_dir)

        self.assertEqual(prediction_type_key(loaded), "evacuation_time")

    def test_bottleneck_key_disambiguates_by_target(self):

        location = load_model(self.bottleneck_location_dir)
        occurrence = load_model(self.bottleneck_occurrence_dir)

        self.assertEqual(prediction_type_key(location), "bottleneck_location")
        self.assertEqual(prediction_type_key(occurrence), "bottleneck_occurrence")


class PredictorTests(RealSavedModelsTestCase):

    def setUp(self):

        self.predictor = Predictor.from_directories(
            [self.evac_rf_dir, self.bottleneck_location_dir, self.exit_usage_dir, self.smoke_dir]
        )

        X, _y, _extra = at.EvacuationTimeModel.build_table(self.dataset)
        self.evac_row = X[0]

        Xb, _yb, _extra = at.BottleneckModel.build_table(self.dataset, target="location")
        self.bottleneck_row = Xb[0]

        Xe, _ye, _extra = at.ExitUsageModel.build_table(self.dataset)
        self.exit_row = Xe[0]

        Xs, _ys, _extras = at.SmokePredictionModel.build_table(self.dataset)
        self.smoke_row = Xs[0]

    def test_available_predictions_lists_every_loaded_model(self):

        self.assertEqual(
            self.predictor.available_predictions(),
            sorted(["evacuation_time", "bottleneck_location", "exit_usage", "smoke_prediction"]),
        )

    def test_predict_unknown_prediction_type_raises_key_error(self):

        with self.assertRaises(KeyError):
            self.predictor.predict("not_a_real_type", self.evac_row)

    def test_regression_prediction_has_no_probability(self):

        prediction = self.predictor.predict("evacuation_time", self.evac_row)

        self.assertIsInstance(prediction, Prediction)
        self.assertIsInstance(prediction.value, float)
        self.assertIsNone(prediction.probability)
        self.assertEqual(prediction.model_name, "evacuation_time")
        self.assertEqual(prediction.algorithm, "random_forest")

    def test_classification_prediction_has_a_probability_between_zero_and_one(self):

        prediction = self.predictor.predict("bottleneck_location", self.bottleneck_row)

        self.assertIsNotNone(prediction.probability)
        self.assertGreaterEqual(prediction.probability, 0.0)
        self.assertLessEqual(prediction.probability, 1.0)

    def test_exit_usage_prediction_is_a_dict_keyed_by_exit_id(self):

        prediction = self.predictor.predict("exit_usage", self.exit_row)

        self.assertIsInstance(prediction.value, dict)
        self.assertGreater(len(prediction.value), 0)

    def test_smoke_prediction_returns_a_zone_identifier(self):

        prediction = self.predictor.predict("smoke_prediction", self.smoke_row)

        self.assertRegex(str(prediction.value), r"^Zone_\d+$")

    def test_predict_all_returns_one_prediction_per_loaded_model(self):

        predictions = self.predictor.predict_all(self.evac_row)

        self.assertEqual(set(predictions.keys()), set(self.predictor.available_predictions()))

    def test_prediction_is_deterministic_across_repeated_calls(self):

        first = self.predictor.predict("evacuation_time", self.evac_row)
        second = self.predictor.predict("evacuation_time", self.evac_row)

        self.assertEqual(first.value, second.value)

    def test_prediction_to_dict_is_plain_data(self):

        prediction = self.predictor.predict("evacuation_time", self.evac_row)
        payload = prediction.to_dict()

        self.assertEqual(
            set(payload.keys()),
            {"prediction_type", "value", "probability", "model_name", "algorithm", "model_version"},
        )


class PredictorWithCacheTests(RealSavedModelsTestCase):

    def test_repeated_predictions_are_served_from_the_cache(self):

        cache = PredictionCache()
        predictor = Predictor.from_directories([self.evac_rf_dir], cache=cache)

        X, _y, _extra = at.EvacuationTimeModel.build_table(self.dataset)
        row = X[0]

        first = predictor.predict("evacuation_time", row)
        second = predictor.predict("evacuation_time", row)

        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hits, 1)
        self.assertEqual(first.value, second.value)

    def test_different_rows_each_produce_their_own_cache_entry(self):

        cache = PredictionCache()
        predictor = Predictor.from_directories([self.evac_rf_dir], cache=cache)

        X, _y, _extra = at.EvacuationTimeModel.build_table(self.dataset)

        # This tiny fixture's narrow occupancy range means some rows
        # are genuinely identical dicts across different scenarios
        # (confirmed: X[0] == X[1]) -- picking the first row that
        # actually differs from X[0] proves the cache keys on feature
        # CONTENT, not on which scenario happened to produce it.
        first_distinct_row = next(row for row in X if row != X[0])

        predictor.predict("evacuation_time", X[0])
        predictor.predict("evacuation_time", first_distinct_row)

        self.assertEqual(cache.misses, 2)
        self.assertEqual(cache.hits, 0)


if __name__ == "__main__":
    unittest.main()
