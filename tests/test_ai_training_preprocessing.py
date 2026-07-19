import unittest

import numpy as np

from ai_training.preprocessing import FeatureSchema, Preprocessor, select_features

from tests.ai_training_fixtures import RealCampaignTestCase


class SelectFeaturesTests(unittest.TestCase):

    def test_exclude_drops_named_columns(self):

        rows = [{"a": 1, "b": 2, "id": "x"}, {"a": 3, "b": 4, "id": "y"}]

        selected = select_features(rows, exclude=("id",))

        self.assertEqual(selected, [{"a": 1, "b": 2}, {"a": 3, "b": 4}])

    def test_include_restricts_to_an_explicit_allow_list(self):

        rows = [{"a": 1, "b": 2, "c": 3}]

        selected = select_features(rows, include=("a", "c"))

        self.assertEqual(selected, [{"a": 1, "c": 3}])

    def test_include_and_exclude_combine(self):

        rows = [{"a": 1, "b": 2, "c": 3}]

        selected = select_features(rows, include=("a", "b", "c"), exclude=("b",))

        self.assertEqual(selected, [{"a": 1, "c": 3}])


class FeatureSchemaInferTests(unittest.TestCase):

    def test_classifies_numeric_and_categorical_columns(self):

        rows = [
            {"count": 1, "growth": 100.5, "state": "OPEN"},
            {"count": 2, "growth": 200.0, "state": "CLOSED"},
        ]

        schema = FeatureSchema.infer(rows)

        self.assertEqual(schema.numeric_columns, ("count", "growth"))
        self.assertEqual(schema.categorical_columns, ("state",))

    def test_column_order_is_always_alphabetical(self):

        rows = [{"zebra": 1, "apple": 2, "middle": "cat"}]

        schema = FeatureSchema.infer(rows)

        self.assertEqual(schema.columns, ("apple", "zebra", "middle"))

    def test_exclude_removes_identifier_columns_from_the_schema(self):

        rows = [{"scenario_id": "s1", "growth": 1.0}]

        schema = FeatureSchema.infer(rows, exclude=("scenario_id",))

        self.assertEqual(schema.columns, ("growth",))

    def test_bool_valued_columns_are_categorical_not_numeric(self):

        rows = [{"flag": True}, {"flag": False}]

        schema = FeatureSchema.infer(rows)

        self.assertEqual(schema.categorical_columns, ("flag",))

    def test_all_missing_values_fall_back_to_categorical(self):

        rows = [{"always_none": None}, {"always_none": None}]

        schema = FeatureSchema.infer(rows)

        self.assertEqual(schema.categorical_columns, ("always_none",))


class PreprocessorDeterminismAndLeakageTests(unittest.TestCase):

    def setUp(self):

        self.train_rows = [
            {"growth": 100.0, "state": "OPEN"},
            {"growth": 200.0, "state": "CLOSED"},
            {"growth": None, "state": "OPEN"},
            {"growth": 300.0, "state": None},
        ]
        self.schema = FeatureSchema.infer(self.train_rows)

    def test_fit_transform_is_deterministic_across_separate_instances(self):

        first = Preprocessor(self.schema).fit(self.train_rows).transform(self.train_rows)
        second = Preprocessor(self.schema).fit(self.train_rows).transform(self.train_rows)

        np.testing.assert_array_equal(first, second)

    def test_missing_numeric_value_is_imputed_with_the_training_median(self):

        preprocessor = Preprocessor(self.schema).fit(self.train_rows)

        # median of the three observed growth values (100, 200, 300) is
        # 200 -- the row with growth=None must be imputed to exactly
        # that, then standardized the same way every other row is.
        transformed = preprocessor.transform([{"growth": None, "state": "OPEN"}])
        transformed_median_row = preprocessor.transform([{"growth": 200.0, "state": "OPEN"}])

        np.testing.assert_allclose(transformed, transformed_median_row)

    def test_transform_never_refits_on_new_data_no_leakage(self):

        preprocessor = Preprocessor(self.schema).fit(self.train_rows)

        before = preprocessor.transform(self.train_rows)

        # Transforming wildly different-looking data must not change
        # what fit() already captured -- calling transform() again on
        # the original rows must produce the exact same result.
        preprocessor.transform([{"growth": 999_999.0, "state": "UNSEEN_CATEGORY"}])
        after = preprocessor.transform(self.train_rows)

        np.testing.assert_array_equal(before, after)

    def test_unseen_category_at_transform_time_yields_an_all_zero_one_hot_block(self):

        preprocessor = Preprocessor(self.schema).fit(self.train_rows)
        names = preprocessor.feature_names_out()

        categorical_positions = [i for i, name in enumerate(names) if name.startswith("state=")]

        transformed = preprocessor.transform([{"growth": 100.0, "state": "NEVER_SEEN"}])

        for position in categorical_positions:
            self.assertEqual(transformed[0, position], 0.0)

    def test_feature_names_out_reports_one_hot_categories(self):

        preprocessor = Preprocessor(self.schema).fit(self.train_rows)
        names = preprocessor.feature_names_out()

        self.assertIn("growth", names)
        self.assertTrue(any(name.startswith("state=") for name in names))

    def test_transform_before_fit_raises(self):

        preprocessor = Preprocessor(self.schema)

        with self.assertRaises(RuntimeError):
            preprocessor.transform(self.train_rows)


class PreprocessorRealDatasetIntegrationTests(RealCampaignTestCase):

    def test_fits_and_transforms_real_scenario_feature_rows(self):

        rows = select_features(
            self.dataset.scenario_feature_rows(), exclude=("scenario_id", "definition_id", "seed"),
        )
        schema = FeatureSchema.infer(rows)
        preprocessor = Preprocessor(schema).fit(rows)

        transformed = preprocessor.transform(rows)

        self.assertEqual(transformed.shape[0], len(rows))
        self.assertEqual(transformed.shape[1], len(preprocessor.feature_names_out()))
        self.assertFalse(np.isnan(transformed).any())


if __name__ == "__main__":
    unittest.main()
