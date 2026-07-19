import unittest

import ai_training as at
from ai_inference.ensemble import Ensemble, EnsembleMember
from ai_inference.loader import load_model

from tests.ai_inference_fixtures import RealSavedModelsTestCase


class _StubModel:

    def __init__(self, prediction, *, is_classifier, target=None):
        self.is_classifier = is_classifier
        self._prediction = prediction
        if target is not None:
            self.target = target

    def predict(self, X_rows):
        return [self._prediction for _ in X_rows]


class _StubProvenance:

    def __init__(self, model_name, experiment_name):
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.algorithm = "stub"
        self.metrics = {}
        self.model_version = f"stub-{experiment_name}"
        self.dataset_version = None


class _StubLoaded:

    def __init__(self, prediction, *, experiment_name, model_name="bottleneck", target="location"):
        self.model = _StubModel(prediction, is_classifier=True, target=target)
        self.provenance = _StubProvenance(model_name, experiment_name)


class MajorityVoteTieBreakTests(unittest.TestCase):

    # Uses lightweight stubs (rather than real trained models) to pin
    # down the exact deterministic tie-break rule -- real model
    # predictions can't be reliably forced into an exact tie on demand.

    def test_highest_total_weight_wins(self):

        members = [
            EnsembleMember(_StubLoaded("zone-1", experiment_name="a"), weight=1.0),
            EnsembleMember(_StubLoaded("zone-1", experiment_name="b"), weight=1.0),
            EnsembleMember(_StubLoaded("zone-2", experiment_name="c"), weight=1.5),
        ]
        ensemble = Ensemble(members)

        self.assertEqual(ensemble.predict_majority_vote({}), "zone-1")

    def test_exact_tie_breaks_alphabetically_by_label(self):

        members = [
            EnsembleMember(_StubLoaded("zone-2", experiment_name="a"), weight=1.0),
            EnsembleMember(_StubLoaded("zone-1", experiment_name="b"), weight=1.0),
        ]
        ensemble = Ensemble(members)

        self.assertEqual(ensemble.predict_majority_vote({}), "zone-1")


class EnsembleConstructionTests(RealSavedModelsTestCase):

    def test_requires_at_least_one_member(self):

        with self.assertRaises(ValueError):
            Ensemble([])

    def test_rejects_members_with_different_prediction_types(self):

        evac_member = EnsembleMember(load_model(self.evac_rf_dir))
        bottleneck_member = EnsembleMember(load_model(self.bottleneck_location_dir))

        with self.assertRaises(ValueError):
            Ensemble([evac_member, bottleneck_member])

    def test_prediction_type_is_derived_from_its_members(self):

        members = [EnsembleMember(load_model(self.evac_rf_dir))]
        ensemble = Ensemble(members)

        self.assertEqual(ensemble.prediction_type, "evacuation_time")


class RegressionEnsembleTests(RealSavedModelsTestCase):

    def setUp(self):

        self.members = [
            EnsembleMember(load_model(self.evac_dirs["random_forest"]), weight=2.0),
            EnsembleMember(load_model(self.evac_dirs["gradient_boosting"]), weight=1.0),
            EnsembleMember(load_model(self.evac_dirs["xgboost"]), weight=1.0),
        ]
        self.ensemble = Ensemble(self.members)

        X, _y, _extra = at.EvacuationTimeModel.build_table(self.dataset)
        self.row = X[0]

    def test_is_classifier_is_false_for_a_regression_ensemble(self):

        self.assertFalse(self.ensemble.is_classifier)

    def test_predict_all_members_returns_one_value_per_member(self):

        predictions = self.ensemble.predict_all_members(self.row)

        self.assertEqual(len(predictions), 3)
        self.assertEqual(
            set(predictions.keys()), {"evac-random_forest", "evac-gradient_boosting", "evac-xgboost"},
        )

    def test_average_is_the_unweighted_mean_of_member_predictions(self):

        raw_values = list(self.ensemble.predict_all_members(self.row).values())
        expected = sum(raw_values) / len(raw_values)

        average = self.ensemble.predict_average(self.row)

        self.assertAlmostEqual(average, expected, places=6)

    def test_weighted_average_differs_from_plain_average_when_weights_differ(self):

        average = self.ensemble.predict_average(self.row)
        weighted = self.ensemble.predict_weighted_average(self.row)

        raw_values = self.ensemble.predict_all_members(self.row)
        expected_weighted = (
            2.0 * raw_values["evac-random_forest"]
            + 1.0 * raw_values["evac-gradient_boosting"]
            + 1.0 * raw_values["evac-xgboost"]
        ) / 4.0

        self.assertAlmostEqual(weighted, expected_weighted, places=6)
        # Not asserting average != weighted in general (could coincide),
        # just that weighted matches its own, differently-weighted formula.

    def test_predict_average_raises_for_a_classification_ensemble(self):

        classification_ensemble = Ensemble([EnsembleMember(load_model(self.bottleneck_location_dir))])

        with self.assertRaises(TypeError):
            classification_ensemble.predict_average(self.row)

    def test_best_model_picks_the_lowest_mae_member(self):

        best = self.ensemble.best_model("mae", minimize=True)

        expected_algorithm = min(
            self.members, key=lambda m: m.loaded_model.provenance.metrics["mae"],
        ).loaded_model.provenance.algorithm

        self.assertEqual(best.loaded_model.provenance.algorithm, expected_algorithm)

    def test_predict_best_matches_the_best_members_own_prediction(self):

        best_member = self.ensemble.best_model("mae", minimize=True)
        expected = best_member.loaded_model.model.predict([self.row])[0]

        predicted = self.ensemble.predict_best(self.row, "mae", minimize=True)

        self.assertEqual(predicted, expected)

    def test_best_model_raises_for_an_unknown_metric(self):

        with self.assertRaises(KeyError):
            self.ensemble.best_model("not_a_real_metric", minimize=True)


class ClassificationEnsembleTests(RealSavedModelsTestCase):

    def setUp(self):

        self.ensemble = Ensemble([
            EnsembleMember(load_model(self.bottleneck_location_dir), weight=1.0),
        ])

        Xb, _yb, _extra = at.BottleneckModel.build_table(self.dataset, target="location")
        self.row = Xb[0]

    def test_is_classifier_is_true(self):

        self.assertTrue(self.ensemble.is_classifier)

    def test_majority_vote_with_a_single_member_matches_that_members_own_prediction(self):

        expected = self.ensemble.members[0].loaded_model.model.predict([self.row])[0]

        vote = self.ensemble.predict_majority_vote(self.row)

        self.assertEqual(vote, expected)

    def test_predict_majority_vote_raises_for_a_regression_ensemble(self):

        regression_ensemble = Ensemble([EnsembleMember(load_model(self.evac_rf_dir))])

        with self.assertRaises(TypeError):
            regression_ensemble.predict_majority_vote(self.row)

    def test_majority_vote_is_deterministic(self):

        first = self.ensemble.predict_majority_vote(self.row)
        second = self.ensemble.predict_majority_vote(self.row)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
