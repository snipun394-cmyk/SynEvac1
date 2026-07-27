import unittest

import numpy as np

from predictive_model.baselines import DecisionTreeBaseline
from predictive_model.sanity_checks import (
    feature_family_ablation_report,
    label_shuffle_test,
    leakage_correlation_recheck,
)


# =====================================================
# Localized Predictive Model V2 milestone, Phase 21 -- "shuffled-label
# sanity machinery" gets its own dedicated unit test here (V1 only ever
# exercised predictive_model.sanity_checks end-to-end via its training
# script, with no unit test file of its own). A tiny, fully synthetic
# fixture is used deliberately: these tests must never require
# retraining against the real campaign-scale dataset. A REAL sklearn
# learner (DecisionTreeBaseline, already in this package) is used here
# rather than a hand-rolled toy model -- a toy model that just rescales
# one column by label-shuffled constants can stay monotonic in that
# column regardless of shuffling, which would defeat the point of this
# test (an earlier draft of this fixture had exactly that bug).
# =====================================================


def _train_fn(X, y):
    return DecisionTreeBaseline(max_depth=4, seed=1).fit(X, y)


def _synthetic_xy(n=400, seed=0):

    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    # column 0 is genuinely informative (shifted by label); column 1 is
    # pure noise, uncorrelated with y.
    informative = y * 5.0 + rng.normal(0, 1.0, size=n)
    noise = rng.normal(0, 1.0, size=n)
    X = np.column_stack([informative, noise])
    return X, y


class SanityChecksTests(unittest.TestCase):

    def setUp(self):
        self.X_train, self.y_train = _synthetic_xy(n=1500, seed=1)
        self.X_val, self.y_val = _synthetic_xy(n=500, seed=2)

    def test_label_shuffle_collapses_towards_chance(self):

        result = label_shuffle_test(_train_fn, self.X_train, self.y_train, self.X_val, self.y_val, seed=1)

        self.assertIn("shuffled_label_roc_auc_on_real_val_labels", result)
        self.assertIn("near_chance", result)
        # a model trained on SHUFFLED labels, scored against REAL val
        # labels, must not do meaningfully better than chance -- a
        # shallow decision tree on a modest sample can still pick up a
        # little spurious signal from shuffled noise, so the tolerance
        # here is generous (0.2) rather than V1's own tight real-data
        # finding (0.499) -- the point of this unit test is "clearly far
        # from the ~0.8+ real-signal case below", not exact replication.
        self.assertLess(abs(result["shuffled_label_roc_auc_on_real_val_labels"] - 0.5), 0.2)

    def test_label_shuffle_result_is_deterministic_given_seed(self):

        first = label_shuffle_test(_train_fn, self.X_train, self.y_train, self.X_val, self.y_val, seed=42)
        second = label_shuffle_test(_train_fn, self.X_train, self.y_train, self.X_val, self.y_val, seed=42)

        self.assertEqual(
            first["shuffled_label_roc_auc_on_real_val_labels"],
            second["shuffled_label_roc_auc_on_real_val_labels"],
        )

    def test_unshuffled_model_clears_chance_on_the_same_fixture(self):
        """Sanity-check the fixture itself: a model trained on the REAL
        (unshuffled) train labels must clearly beat chance on val -- so
        the label_shuffle_test collapse above is a real finding, not an
        artifact of an uninformative fixture."""

        model = _train_fn(self.X_train, self.y_train)
        probs = model.predict_proba(self.X_val)

        from sklearn.metrics import roc_auc_score
        real_roc_auc = roc_auc_score(self.y_val, probs)

        self.assertGreater(real_roc_auc, 0.8)

    def test_leakage_correlation_recheck_flags_a_near_duplicate_target_column(self):

        # column 2 IS the target (cast to float) -- a deliberately
        # planted leakage channel this check must catch.
        X_with_leak = np.column_stack([self.X_train, self.y_train.astype(float)])
        feature_names = ("informative", "noise", "leaked_target")

        result = leakage_correlation_recheck(X_with_leak, self.y_train, feature_names)

        self.assertIn("leaked_target", result["flagged_for_leakage_review"])

    def test_leakage_correlation_recheck_does_not_flag_noise_column(self):

        feature_names = ("informative", "noise")
        result = leakage_correlation_recheck(self.X_train, self.y_train, feature_names)

        self.assertNotIn("noise", result["flagged_for_leakage_review"])

    def test_feature_family_ablation_reports_a_drop_for_the_informative_family(self):

        families = {"informative_family": ["informative"], "noise_family": ["noise"]}
        feature_names = ("informative", "noise")

        report = feature_family_ablation_report(
            _train_fn, self.X_train, self.y_train, self.X_val, self.y_val, feature_names, families,
        )

        self.assertIn("informative_family", report["by_family"])
        self.assertIn("noise_family", report["by_family"])
        # removing the genuinely informative column should hurt ROC-AUC
        # more than removing the pure-noise column.
        informative_drop = report["by_family"]["informative_family"]["roc_auc_drop"]
        noise_drop = report["by_family"]["noise_family"]["roc_auc_drop"]
        self.assertGreater(informative_drop, noise_drop)


if __name__ == "__main__":
    unittest.main()
