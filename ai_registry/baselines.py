from typing import Any, Dict, List, Sequence

import numpy as np

from sklearn.dummy import DummyClassifier, DummyRegressor

from ai_training.metrics import classification_metrics, regression_metrics


# =====================================================
# Phase 6 -- trivial baselines every live-compatible model must beat
# before being considered anything more than EXPERIMENTAL. Reuses
# sklearn's own well-tested DummyRegressor/DummyClassifier (already a
# project dependency via ai_training) rather than hand-rolling mean/
# median/majority-class arithmetic, and the existing ai_training.metrics
# functions for scoring -- "use existing evaluation utilities where
# possible" (Phase 6's own instruction).
# =====================================================


def evacuation_time_baselines(
    y_train: Sequence[float], y_test: Sequence[float],
) -> Dict[str, Dict[str, float]]:

    results = {}

    for strategy in ("mean", "median"):

        dummy = DummyRegressor(strategy=strategy)
        dummy.fit(np.zeros((len(y_train), 1)), y_train)

        y_pred = dummy.predict(np.zeros((len(y_test), 1)))

        results[strategy] = regression_metrics(y_test, y_pred)

    return results


# =====================================================


def bottleneck_occurrence_baselines(
    y_train: Sequence[Any], y_test: Sequence[Any],
) -> Dict[str, Dict[str, Any]]:

    results = {}

    for strategy in ("most_frequent", "stratified"):

        dummy = DummyClassifier(strategy=strategy, random_state=0)
        dummy.fit(np.zeros((len(y_train), 1)), y_train)

        y_pred = dummy.predict(np.zeros((len(y_test), 1)))
        y_proba = dummy.predict_proba(np.zeros((len(y_test), 1)))

        results[strategy] = classification_metrics(y_test, y_pred, y_proba)

    return results


# =====================================================


def class_balance(y: Sequence[Any]) -> Dict[str, float]:

    values, counts = np.unique(np.asarray(list(y)), return_counts=True)
    total = len(y)

    return {str(value): count / total for value, count in zip(values, counts)}
