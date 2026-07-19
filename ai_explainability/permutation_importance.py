from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ai_training.metrics import (
    classification_metrics,
    multioutput_regression_metrics,
    regression_metrics,
)
from ai_training.models.base import BaseModel

from ai_explainability.feature_importance import RankedFeature


# =====================================================
# Phase 1 -- model-agnostic feature importance. Unlike
# feature_importance.py (impurity-based, requires estimator.
# feature_importances_ -- tree ensembles only), this shuffles one RAW
# feature column at a time across already-fit-and-held-out rows and
# measures the drop in a scoring function -- works for any BaseModel
# regardless of what its underlying estimator exposes. Never refits
# the model; only calls predict()/predict_proba() on already-fitted
# models.
# =====================================================


ScoreFn = Callable[[BaseModel, Sequence[Dict[str, Any]], Sequence[Any]], float]


def default_score(model: BaseModel, X_rows: Sequence[Dict[str, Any]], y_true: Sequence[Any]) -> float:

    # "Higher is always better", so that (baseline - permuted) is a
    # positive importance whenever permuting a column hurts the model:
    #   - classification: accuracy
    #   - single-output regression: negative MAE
    #   - multi-output regression (ExitUsageModel): negative macro-
    #     average MAE
    # Reuses ai_training's own metric implementations rather than
    # recomputing them.

    predictions = model.predict(X_rows)

    if model.is_classifier:
        return classification_metrics(y_true, predictions)["accuracy"]

    if model.output_names is not None:

        y_pred = [[row[name] for name in model.output_names] for row in predictions]

        return -multioutput_regression_metrics(y_true, y_pred, model.output_names)["macro_average"]["mae"]

    return -regression_metrics(y_true, predictions)["mae"]


# =====================================================


def permutation_importance(
    model: BaseModel,
    X_rows: Sequence[Dict[str, Any]],
    y_true: Sequence[Any],
    *,
    score_fn: Optional[ScoreFn] = None,
    n_repeats: int = 5,
    random_state: int = 0,
) -> Dict[str, float]:

    score_fn = score_fn or default_score
    rng = np.random.default_rng(random_state)

    baseline = score_fn(model, X_rows, y_true)
    columns = sorted({key for row in X_rows for key in row.keys()})

    importances: Dict[str, float] = {}

    for column in columns:

        drops = []

        for _ in range(n_repeats):

            shuffled_rows = _shuffle_column(X_rows, column, rng)
            score = score_fn(model, shuffled_rows, y_true)
            drops.append(baseline - score)

        importances[column] = float(np.mean(drops))

    return importances


# =====================================================


def _shuffle_column(
    X_rows: Sequence[Dict[str, Any]], column: str, rng: np.random.Generator,
) -> List[Dict[str, Any]]:

    values = [row.get(column) for row in X_rows]
    permutation = rng.permutation(len(values))
    shuffled_values = [values[index] for index in permutation]

    return [
        {**row, column: shuffled_value}
        for row, shuffled_value in zip(X_rows, shuffled_values)
    ]


# =====================================================


def rank_permutation_importances(
    model: BaseModel,
    X_rows: Sequence[Dict[str, Any]],
    y_true: Sequence[Any],
    *,
    top_n: Optional[int] = None,
    score_fn: Optional[ScoreFn] = None,
    n_repeats: int = 5,
    random_state: int = 0,
) -> List[RankedFeature]:

    importances = permutation_importance(
        model, X_rows, y_true, score_fn=score_fn, n_repeats=n_repeats, random_state=random_state,
    )

    ordered = sorted(importances.items(), key=lambda item: (-item[1], item[0]))

    if top_n is not None:
        ordered = ordered[:top_n]

    return [
        RankedFeature(rank=index + 1, feature=name, importance=value)
        for index, (name, value) in enumerate(ordered)
    ]
