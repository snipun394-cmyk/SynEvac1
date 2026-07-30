import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ai_training.preprocessing import FeatureSchema
from prediction_evaluation.classification_metrics import compute_classification_metrics
from prediction_evaluation.regression_metrics import compute_regression_metrics


# =====================================================
# Predictive Model Development & Benchmark Campaign milestone, Phase 6 --
# feature importance. Two independent, complementary techniques:
#
#   1. NATIVE importance -- read directly off the fitted estimator
#      (.feature_importances_ for tree ensembles, |.coef_| for linear
#      models) where the model type supports it. Reported against the
#      Preprocessor's own one-hot-expanded feature_names_out() (the
#      exact columns the estimator actually saw).
#   2. PERMUTATION importance -- model-agnostic, works identically for
#      every algorithm including MLP/Dummy which have no native
#      importance concept. Operates at the RAW canonical-feature level
#      (shuffling one whole ai_features column's values across the
#      evaluation rows and re-scoring via the model's own
#      predict()/predict_proba(), never reaching into the one-hot-
#      expanded numpy representation) -- more interpretable (one number
#      per real feature, e.g. "total_occupant_count", not one per
#      one-hot dummy) and reuses BaseModel's own public interface,
#      never sklearn.inspection.permutation_importance (which assumes a
#      plain array-in/array-out estimator, not this codebase's
#      dict-row BaseModel contract).
#
# Explicitly does NOT modify ai_features/feature_extraction in any way
# -- this module only ever READS already-extracted rows.
# =====================================================


def native_importance(model) -> Optional[Dict[str, float]]:

    estimator = model.estimator
    names = model.preprocessor.feature_names_out()

    if hasattr(estimator, "feature_importances_"):

        values = np.asarray(estimator.feature_importances_, dtype=float)
        return dict(zip(names, values.tolist()))

    if hasattr(estimator, "coef_"):

        coef = np.asarray(estimator.coef_, dtype=float)
        if coef.ndim > 1:
            coef = coef[0]  # binary classification's own single decision boundary

        return dict(zip(names, np.abs(coef).tolist()))

    return None  # e.g. MLP, Dummy -- no native importance concept


# =====================================================


def permutation_importance_classification(
    model, X_rows: Sequence[Dict[str, Any]], y_true: Sequence[bool], feature_names: Sequence[str],
    *, n_repeats: int = 3, random_state: int = 0,
) -> Dict[str, float]:

    rng = np.random.RandomState(random_state)

    baseline_pred = model.predict(X_rows)
    baseline_proba = _positive_proba(model, X_rows)
    baseline_score = compute_classification_metrics([bool(v) for v in y_true], [bool(v) for v in baseline_pred], baseline_proba).f1 or 0.0

    importances = {}

    for feature in feature_names:

        drops = []

        for _ in range(n_repeats):

            permuted_rows = _permute_column(X_rows, feature, rng)
            pred = model.predict(permuted_rows)
            proba = _positive_proba(model, permuted_rows)

            score = compute_classification_metrics([bool(v) for v in y_true], [bool(v) for v in pred], proba).f1 or 0.0
            drops.append(baseline_score - score)

        importances[feature] = float(np.mean(drops))

    return importances


def permutation_importance_regression(
    model, X_rows: Sequence[Dict[str, Any]], y_true: Sequence[float], feature_names: Sequence[str],
    *, n_repeats: int = 3, random_state: int = 0,
) -> Dict[str, float]:

    rng = np.random.RandomState(random_state)

    baseline_pred = model.predict(X_rows)
    baseline_mae = compute_regression_metrics(y_true, baseline_pred).mae or 0.0

    importances = {}

    for feature in feature_names:

        increases = []

        for _ in range(n_repeats):

            permuted_rows = _permute_column(X_rows, feature, rng)
            pred = model.predict(permuted_rows)

            mae = compute_regression_metrics(y_true, pred).mae or 0.0
            increases.append(mae - baseline_mae)  # positive = feature mattered (removing it hurt)

        importances[feature] = float(np.mean(increases))

    return importances


def _permute_column(X_rows: Sequence[Dict[str, Any]], feature: str, rng: np.random.RandomState) -> List[Dict[str, Any]]:

    values = [row.get(feature) for row in X_rows]
    order = rng.permutation(len(values))
    shuffled = [values[i] for i in order]

    permuted = []
    for row, new_value in zip(X_rows, shuffled):
        new_row = dict(row)
        new_row[feature] = new_value
        permuted.append(new_row)

    return permuted


def _positive_proba(model, X_rows: Sequence[Dict[str, Any]]) -> Tuple[float, ...]:

    try:
        proba = model.predict_proba(X_rows)
        positive_index = list(model.label_encoder.classes_).index(True) if True in list(model.label_encoder.classes_) else 1
        return tuple(float(v) for v in proba[:, positive_index])
    except (AttributeError, TypeError):
        return ()


# =====================================================


@dataclass(frozen=True)
class CorrelatedPair:

    feature_a: str
    feature_b: str
    correlation: float


def correlation_analysis(X_rows: Sequence[Dict[str, Any]], *, threshold: float = 0.8) -> Tuple[Tuple[CorrelatedPair, ...], Tuple[str, ...]]:

    # Numeric-only (categorical/boolean-string columns are excluded --
    # Pearson correlation over one-hot dummies is not a meaningful
    # question the same way it is for genuinely continuous/count
    # features). Reuses ai_training.preprocessing.FeatureSchema's own
    # numeric-vs-categorical classification rather than reimplementing
    # a second one.

    schema = FeatureSchema.infer(X_rows)
    numeric_columns = schema.numeric_columns

    if len(numeric_columns) < 2:
        return (), numeric_columns

    matrix = np.array([
        [float(row.get(col)) if row.get(col) is not None else np.nan for col in numeric_columns]
        for row in X_rows
    ])

    # Median-impute NaNs column-wise (matches Preprocessor's own numeric
    # imputation strategy) purely for this correlation computation --
    # never mutates the actual rows a model trains on.
    col_medians = np.nanmedian(matrix, axis=0)
    inds = np.where(np.isnan(matrix))
    matrix[inds] = np.take(col_medians, inds[1])

    corr = np.corrcoef(matrix, rowvar=False)

    pairs = []
    for i in range(len(numeric_columns)):
        for j in range(i + 1, len(numeric_columns)):

            value = corr[i, j]
            if np.isfinite(value) and abs(value) >= threshold:
                pairs.append(CorrelatedPair(feature_a=numeric_columns[i], feature_b=numeric_columns[j], correlation=float(value)))

    pairs.sort(key=lambda p: -abs(p.correlation))

    return tuple(pairs), numeric_columns
