from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from sklearn.metrics import brier_score_loss

from ai_training.models.base import BaseModel


# =====================================================
# Phase 9 -- confidence/uncertainty. Never fabricates a confidence
# percentage: regression uncertainty is exposed only when the
# underlying estimator genuinely supports it (an ensemble of trees --
# the smallest architecture-compatible method, per this phase's own
# instruction to prefer the smallest option, ahead of building a
# separate quantile or conformal-prediction model); classification
# confidence is always the model's own raw predict_proba, retained
# as-is per this phase's own explicit instruction, with calibration
# quality reported (not silently assumed) alongside it.
# =====================================================


def regression_ensemble_uncertainty(
    model: BaseModel, X_rows: Sequence[Dict[str, Any]],
) -> Optional[List[float]]:

    # Per-row prediction std across a RandomForestRegressor's (or any
    # bagging_ensemble.estimators_-shaped model's) individual trees --
    # an honest, already-available-for-free uncertainty signal, never
    # invented. Returns None (never a fabricated 0.0) when the fitted
    # estimator has no such ensemble structure (e.g. gradient_boosting/
    # xgboost) -- a caller must then honestly report "confidence
    # unavailable", never guess.

    estimator = model.estimator

    if not hasattr(estimator, "estimators_") or model.preprocessor is None:
        return None

    X = model.preprocessor.transform(X_rows)

    try:
        per_tree_predictions = np.array([tree.predict(X) for tree in estimator.estimators_])
    except (AttributeError, TypeError):
        return None

    return per_tree_predictions.std(axis=0).tolist()


# =====================================================


def probability_calibration_report(
    y_true: Sequence[bool], y_proba_positive: Sequence[float], *, n_bins: int = 5,
) -> Dict[str, Any]:

    # A reliability check on the model's own raw predict_proba, NOT a
    # recalibration -- this milestone retains raw model probability as
    # the reported confidence (Phase 9's own instruction); this report
    # exists so that decision is an evaluated one, not an assumed one.
    # A future milestone finding poor calibration here is the trigger
    # for actually wrapping the estimator in something like
    # sklearn.calibration.CalibratedClassifierCV -- not attempted here.

    y_true = np.asarray(y_true, dtype=float)
    y_proba_positive = np.asarray(y_proba_positive, dtype=float)

    brier_score = float(brier_score_loss(y_true, y_proba_positive))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []

    for low, high in zip(bin_edges[:-1], bin_edges[1:]):

        mask = (y_proba_positive >= low) & (y_proba_positive < high if high < 1.0 else y_proba_positive <= high)

        if not mask.any():
            bins.append({"range": [float(low), float(high)], "count": 0, "observed_positive_rate": None})
            continue

        bins.append({
            "range": [float(low), float(high)],
            "count": int(mask.sum()),
            "observed_positive_rate": float(y_true[mask].mean()),
            "mean_predicted_probability": float(y_proba_positive[mask].mean()),
        })

    return {"brier_score": brier_score, "bins": bins}
