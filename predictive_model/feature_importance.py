from typing import Any, Dict, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


# =====================================================
# First Localized Predictive Congestion Model milestone, Phase 8 --
# feature importance. Two methods, deliberately kept separate rather
# than blended into one number:
#   - built-in: model.feature_importances_, when the model type exposes
#     one (RandomForest, XGBoost -- NOT HistGradientBoostingClassifier,
#     which sklearn does not give this attribute to).
#   - permutation: model-agnostic (works for every model in this
#     package, including HistGradientBoosting/baselines), answers "how
#     much does shuffling this one column hurt ROC-AUC" directly against
#     THIS model's predict_proba, not a s a proxy split-count metric --
#     the PREFERRED number per Phase 8's own "permutation importance
#     (preferred if practical)" instruction. Implemented manually here
#     (not sklearn.inspection.permutation_importance) because this
#     package's model interface (predict_proba returning a 1D positive-
#     class array, not sklearn's 2-column convention) isn't what
#     sklearn's own scorer machinery expects.
# =====================================================


def builtin_feature_importance(model, feature_names: Sequence[str]) -> Dict[str, float]:

    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            f"{type(model).__name__} has no feature_importances_ -- use permutation_importance_report instead."
        )

    importances = model.feature_importances_
    paired = sorted(zip(feature_names, importances.tolist()), key=lambda kv: -kv[1])
    return dict(paired)


def permutation_importance_report(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    *,
    n_repeats: int = 5,
    sample_size: int = 20000,
    seed: int = 20260726,
) -> Dict[str, Any]:
    """Permutation importance against ROC-AUC, computed on a (seeded,
    reproducible) subsample of X/y for practicality -- at this
    campaign's row counts, computing on the full split would mean
    len(feature_names) * n_repeats full-dataset predict_proba calls."""

    rng = np.random.default_rng(seed)
    n = len(X)

    if n > sample_size:
        sample_idx = rng.choice(n, size=sample_size, replace=False)
        X_sample, y_sample = X[sample_idx], y[sample_idx]
    else:
        X_sample, y_sample = X, y

    baseline_score = float(roc_auc_score(y_sample, model.predict_proba(X_sample)))

    importances: Dict[str, Any] = {}

    for feature_index, feature_name in enumerate(feature_names):

        score_drops = []

        for _ in range(n_repeats):

            X_permuted = X_sample.copy()
            permuted_order = rng.permutation(len(X_permuted))
            X_permuted[:, feature_index] = X_permuted[permuted_order, feature_index]

            permuted_score = float(roc_auc_score(y_sample, model.predict_proba(X_permuted)))
            score_drops.append(baseline_score - permuted_score)

        importances[feature_name] = {
            "mean_importance": float(np.mean(score_drops)),
            "std_importance": float(np.std(score_drops)),
        }

    ranked = dict(sorted(importances.items(), key=lambda kv: -kv[1]["mean_importance"]))

    return {
        "baseline_roc_auc": baseline_score,
        "sample_size": len(X_sample),
        "n_repeats": n_repeats,
        "importances": ranked,
    }
