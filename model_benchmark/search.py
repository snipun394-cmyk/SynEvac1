import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold

from prediction_evaluation.classification_metrics import compute_classification_metrics
from prediction_evaluation.regression_metrics import compute_regression_metrics


# =====================================================
# Predictive Model Development & Benchmark Campaign milestone, Phase 3 --
# reproducible hyperparameter search. A deterministic grid search (every
# combination in a fixed, explicit grid -- see model_benchmark/
# algorithms.py) with GroupKFold cross-validation (scenario-grouped, so
# no scenario's rows ever straddle a fold's own train/validation split --
# the same discipline ai_training.split.make_split already applies to
# the outer train/val/test split, reused here at the fold level).
# GroupKFold has no shuffling/random_state parameter at all -- it is
# already fully deterministic by construction, satisfying Phase 3's own
# "ensure all searches are deterministic" requirement without needing to
# fix an extra seed.
#
# Deliberately NOT sklearn's GridSearchCV: every model here is an
# ai_training.models.base.BaseModel subclass operating on
# List[Dict[str, Any]] rows (its own Preprocessor is fit fresh per
# fold, exactly like the outer train/test split already does) rather
# than a plain sklearn Estimator over a numpy array -- GridSearchCV
# assumes the latter. This module reimplements only the combination-
# enumeration and score-averaging GridSearchCV would otherwise do,
# never estimator fitting/prediction logic (that stays entirely in
# BaseModel.fit()/.predict()/.predict_proba()).
# =====================================================


@dataclass(frozen=True)
class SearchTrial:

    params: Dict[str, Any]
    fold_scores: Tuple[float, ...]
    mean_score: float
    std_score: float


@dataclass(frozen=True)
class SearchResult:

    algorithm: str
    scoring: str  # "f1" (classification) or "neg_mae" (regression) -- higher is always better
    trials: Tuple[SearchTrial, ...]
    best_params: Dict[str, Any]
    best_mean_score: float
    n_folds: int


def _param_combinations(grid: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:

    if not grid:
        return [{}]

    keys = sorted(grid.keys())  # deterministic order regardless of dict insertion order
    value_lists = [grid[key] for key in keys]

    return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]


def grid_search_classification(
    model_factory, X_rows: Sequence[Dict[str, Any]], y: Sequence[bool], groups: Sequence[str],
    param_grid: Dict[str, Sequence[Any]], *, n_folds: int = 3,
) -> SearchResult:

    combinations = _param_combinations(param_grid)
    n_folds_effective = min(n_folds, len(set(groups)))

    trials: List[SearchTrial] = []

    for params in combinations:

        fold_scores = []
        splitter = GroupKFold(n_splits=n_folds_effective)

        for train_idx, val_idx in splitter.split(np.arange(len(X_rows)), groups=list(groups)):

            X_train = [X_rows[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            X_val = [X_rows[i] for i in val_idx]
            y_val = [y[i] for i in val_idx]

            if len(set(y_train)) < 2:
                continue  # degenerate fold (single class) -- cannot meaningfully fit/score

            model = model_factory(params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            try:
                y_proba = model.predict_proba(X_val)
                positive_index = list(model.label_encoder.classes_).index(True) if True in list(model.label_encoder.classes_) else 1
                proba_positive = tuple(float(v) for v in y_proba[:, positive_index])
            except (AttributeError, TypeError):
                proba_positive = ()

            metrics = compute_classification_metrics([bool(v) for v in y_val], [bool(v) for v in y_pred], proba_positive)
            fold_scores.append(metrics.f1 if metrics.f1 is not None else 0.0)

        if not fold_scores:
            fold_scores = [0.0]

        trials.append(SearchTrial(
            params=params, fold_scores=tuple(fold_scores),
            mean_score=float(np.mean(fold_scores)), std_score=float(np.std(fold_scores)),
        ))

    best = max(trials, key=lambda t: t.mean_score)

    return SearchResult(
        algorithm="", scoring="f1", trials=tuple(trials),
        best_params=best.params, best_mean_score=best.mean_score, n_folds=n_folds_effective,
    )


def grid_search_regression(
    model_factory, X_rows: Sequence[Dict[str, Any]], y: Sequence[float], groups: Sequence[str],
    param_grid: Dict[str, Sequence[Any]], *, n_folds: int = 3,
) -> SearchResult:

    combinations = _param_combinations(param_grid)
    n_folds_effective = min(n_folds, len(set(groups)))

    trials: List[SearchTrial] = []

    for params in combinations:

        fold_scores = []
        splitter = GroupKFold(n_splits=n_folds_effective)

        for train_idx, val_idx in splitter.split(np.arange(len(X_rows)), groups=list(groups)):

            X_train = [X_rows[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            X_val = [X_rows[i] for i in val_idx]
            y_val = [y[i] for i in val_idx]

            model = model_factory(params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            metrics = compute_regression_metrics(y_val, y_pred)

            fold_scores.append(-metrics.mae if metrics.mae is not None else -float("inf"))

        trials.append(SearchTrial(
            params=params, fold_scores=tuple(fold_scores),
            mean_score=float(np.mean(fold_scores)), std_score=float(np.std(fold_scores)),
        ))

    best = max(trials, key=lambda t: t.mean_score)

    return SearchResult(
        algorithm="", scoring="neg_mae", trials=tuple(trials),
        best_params=best.params, best_mean_score=best.mean_score, n_folds=n_folds_effective,
    )
