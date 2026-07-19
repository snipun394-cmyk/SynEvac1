from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ai_training.models.base import BaseModel


# =====================================================
# Phase 1 -- Feature Importance. Impurity-based: reads
# estimator.feature_importances_ directly off an already-fitted Random
# Forest/Gradient Boosting/XGBoost model (ai_explainability never fits
# or refits anything). Model-agnostic importance (any estimator, not
# just tree ensembles) lives in permutation_importance.py instead.
# =====================================================


def extract_raw_importances(estimator) -> np.ndarray:

    # Works for a plain RandomForestRegressor/Classifier,
    # GradientBoostingRegressor/Classifier, or XGBRegressor/Classifier
    # (all four expose feature_importances_ directly), and for a
    # MultiOutputRegressor wrapping one of those per output
    # (ExitUsageModel's non-random_forest algorithms) by averaging each
    # sub-estimator's own feature_importances_ across outputs.

    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_, dtype=float)

    if hasattr(estimator, "estimators_"):

        per_output = [extract_raw_importances(sub) for sub in estimator.estimators_]

        return np.mean(per_output, axis=0)

    raise TypeError(
        f"{type(estimator).__name__} does not expose feature_importances_ -- "
        "only Random Forest/Gradient Boosting/XGBoost (or a MultiOutputRegressor "
        "wrapping one of those) are supported here."
    )


# =====================================================


def get_feature_importances(model: BaseModel, *, aggregate_encoded: bool = True) -> Dict[str, float]:

    # Maps a fitted model's raw feature_importances_ (computed over the
    # Preprocessor's TRANSFORMED feature space -- one-hot expanded
    # categoricals included) back onto meaningful names.
    #
    # aggregate_encoded=True (the default) sums a one-hot-encoded
    # categorical column's per-category importances back onto its
    # ORIGINAL column name (e.g. the transformed names "Door_3_State=
    # OPEN"/"Door_3_State=CLOSED"/"Door_3_State=LOCKED" collapse back
    # to one "Door_3_State" entry) -- matching how a fire engineer
    # reads a ranked list ("Door 3 Locked" is one fact about Door 3,
    # not three independent features).

    if model.preprocessor is None:
        raise RuntimeError(f"{type(model).__name__} must be fit before computing feature importances")

    names = model.preprocessor.feature_names_out()
    importances = extract_raw_importances(model.estimator)

    raw = dict(zip(names, importances.tolist()))

    if not aggregate_encoded:
        return raw

    aggregated: Dict[str, float] = {}

    for name, value in raw.items():

        original_column = name.split("=", 1)[0]
        aggregated[original_column] = aggregated.get(original_column, 0.0) + value

    return aggregated


# =====================================================


@dataclass(frozen=True)
class RankedFeature:

    rank: int
    feature: str
    importance: float


# =====================================================


def rank_feature_importances(
    model: BaseModel, *, top_n: Optional[int] = None, aggregate_encoded: bool = True,
) -> List[RankedFeature]:

    importances = get_feature_importances(model, aggregate_encoded=aggregate_encoded)

    # Sorted by (descending importance, ascending name) -- deterministic
    # even when several features tie exactly (e.g. all-zero importance),
    # rather than relying on whatever order a dict happened to iterate in.
    ordered = sorted(importances.items(), key=lambda item: (-item[1], item[0]))

    if top_n is not None:
        ordered = ordered[:top_n]

    return [
        RankedFeature(rank=index + 1, feature=name, importance=value)
        for index, (name, value) in enumerate(ordered)
    ]
