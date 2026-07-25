from typing import Dict, Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

try:
    import xgboost  # noqa: F401
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm  # noqa: F401
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost  # noqa: F401
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


# =====================================================
# First Localized Predictive Congestion Model milestone, Phase 4 --
# tree-based models. Library availability in THIS environment (checked
# at import time, not assumed):
#   xgboost   -- AVAILABLE (see XGBOOST_AVAILABLE)
#   lightgbm  -- NOT INSTALLED (see LIGHTGBM_AVAILABLE) -- not evaluated,
#                documented rather than silently skipped
#   catboost  -- NOT INSTALLED (see CATBOOST_AVAILABLE) -- same
#   Random Forest / Gradient Boosting -- both from sklearn, always
#                available
#
# "Gradient Boosting" is implemented as sklearn's
# HistGradientBoostingClassifier, NOT the classic
# sklearn.ensemble.GradientBoostingClassifier: the classic
# implementation builds one tree at a time with no histogram binning,
# which is impractical at this campaign's row counts (hundreds of
# thousands of trainable rows per horizon); HistGradientBoosting is
# sklearn's own modern, LightGBM-style histogram-based successor
# intended for exactly this data scale, and reports the same family of
# feature_importances_-style splits information. Documented substitution,
# not a silent one.
# =====================================================


class RandomForestModel:

    name = "random_forest"

    def __init__(self, seed: int = 20260726, class_weight: Optional[str] = "balanced", n_estimators: int = 300) -> None:
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_leaf=5,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "RandomForestModel":
        self._model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_

    @property
    def underlying(self):
        return self._model


class HistGradientBoostingModel:

    name = "gradient_boosting"  # HistGradientBoostingClassifier -- see module docstring for why

    def __init__(self, seed: int = 20260726, class_weight: Optional[str] = "balanced", max_iter: int = 300) -> None:
        self._model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "HistGradientBoostingModel":
        self._model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    @property
    def underlying(self):
        return self._model


class XGBoostModel:

    name = "xgboost"

    def __init__(self, seed: int = 20260726, n_estimators: int = 300) -> None:
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is not installed in this environment.")
        self._n_estimators = n_estimators
        self._seed = seed
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "XGBoostModel":

        from xgboost import XGBClassifier

        # scale_pos_weight is xgboost's own class-imbalance handling
        # (Phase 5) -- ratio of negative to positive training examples.
        n_pos = float(np.sum(y == 1))
        n_neg = float(np.sum(y == 0))
        scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

        self._model = XGBClassifier(
            n_estimators=self._n_estimators,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight if sample_weight is None else 1.0,
            random_state=self._seed,
            n_jobs=-1,
        )
        self._model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_

    @property
    def underlying(self):
        return self._model


def build_tree_models(seed: int = 20260726) -> Dict[str, object]:

    models = {
        "random_forest": RandomForestModel(seed=seed),
        "gradient_boosting": HistGradientBoostingModel(seed=seed),
    }

    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBoostModel(seed=seed)

    return models


def library_availability_report() -> Dict[str, bool]:
    return {
        "xgboost": XGBOOST_AVAILABLE,
        "lightgbm": LIGHTGBM_AVAILABLE,
        "catboost": CATBOOST_AVAILABLE,
        "random_forest_sklearn": True,
        "gradient_boosting_sklearn_hist": True,
    }
