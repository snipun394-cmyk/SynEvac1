from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# =====================================================
# First Localized Predictive Congestion Model milestone, Phase 3 --
# simple baselines. Every model in this package (baselines and Phase 4
# tree models alike) shares the same tiny interface: fit(X, y,
# sample_weight=None) and predict_proba(X) -> 1D array of P(target=1).
# The point of these baselines is answering "does ML add value at
# all" -- if a tree-based model can't beat MajorityClass/AlwaysNegative
# by a wide margin on PR-AUC (14.8% overall positive rate -- see
# docs/architecture/predictive_dataset_campaign_v1.md Section 3), it
# isn't doing anything useful.
# =====================================================


class MajorityClassBaseline:
    """Predicts the training set's majority class for every row, expressed
    as a constant probability (1.0 if positive was the majority class,
    0.0 otherwise) -- not a tuned probability, an honest baseline."""

    name = "majority_class"

    def __init__(self) -> None:
        self._constant_proba: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "MajorityClassBaseline":
        positive_rate = float(np.mean(y))
        self._constant_proba = 1.0 if positive_rate >= 0.5 else 0.0
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.full(shape=(len(X),), fill_value=self._constant_proba, dtype=float)


class AlwaysNegativeBaseline:
    """Predicts P(target=1) = 0.0 for every row -- the "do nothing" baseline
    a real dataset with a 14.8% overall positive rate should already beat
    handily on recall/F1, even though it looks deceptively strong on raw
    accuracy alone (this is exactly why balanced accuracy / PR-AUC matter
    more than accuracy for an imbalanced problem like this one)."""

    name = "always_negative"

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "AlwaysNegativeBaseline":
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(shape=(len(X),), dtype=float)


class RandomBaseline:
    """Predicts an independent Uniform(0, 1) probability per row, seeded
    for reproducibility -- the "a model with zero information about the
    input" baseline. Any real model should clear this by a wide margin
    on ROC-AUC (random's expectation is exactly 0.5)."""

    name = "random"

    def __init__(self, seed: int = 20260726) -> None:
        self._rng = np.random.default_rng(seed)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "RandomBaseline":
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._rng.uniform(0.0, 1.0, size=len(X))


class LogisticRegressionBaseline:
    """A simple, standardized-input logistic regression -- the "can a
    linear model separate this" baseline, between the trivial baselines
    above and the Phase 4 tree-based models."""

    name = "logistic_regression"

    def __init__(self, class_weight: Optional[str] = "balanced", seed: int = 20260726) -> None:
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._model = LogisticRegression(max_iter=2000, class_weight=class_weight, random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "LogisticRegressionBaseline":
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std == 0.0] = 1.0
        X_scaled = (X - self._mean) / self._std
        self._model.fit(X_scaled, y, sample_weight=sample_weight)
        return self

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mean) / self._std

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(self._scale(X))[:, 1]


class DecisionTreeBaseline:
    """A single, shallow decision tree -- deliberately NOT depth-tuned
    (max_depth=6), so it stays a baseline rather than competing with
    Phase 4's actually-tuned tree ensembles."""

    name = "decision_tree"

    def __init__(self, max_depth: int = 6, class_weight: Optional[str] = "balanced", seed: int = 20260726) -> None:
        self._model = DecisionTreeClassifier(max_depth=max_depth, class_weight=class_weight, random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "DecisionTreeBaseline":
        self._model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]


class DeterministicCurrentStateBaseline:
    """Localized Predictive Model V3 milestone, Phase 29 -- SynEvac
    already has deterministic live intelligence (crowd_intelligence's
    own congestion_level classification and trend direction). This
    baseline answers "how well would an operator do using ONLY that
    already-existing current-state read, with no learning at all" --
    the real bar Model V3 must clear, not just beating a trivial
    majority-class/random baseline. Purely a fixed RULE over two
    already-one-hot-encoded feature columns (candidate_congestion_level,
    candidate_congestion_trend) -- fit() does nothing (no parameters are
    ever learned from data), and NO new intelligence engine is created:
    this reads the exact same categorical fields simulation_extractor.py/
    live_extractor.py already produce from crowd_intelligence's own
    congestion_level/trend computation.

    score = congestion_level_rank(0-4) + 1.0 if trend==RISING else
            (+0.5 if trend==UNKNOWN else +0.0), normalized to [0, 1] by
    dividing by the maximum possible score (5.0) -- a simple, fully
    transparent, monotonic ordinal score usable for ROC-AUC/PR-AUC
    exactly like any real classifier's predict_proba() output, without
    ever being fit to labels."""

    name = "deterministic_current_state"

    _CONGESTION_LEVEL_RANK: Dict[str, int] = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "VERY_HIGH": 3, "CRITICAL": 4}
    _MAX_SCORE = 5.0  # CRITICAL (4) + RISING (1)

    def __init__(self, feature_names: Sequence[str]) -> None:

        feature_names = list(feature_names)

        self._level_column_indices = {
            level: feature_names.index(f"candidate_congestion_level={level}")
            for level in self._CONGESTION_LEVEL_RANK
            if f"candidate_congestion_level={level}" in feature_names
        }
        self._trend_rising_index = (
            feature_names.index("candidate_congestion_trend=RISING")
            if "candidate_congestion_trend=RISING" in feature_names else None
        )
        self._trend_unknown_index = (
            feature_names.index("candidate_congestion_trend=UNKNOWN")
            if "candidate_congestion_trend=UNKNOWN" in feature_names else None
        )

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "DeterministicCurrentStateBaseline":
        # Deliberately a no-op -- every parameter of this rule is fixed
        # in advance from existing feature semantics, never learned.
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        score = np.zeros(len(X), dtype=float)

        for level, rank in self._CONGESTION_LEVEL_RANK.items():
            column = self._level_column_indices.get(level)
            if column is not None:
                score += X[:, column] * rank

        if self._trend_rising_index is not None:
            score += X[:, self._trend_rising_index] * 1.0
        if self._trend_unknown_index is not None:
            score += X[:, self._trend_unknown_index] * 0.5

        return np.clip(score / self._MAX_SCORE, 0.0, 1.0)


def build_baselines(seed: int = 20260726):
    return {
        "majority_class": MajorityClassBaseline(),
        "always_negative": AlwaysNegativeBaseline(),
        "random": RandomBaseline(seed=seed),
        "logistic_regression": LogisticRegressionBaseline(seed=seed),
        "decision_tree": DecisionTreeBaseline(seed=seed),
    }
