from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


# =====================================================
# First Localized Predictive Congestion Model milestone, Phase 10 --
# calibration. Both calibrators are fit on the VALIDATION split's
# (already-trained-model) predicted probabilities vs. true labels --
# never on train (the model already saw train) and never on test (test
# stays untouched until final evaluation, exactly like Phase 5's
# threshold tuning discipline).
# =====================================================


class PlattScaler:
    """Platt scaling -- a 1D logistic regression mapping raw model
    probability -> calibrated probability."""

    def __init__(self) -> None:
        self._lr = LogisticRegression()

    def fit(self, val_probs: np.ndarray, val_y: np.ndarray) -> "PlattScaler":
        self._lr.fit(val_probs.reshape(-1, 1), val_y)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(probs.reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Isotonic regression -- a non-parametric, monotonic calibration map.
    More flexible than Platt scaling, but needs enough validation rows
    per bin to avoid overfitting the calibration curve itself."""

    def __init__(self) -> None:
        self._iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, val_probs: np.ndarray, val_y: np.ndarray) -> "IsotonicCalibrator":
        self._iso.fit(val_probs, val_y)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        return self._iso.transform(probs)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 10) -> float:
    """Standard binned ECE: weighted mean absolute gap between each bin's
    mean predicted probability and its actual observed positive rate."""

    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_prob, bin_edges[1:-1], right=True), 0, n_bins - 1)

    total = len(y_true)
    ece = 0.0

    for bin_index in range(n_bins):

        mask = bin_indices == bin_index
        count = int(mask.sum())
        if count == 0:
            continue

        observed_rate = float(y_true[mask].mean())
        mean_predicted = float(y_prob[mask].mean())
        ece += (count / total) * abs(observed_rate - mean_predicted)

    return ece


@dataclass(frozen=True)
class CalibrationReport:

    before: Dict[str, Any]
    after_platt: Dict[str, Any]
    after_isotonic: Dict[str, Any]
    recommended_method: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "before": self.before,
            "after_platt": self.after_platt,
            "after_isotonic": self.after_isotonic,
            "recommended_method": self.recommended_method,
        }


def evaluate_calibration_methods(
    val_y: np.ndarray, val_prob: np.ndarray, test_y: np.ndarray, test_prob: np.ndarray,
) -> CalibrationReport:
    """Fit both calibrators on (val_prob, val_y), apply to test, and report
    Brier score + ECE before/after each -- the evidence Phase 10 asks for
    to decide whether calibration helps at all."""

    def _summarize(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        return {
            "brier_score": float(brier_score_loss(y_true, y_prob)),
            "expected_calibration_error": expected_calibration_error(y_true, y_prob),
        }

    before = _summarize(test_y, test_prob)

    platt = PlattScaler().fit(val_prob, val_y)
    test_prob_platt = platt.transform(test_prob)
    after_platt = _summarize(test_y, test_prob_platt)

    isotonic = IsotonicCalibrator().fit(val_prob, val_y)
    test_prob_isotonic = isotonic.transform(test_prob)
    after_isotonic = _summarize(test_y, test_prob_isotonic)

    candidates = {"none": before, "platt": after_platt, "isotonic": after_isotonic}
    recommended_method = min(candidates, key=lambda name: candidates[name]["brier_score"])

    return CalibrationReport(
        before=before, after_platt=after_platt, after_isotonic=after_isotonic,
        recommended_method=recommended_method,
    )
