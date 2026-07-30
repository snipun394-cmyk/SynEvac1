from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


# =====================================================
# Prediction vs Reality Evaluation Framework milestone, Phase 4 --
# regression metrics. Pure functions over plain (y_true, y_pred) lists,
# same genericity discipline as classification_metrics.py.
# =====================================================


@dataclass(frozen=True)
class RegressionMetrics:

    sample_count: int

    mae: Optional[float] = None
    rmse: Optional[float] = None

    # None whenever ANY true value is exactly 0 -- MAPE is undefined
    # (division by zero) in that case, never silently skipped/clamped
    # (Phase 4's own "MAPE where meaningful" instruction, taken
    # literally).
    mape: Optional[float] = None

    bias: Optional[float] = None  # mean(predicted - actual); positive = over-prediction

    mean_error: Optional[float] = None
    median_error: Optional[float] = None
    std_error: Optional[float] = None

    worst_case_error: Optional[float] = None
    best_case_error: Optional[float] = None

    # 95% CI on the mean SIGNED error, normal-approximation
    # (mean +/- 1.96 * standard_error) -- a standard, disclosed
    # approximation, not a fabricated exact interval; None below 2
    # samples (no honest variance estimate).
    error_ci_95_low: Optional[float] = None
    error_ci_95_high: Optional[float] = None


def compute_regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> RegressionMetrics:

    sample_count = len(y_true)

    if sample_count == 0:
        return RegressionMetrics(sample_count=0)

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    errors = y_pred_arr - y_true_arr  # signed: predicted - actual
    abs_errors = np.abs(errors)

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(mean_squared_error(y_true_arr, y_pred_arr) ** 0.5)

    mape = None
    if not np.any(y_true_arr == 0.0):
        mape = float(mean_absolute_percentage_error(y_true_arr, y_pred_arr))

    bias = float(errors.mean())
    mean_error = float(errors.mean())
    median_error = float(np.median(errors))

    std_error = None
    error_ci_low = None
    error_ci_high = None

    if sample_count >= 2:

        std_error = float(errors.std(ddof=1))
        standard_error_of_mean = std_error / (sample_count ** 0.5)
        error_ci_low = mean_error - 1.96 * standard_error_of_mean
        error_ci_high = mean_error + 1.96 * standard_error_of_mean

    worst_case_error = float(abs_errors.max())
    best_case_error = float(abs_errors.min())

    return RegressionMetrics(
        sample_count=sample_count,
        mae=mae, rmse=rmse, mape=mape, bias=bias,
        mean_error=mean_error, median_error=median_error, std_error=std_error,
        worst_case_error=worst_case_error, best_case_error=best_case_error,
        error_ci_95_low=error_ci_low, error_ci_95_high=error_ci_high,
    )
