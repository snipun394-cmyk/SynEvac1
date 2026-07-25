from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from predictive_dataset.schema import field_by_name


# =====================================================
# First Localized Predictive Congestion Model milestone -- shared
# feature preparation. predictive_dataset/schema.py defines the frozen,
# versioned FEATURE set (CANDIDATE_FEATURE_NAMES); this module only
# converts those same named columns into numeric arrays a model can
# consume (one-hot encoding, missing-value imputation) -- it never
# adds, drops, or redefines a feature. Categories are FIXED vocabulary
# lists (not fit from data) so encoding is identical and leak-free
# across train/val/test and across horizons.
#
# EXCLUDED FROM MODEL INPUT, on purpose:
#   - scenario_id / observation_time / candidate_id: row identity, not
#     signal (predictive_dataset.schema.ROW_IDENTITY_COLUMNS's own
#     "bookkeeping, not trainable input features" rule, reused here).
#   - prediction_horizon: selects WHICH per-horizon dataset slice/model
#     this is -- not a feature within a single-horizon model.
#   - currently_congested: by construction always False for every
#     trainable row (target is None exactly when it's True) -- constant,
#     zero signal for anything this loader hands to a model.
#   - had_any_activity_in_window / target: computed from the future
#     window (time, time+horizon] by predictive_dataset.target_generator
#     -- these are LABEL-SIDE only. Feeding had_any_activity_in_window
#     into a model would leak the future outcome the model is supposed
#     to predict; this is exactly the leakage boundary
#     docs/architecture/localized_predictive_ai_dataset.md already
#     documents and tests/test_predictive_dataset_leakage_guards.py
#     already enforces at the predictive_dataset layer -- reused here,
#     not re-litigated.
# =====================================================


CANDIDATE_TYPE_CATEGORIES: Tuple[str, ...] = ("Door", "Exit", "Stair")
CONGESTION_LEVEL_CATEGORIES: Tuple[str, ...] = ("LOW", "MODERATE", "HIGH", "VERY_HIGH", "CRITICAL")

NUMERIC_FEATURES: Tuple[str, ...] = (
    "total_active_occupant_count",
    "candidate_capacity",
    "candidate_walking_distance",
    "candidate_adjacent_zone_occupancy",
    "candidate_queue_length",
    "candidate_approaching_count",
)

BOOL_FEATURES: Tuple[str, ...] = ("candidate_traversable",)

CATEGORICAL_FEATURES: Tuple[str, ...] = ("candidate_type", "candidate_congestion_level")


@dataclass(frozen=True)
class PreparedFeatures:

    X: np.ndarray
    y: np.ndarray
    scenario_ids: np.ndarray
    candidate_types: np.ndarray
    feature_names: Tuple[str, ...]


def trainable_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows where the target question is even applicable -- excludes
    rows currently_congested at observation time (target is None there
    by predictive_dataset.target_generator's own "not applicable"
    policy, see that module's CandidateLabel docstring)."""

    return frame[frame["target"].notna()].copy()


def build_feature_matrix(frame: pd.DataFrame) -> PreparedFeatures:
    """Turn a (already-trainable-filtered) frame into a numeric matrix.

    Missing numeric values (candidate_adjacent_zone_occupancy is the
    only field with real missingness at this campaign's scale, per
    docs/architecture/predictive_dataset_campaign_v1.md Section 5) are
    imputed with -1 (a sentinel value outside the valid occupancy range,
    never confusable with a real 0) plus an explicit "<feature>_missing"
    indicator column -- imputing with e.g. the mean would fabricate a
    plausible-looking value where none exists, which is exactly the
    "never fabricated as 0" discipline predictive_dataset/schema.py's
    own missing_value_note already commits to; this loader preserves
    that discipline in the model-input encoding rather than quietly
    discarding it during imputation. Missing categorical values
    (candidate_congestion_level can be None) get an explicit
    "<category>_missing" indicator column instead of a fabricated
    category membership.
    """

    # Whether to emit a "<feature>_missing" indicator column is decided
    # from the FROZEN SCHEMA's own nullable flag, never from whether
    # this particular frame happens to contain a missing value --
    # otherwise train/val/test (or different horizon slices) could
    # silently produce feature matrices with different column counts
    # depending on which split happened to draw a missing value.
    columns = []
    names = []

    for feature in NUMERIC_FEATURES:

        values = frame[feature].to_numpy(dtype=float)
        missing_mask = np.isnan(values)

        imputed = np.where(missing_mask, -1.0, values)
        columns.append(imputed)
        names.append(feature)

        if field_by_name(feature).nullable:
            columns.append(missing_mask.astype(float))
            names.append(f"{feature}_missing")

    for feature in BOOL_FEATURES:

        values = frame[feature].astype(bool).to_numpy(dtype=float)
        columns.append(values)
        names.append(feature)

    for feature in CATEGORICAL_FEATURES:

        categories = CANDIDATE_TYPE_CATEGORIES if feature == "candidate_type" else CONGESTION_LEVEL_CATEGORIES
        raw = frame[feature]

        for category in categories:
            indicator = (raw == category).to_numpy(dtype=float)
            columns.append(indicator)
            names.append(f"{feature}={category}")

        if field_by_name(feature).nullable:
            known_mask = raw.isin(categories).to_numpy()
            missing_mask = ~known_mask
            columns.append(missing_mask.astype(float))
            names.append(f"{feature}=__missing__")

    X = np.column_stack(columns)
    y = frame["target"].astype(bool).astype(int).to_numpy()
    scenario_ids = frame["scenario_id"].to_numpy()
    candidate_types = frame["candidate_type"].to_numpy()

    return PreparedFeatures(
        X=X, y=y, scenario_ids=scenario_ids, candidate_types=candidate_types,
        feature_names=tuple(names),
    )
