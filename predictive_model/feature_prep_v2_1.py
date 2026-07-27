from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from predictive_model.feature_prep import (
    BOOL_FEATURES,
    CANDIDATE_TYPE_CATEGORIES,
    CATEGORICAL_FEATURES,
    CONGESTION_LEVEL_CATEGORIES,
    NUMERIC_FEATURES,
    PreparedFeatures,
)
from predictive_dataset.schema import field_by_name


# =====================================================
# Localized Predictive Model V2.1 milestone, Phase 11 -- EXPERIMENTAL
# feature-matrix builder for the 3 new fields
# (predictive_dataset.simulation_extractor_v2_1). Does NOT modify
# predictive_model/feature_prep.py (the frozen V1/V2 builder) -- this
# is an additive variant used ONLY by the targeted V2.1 hypothesis-test
# campaign, reusing that module's own NUMERIC_FEATURES/CATEGORICAL_
# FEATURES/BOOL_FEATURES/PreparedFeatures rather than redefining them.
#
# candidate_recent_flow_rate/candidate_alternative_route_count are
# NEVER null (always a real, computed integer -- see
# simulation_extractor_v2_1.py's own docstrings) so neither gets a
# "_missing" indicator column, unlike the frozen schema's nullable
# numeric fields. candidate_congestion_trend's "UNKNOWN" is a real,
# meaningful category (no earlier observation existed) -- not treated
# as a missing-value indicator the way schema.py's nullable
# categoricals are.
# =====================================================


EXPERIMENTAL_NUMERIC_FEATURES: Tuple[str, ...] = (
    "candidate_recent_flow_rate",
    "candidate_alternative_route_count",
)

CONGESTION_TREND_CATEGORIES: Tuple[str, ...] = ("RISING", "STABLE", "FALLING", "UNKNOWN")


def trainable_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["target"].notna()].copy()


def build_experimental_feature_matrix(frame: pd.DataFrame) -> PreparedFeatures:
    """Same discipline as predictive_model.feature_prep.build_feature_matrix,
    extended with the 3 V2.1 experimental fields."""

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

    for feature in EXPERIMENTAL_NUMERIC_FEATURES:

        values = frame[feature].to_numpy(dtype=float)
        columns.append(values)
        names.append(feature)

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

    raw_trend = frame["candidate_congestion_trend"]
    for category in CONGESTION_TREND_CATEGORIES:
        indicator = (raw_trend == category).to_numpy(dtype=float)
        columns.append(indicator)
        names.append(f"candidate_congestion_trend={category}")

    X = np.column_stack(columns)
    y = frame["target"].astype(bool).astype(int).to_numpy()
    scenario_ids = frame["scenario_id"].to_numpy()
    candidate_types = frame["candidate_type"].to_numpy()

    return PreparedFeatures(
        X=X, y=y, scenario_ids=scenario_ids, candidate_types=candidate_types,
        feature_names=tuple(names),
    )
