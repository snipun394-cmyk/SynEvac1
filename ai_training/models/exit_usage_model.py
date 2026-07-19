from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ai_training.dataset import NON_FEATURE_SCENARIO_COLUMNS, TrainingDataset
from ai_training.models.base import BaseModel, build_regressor
from ai_training.preprocessing import select_features


class ExitUsageModel(BaseModel):

    # Multi-output regression: predicts, for every exit the training
    # campaign's Building has, the percentage of evacuated occupants
    # who used it. Exit identity is data-dependent (a Building
    # property, not fixed at import time), so the exit-id vocabulary
    # (self.output_names) is discovered from the TRAINING data at fit()
    # time and persisted alongside the model -- predict() always
    # returns one dict per scenario keyed by those exact exit ids.

    target_name = "exit_usage_percentage"
    is_classifier = False

    def default_estimator(self):

        algorithm = self.config.get("algorithm", "random_forest")
        base_estimator = build_regressor(algorithm, self.config.get("algorithm_kwargs"))

        if algorithm == "random_forest":
            return base_estimator  # RandomForestRegressor natively supports multi-output targets

        from sklearn.multioutput import MultiOutputRegressor

        return MultiOutputRegressor(base_estimator)

    # =====================================================

    def fit(self, X_rows: Sequence[Dict[str, Any]], y: Sequence[Sequence[float]], *, output_names: Sequence[str]):

        self.output_names = list(output_names)

        return super().fit(X_rows, y)

    # =====================================================

    def predict(self, X_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, float]]:

        if self.output_names is None:
            raise RuntimeError("ExitUsageModel.fit() must be called before predict()")

        raw = np.asarray(super().predict(X_rows))

        # A single-exit Building makes this a single-output regression
        # (n_outputs=1) -- RandomForestRegressor.predict() then returns
        # a flat 1-D array rather than an (n, 1) matrix, same as
        # scikit-learn does for any single-column multi-output target.
        # Restored to 2-D here so the zip() below always sees one row
        # per scenario regardless of how many exits the Building has.
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)

        return [dict(zip(self.output_names, row)) for row in raw]

    # =====================================================

    @staticmethod
    def build_table(
        dataset: TrainingDataset,
    ) -> Tuple[List[Dict[str, Any]], List[List[float]], Dict[str, Any]]:

        feature_rows = select_features(
            dataset.scenario_feature_rows(), exclude=NON_FEATURE_SCENARIO_COLUMNS,
        )

        per_scenario_counts: List[Dict[str, float]] = []
        exit_ids = set()

        for record in dataset.records:

            counts: Dict[str, float] = {}

            for row in record.zone_results:

                exit_id = row.get("exit_used")
                evacuated = row.get("evacuated") or 0

                if exit_id:
                    counts[exit_id] = counts.get(exit_id, 0.0) + float(evacuated)
                    exit_ids.add(exit_id)

            per_scenario_counts.append(counts)

        sorted_exit_ids = tuple(sorted(exit_ids))

        y: List[List[float]] = []

        for counts in per_scenario_counts:

            total = sum(counts.values())

            if total <= 0 or not sorted_exit_ids:
                y.append([0.0] * len(sorted_exit_ids))
            else:
                y.append(
                    [100.0 * counts.get(exit_id, 0.0) / total for exit_id in sorted_exit_ids]
                )

        return feature_rows, y, {"output_names": sorted_exit_ids}
