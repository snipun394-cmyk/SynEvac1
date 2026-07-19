from typing import Any, Dict, List, Tuple

from ai_training.dataset import NON_FEATURE_SCENARIO_COLUMNS, TrainingDataset
from ai_training.models.base import BaseModel, build_regressor
from ai_training.preprocessing import select_features

TARGET_COLUMN = "total_evacuation_time"


class EvacuationTimeModel(BaseModel):

    # Regression: predicts total_evacuation_time from a scenario's
    # pre-simulation configuration (Scenario Features) alone -- every
    # input is known before a simulation ever runs, so there is no risk
    # of leaking simulation outputs back in as predictors.

    target_name = TARGET_COLUMN
    is_classifier = False

    def default_estimator(self):

        algorithm = self.config.get("algorithm", "random_forest")

        return build_regressor(algorithm, self.config.get("algorithm_kwargs"))

    # =====================================================

    @staticmethod
    def build_table(
        dataset: TrainingDataset,
    ) -> Tuple[List[Dict[str, Any]], List[float], Dict[str, Any]]:

        X_rows = select_features(
            dataset.scenario_feature_rows(), exclude=NON_FEATURE_SCENARIO_COLUMNS,
        )
        y = [outcome[TARGET_COLUMN] for outcome in dataset.outcome_rows()]

        return X_rows, y, {}
