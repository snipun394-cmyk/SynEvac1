import re
from typing import Any, Dict, List, Tuple

from ai_training.dataset import TrainingDataset
from ai_training.models.base import BaseModel, build_classifier
from ai_training.preprocessing import select_features

_SMOKE_COLUMN_PATTERN = re.compile(r"^Zone_(\d+)_Smoke$")

# Only scenario_id (a row identifier) is excluded -- unlike Scenario
# Features, simulation_time and every other timeline column are
# legitimate predictive features here.
_TIMELINE_ID_COLUMNS = ("scenario_id",)


class SmokePredictionModel(BaseModel):

    # Classification: given one simulated timestep's full state
    # (Timeline row t -- Scenario Features replicated per-tick, plus
    # that tick's dynamic hazard/occupancy/congestion readings),
    # predict which zone will read the highest smoke level at the NEXT
    # timestep (t+1). Only ever trained on (t -> t+1) pairs *within*
    # the same scenario -- a training row never uses information from
    # a time later than its own label's timestep.

    is_classifier = True
    target_name = "next_highest_smoke_zone"

    def default_estimator(self):

        algorithm = self.config.get("algorithm", "random_forest")

        return build_classifier(algorithm, self.config.get("algorithm_kwargs"))

    # =====================================================

    @staticmethod
    def build_table(
        dataset: TrainingDataset,
    ) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:

        # Returns (X_rows, y, {"groups": ...}) -- groups is the owning
        # scenario_id for every row, meant to be passed straight to
        # split.make_split(groups=...) so a scenario's consecutive
        # timesteps never straddle train/test.

        X_rows: List[Dict[str, Any]] = []
        y: List[str] = []
        groups: List[str] = []

        for record in dataset.records:

            timeline = sorted(record.timeline, key=lambda row: row.get("simulation_time") or 0.0)

            if len(timeline) < 2:
                continue

            smoke_columns = _smoke_columns(timeline[0])

            if not smoke_columns:
                continue

            for current_row, next_row in zip(timeline, timeline[1:]):

                feature_row = select_features([current_row], exclude=_TIMELINE_ID_COLUMNS)[0]

                X_rows.append(feature_row)
                y.append(_highest_smoke_zone(next_row, smoke_columns))
                groups.append(record.scenario_id)

        return X_rows, y, {"groups": groups}


# =====================================================


def _smoke_columns(row: Dict[str, Any]) -> List[str]:

    return sorted(
        (column for column in row.keys() if _SMOKE_COLUMN_PATTERN.match(column)),
        key=lambda column: int(_SMOKE_COLUMN_PATTERN.match(column).group(1)),
    )


def _highest_smoke_zone(row: Dict[str, Any], smoke_columns: List[str]) -> str:

    # max() returns the FIRST item achieving the maximum key value when
    # iterating left-to-right -- since smoke_columns is sorted
    # ascending by zone number, ties resolve to the lowest-numbered
    # zone, deterministically.
    best_column = max(smoke_columns, key=lambda column: row.get(column) or 0.0)

    zone_number = _SMOKE_COLUMN_PATTERN.match(best_column).group(1)

    return f"Zone_{zone_number}"
