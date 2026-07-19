from typing import Any, Dict, List, Optional, Tuple

from ai_training.dataset import NON_FEATURE_SCENARIO_COLUMNS, TrainingDataset
from ai_training.models.base import BaseModel, ModelBundle, build_classifier
from ai_training.preprocessing import select_features

VALID_TARGETS = ("occurrence", "location")

# The classification label used for scenarios where Ground Truth
# recorded no congestion peak at all -- kept as its own explicit class
# ("no bottleneck" is a meaningful, predictable outcome) rather than
# dropping those rows.
NO_BOTTLENECK_LABEL = "NONE"


class BottleneckModel(BaseModel):

    # Classification over a scenario's pre-simulation Scenario
    # Features. Two sub-tasks, selected via target=:
    #   - "occurrence" (default): did ANY door become a bottleneck
    #     during the run? (Ground Truth's
    #     doors_that_became_bottlenecks, non-empty -> True)
    #   - "location": WHICH location peaked congestion first (Ground
    #     Truth's peak_congestion_location_id, or NO_BOTTLENECK_LABEL)
    # Both labels come from Ground Truth (an already-computed
    # simulation result) -- this model only predicts them from what
    # was configured before the simulation ran, never recomputes them.

    is_classifier = True

    def __init__(
        self, estimator: Any = None, config: Optional[Dict[str, Any]] = None, *, target: str = "occurrence",
    ):

        if target not in VALID_TARGETS:
            raise ValueError(f"target must be one of {VALID_TARGETS}, got {target!r}")

        config = dict(config or {})
        config["target"] = target

        super().__init__(estimator=estimator, config=config)

        self.target = target
        self.target_name = f"bottleneck_{target}"

    # =====================================================

    def default_estimator(self):

        algorithm = self.config.get("algorithm", "random_forest")

        return build_classifier(algorithm, self.config.get("algorithm_kwargs"))

    # =====================================================

    @staticmethod
    def build_table(
        dataset: TrainingDataset, *, target: str = "occurrence",
    ) -> Tuple[List[Dict[str, Any]], List[Any], Dict[str, Any]]:

        if target not in VALID_TARGETS:
            raise ValueError(f"target must be one of {VALID_TARGETS}, got {target!r}")

        feature_rows = select_features(
            dataset.scenario_feature_rows(), exclude=NON_FEATURE_SCENARIO_COLUMNS,
        )

        y: List[Any] = []

        for feature_row, ground_truth in zip(feature_rows, dataset.ground_truth_rows()):

            if ground_truth is None:
                raise ValueError(
                    "BottleneckModel requires ground_truth for every scenario -- "
                    "load the campaign with require_ground_truth=True (the default)."
                )

            if target == "occurrence":
                y.append(bool(ground_truth.get("doors_that_became_bottlenecks")))
            else:
                y.append(ground_truth.get("peak_congestion_location_id") or NO_BOTTLENECK_LABEL)

        return feature_rows, y, {}

    # =====================================================

    @classmethod
    def _construct_for_load(cls, bundle: ModelBundle) -> "BottleneckModel":

        target = bundle.config.get("target", "occurrence")

        return cls(estimator=bundle.estimator, config=bundle.config, target=target)
