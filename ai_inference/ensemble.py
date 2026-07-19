from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from ai_inference.loader import LoadedModel
from ai_inference.predictor import prediction_type_key


# =====================================================
# Phase 3 -- Ensemble Layer. A pluggable set of already-loaded models
# for the SAME prediction_type (e.g. three independently trained
# evacuation_time models: Random Forest/Gradient Boosting/XGBoost) --
# combines their individual, already-fitted model.predict() outputs.
# Never trains, never touches ai_training's MODEL_REGISTRY directly:
# any model_name ai_training/ai_inference's loader already knows how
# to load can be an ensemble member without any change here.
# =====================================================


@dataclass(frozen=True)
class EnsembleMember:

    loaded_model: LoadedModel
    weight: float = 1.0


# =====================================================


class Ensemble:

    def __init__(self, members: Sequence[EnsembleMember]):

        if not members:
            raise ValueError("Ensemble requires at least one member")

        prediction_types = {prediction_type_key(member.loaded_model) for member in members}

        if len(prediction_types) > 1:

            raise ValueError(
                f"All ensemble members must share the same prediction type, got {sorted(prediction_types)}"
            )

        self.members = list(members)
        self.prediction_type = next(iter(prediction_types))

    # =====================================================

    @property
    def is_classifier(self) -> bool:

        return self.members[0].loaded_model.model.is_classifier

    # =====================================================

    def predict_all_members(self, X_row: Dict[str, Any]) -> Dict[str, Any]:

        # experiment_name (Phase 4's own "Experiment ID") -> that
        # member's raw predicted value -- lets a caller see every
        # member's individual opinion, not just the combined one.

        return {
            member.loaded_model.provenance.experiment_name: member.loaded_model.model.predict([X_row])[0]
            for member in self.members
        }

    # =====================================================

    def best_model(self, metric_key: str, *, minimize: bool) -> EnsembleMember:

        # "Best model" strategy -- picks the single member with the
        # best already-recorded evaluation metric from its own
        # ai_training experiment run. Never re-evaluates against new
        # data (ai_inference has no simulation/dataset access to do so
        # even if it wanted to).

        def _value(member: EnsembleMember) -> float:
            return _extract_metric(member.loaded_model.provenance.metrics, metric_key)

        return min(self.members, key=_value) if minimize else max(self.members, key=_value)

    # =====================================================

    def predict_best(self, X_row: Dict[str, Any], metric_key: str, *, minimize: bool) -> Any:

        member = self.best_model(metric_key, minimize=minimize)

        return member.loaded_model.model.predict([X_row])[0]

    # =====================================================

    def predict_average(self, X_row: Dict[str, Any]) -> Any:

        if self.is_classifier:
            raise TypeError("predict_average() is for regression ensembles -- use predict_majority_vote()")

        values = [member.loaded_model.model.predict([X_row])[0] for member in self.members]

        return _combine_numeric(values, weights=None)

    # =====================================================

    def predict_weighted_average(self, X_row: Dict[str, Any]) -> Any:

        if self.is_classifier:
            raise TypeError("predict_weighted_average() is for regression ensembles -- use predict_majority_vote()")

        values = [member.loaded_model.model.predict([X_row])[0] for member in self.members]
        weights = [member.weight for member in self.members]

        return _combine_numeric(values, weights=weights)

    # =====================================================

    def predict_majority_vote(self, X_row: Dict[str, Any]) -> Any:

        if not self.is_classifier:
            raise TypeError("predict_majority_vote() is for classification ensembles -- use predict_average()")

        predictions = [member.loaded_model.model.predict([X_row])[0] for member in self.members]
        weights = [member.weight for member in self.members]

        tally: Dict[Any, float] = {}

        for prediction, weight in zip(predictions, weights):
            tally[prediction] = tally.get(prediction, 0.0) + weight

        # Deterministic tie-break: highest total weight, then
        # ascending by the label's own string form -- never left to
        # whatever order a dict happened to iterate in.
        ordered = sorted(tally.items(), key=lambda item: (-item[1], str(item[0])))

        return ordered[0][0]


# =====================================================


def _extract_metric(metrics: Dict[str, Any], metric_key: str) -> float:

    if metric_key in metrics:
        return metrics[metric_key]

    if "macro_average" in metrics and metric_key in metrics["macro_average"]:
        return metrics["macro_average"][metric_key]

    raise KeyError(f"metric {metric_key!r} not found in recorded evaluation metrics {list(metrics.keys())}")


# =====================================================


def _combine_numeric(values: List[Any], *, weights: List[float] = None) -> Any:

    weights = weights if weights is not None else [1.0] * len(values)
    total_weight = sum(weights)

    if isinstance(values[0], dict):

        keys = values[0].keys()

        return {
            key: sum(value[key] * weight for value, weight in zip(values, weights)) / total_weight
            for key in keys
        }

    return float(sum(value * weight for value, weight in zip(values, weights)) / total_weight)
