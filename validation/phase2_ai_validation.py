"""
Phase 2 -- AI Validation.

Trains EvacuationTimeModel (regression: MAE/RMSE/R2) and BottleneckModel
(classification: accuracy/precision/recall/f1/confusion matrix), each
with three interchangeable estimators (random_forest, gradient_boosting,
xgboost), against a real campaign dataset produced by the existing
Campaign Studio pipeline. Does not modify ai_training/ or any other
package -- only calls its existing public API.
"""

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import confusion_matrix

from scenario_definition import FireDefinition, OccupantDefinition, ScenarioDefinition, UniformRange, WeightedOptions

from tests.training_dataset_fixtures import make_building, make_campaign

from ai_training import (
    BottleneckModel,
    EvacuationTimeModel,
    apply_split,
    evaluate_classification,
    evaluate_regression,
    load_campaign_dataset,
    make_split,
)
from ai_training.dataset import NON_FEATURE_SCENARIO_COLUMNS
from ai_training.preprocessing import select_features

ALGORITHMS = ("random_forest", "gradient_boosting", "xgboost")
CAMPAIGN_COUNT = 800
MASTER_SEED = 900002


def make_high_variance_definition() -> ScenarioDefinition:

    # A deliberately higher-variance definition than tests/
    # training_dataset_fixtures.py's own make_definition() (which pins
    # fire growth to a single FixedValue and a narrow 1-2 occupant
    # range) -- that fixture's whole point is fast, deterministic unit
    # tests, not realistic model-training variance. Widening fire
    # growth, occupancy, and door state here (still using only the
    # existing FixedValue/UniformRange/WeightedOptions distribution
    # kinds -- no new distribution type) gives EvacuationTimeModel/
    # BottleneckModel a genuinely learnable-but-not-trivial target
    # distribution, avoiding a misleadingly perfect R2/accuracy purely
    # from an almost-constant target.

    return ScenarioDefinition(
        fire=FireDefinition(growth_parameter_distribution=UniformRange(50.0, 400.0)),
        occupant=OccupantDefinition(
            occupancy_distribution={
                "zone-1": UniformRange(1, 8, discrete=True),
                "zone-2": UniformRange(1, 8, discrete=True),
            },
            behaviour_profile_distribution={
                "zone-1": WeightedOptions({"Staff_Default": 0.5, "Adult_Default": 0.5}),
                "zone-2": WeightedOptions({"Staff_Default": 0.5, "Adult_Default": 0.5}),
            },
        ),
    )


def _train_and_evaluate_regression(dataset, split, algorithm):

    train_rows, _val_rows, test_rows = apply_split(dataset.records, split)

    X_train, y_train, _ = EvacuationTimeModel.build_table(
        _dataset_view(dataset, train_rows)
    )
    X_test, y_test, _ = EvacuationTimeModel.build_table(
        _dataset_view(dataset, test_rows)
    )

    model = EvacuationTimeModel(config={"algorithm": algorithm})

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    metrics = evaluate_regression(model, X_test, y_test)
    metrics["train_time_seconds"] = train_time
    metrics["train_rows"] = len(X_train)
    metrics["test_rows"] = len(X_test)

    return metrics


def _train_and_evaluate_bottleneck(dataset, split, algorithm):

    train_rows, _val_rows, test_rows = apply_split(dataset.records, split)

    X_train, y_train, _ = BottleneckModel.build_table(_dataset_view(dataset, train_rows), target="occurrence")
    X_test, y_test, _ = BottleneckModel.build_table(_dataset_view(dataset, test_rows), target="occurrence")

    model = BottleneckModel(config={"algorithm": algorithm}, target="occurrence")

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    metrics = evaluate_classification(model, X_test, y_test)
    metrics["train_time_seconds"] = train_time
    metrics["train_rows"] = len(X_train)
    metrics["test_rows"] = len(X_test)

    y_pred = model.predict(X_test)
    labels = sorted(set(list(y_test) + list(y_pred)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    metrics["confusion_matrix"] = {"labels": [str(l) for l in labels], "matrix": cm.tolist()}

    return metrics


class _DatasetView:

    # A thin, read-only slice of a TrainingDataset over a subset of its
    # own records -- reuses TrainingDataset's own accessor logic rather
    # than reimplementing row extraction, by simply constructing a
    # fresh TrainingDataset instance over the subset (TrainingDataset's
    # own __init__ signature accepts exactly (campaign_dir, records)).

    pass


def _dataset_view(dataset, records):

    from ai_training.dataset import TrainingDataset

    return TrainingDataset(dataset.campaign_dir, records)


def main():

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as campaign_dir:

        print(f"Generating campaign of {CAMPAIGN_COUNT} scenarios ...", flush=True)
        summary = make_campaign(
            campaign_dir, count=CAMPAIGN_COUNT, master_seed=MASTER_SEED,
            definition=make_high_variance_definition(),
        )
        print(
            f"Campaign done: accepted={summary.accepted} rejected={summary.rejected} "
            f"avg_sim_duration={summary.average_simulation_duration:.4f}s",
            flush=True,
        )

        dataset = load_campaign_dataset(campaign_dir)
        split = make_split(len(dataset.records), test_size=0.2, random_state=0)

        results = {
            "campaign_summary": {
                "requested": summary.total_requested,
                "generated": summary.total_generated,
                "accepted": summary.accepted,
                "rejected": summary.rejected,
                "average_evacuation_time": summary.average_evacuation_time,
                "average_simulation_duration": summary.average_simulation_duration,
            },
            "evacuation_time_regression": {},
            "bottleneck_occurrence_classification": {},
        }

        for algorithm in ALGORITHMS:
            print(f"Training EvacuationTimeModel [{algorithm}] ...", flush=True)
            results["evacuation_time_regression"][algorithm] = _train_and_evaluate_regression(
                dataset, split, algorithm,
            )

        for algorithm in ALGORITHMS:
            print(f"Training BottleneckModel [{algorithm}] ...", flush=True)
            results["bottleneck_occurrence_classification"][algorithm] = _train_and_evaluate_bottleneck(
                dataset, split, algorithm,
            )

    with open(os.path.join(output_dir, "phase2_ai_validation.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
