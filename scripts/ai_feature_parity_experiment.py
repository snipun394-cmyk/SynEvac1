"""Simulation-to-Live AI Feature Parity milestone -- Phase 8 experiment.

Compares the existing (simulation-only-feature) EvacuationTimeModel
against an additive experimental model trained on nothing but the
canonical live-compatible feature schema (ai_features/feature_schema.py),
over the SAME real campaign's scenarios and the SAME simulation
Ground-Truth labels -- answering "how much predictive performance is
lost when we remove simulation-only information?"

Not a pytest test (same convention as scripts/benchmark_live_camera_
pipeline.py): this is a one-shot report, run manually
(`python scripts/ai_feature_parity_experiment.py`) and read. Trains
nothing into any production model store -- purely an in-memory
comparison, reported to stdout and copied into
docs/architecture/ai_live_feature_parity.md by hand.
"""

import shutil
import sys
import tempfile

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.training_dataset_fixtures import make_campaign, make_building

import ai_training as at
from ai_training.dataset import ScenarioRecord, TrainingDataset
from ai_training.experiment import ExperimentConfig, ExperimentRunner

import ai_features as af


CAMPAIGN_COUNT = 80
CAMPAIGN_SEED = 2026


def build_live_compatible_dataset(legacy_dataset: TrainingDataset) -> TrainingDataset:

    building = make_building()  # same fixed Building every campaign scenario ran against

    live_records = []

    for record in legacy_dataset.records:

        ignition_zone_id = record.features.get("ignition_zone")
        total_occupants = record.features.get("total_occupants") or 0

        live_row = af.extract_canonical_training_row(
            building, total_occupants=total_occupants, ignition_zone_id=ignition_zone_id,
        )

        live_records.append(ScenarioRecord(
            scenario_id=record.scenario_id,
            features=live_row,
            outcome=record.outcome,
            zone_results=record.zone_results,
            timeline=record.timeline,
            ground_truth=record.ground_truth,
            decision_policy=record.decision_policy,
        ))

    return TrainingDataset(
        campaign_dir=legacy_dataset.campaign_dir, records=live_records, errors=legacy_dataset.errors,
    )


def main():

    tmp_dir = tempfile.mkdtemp(prefix="ai_feature_parity_experiment_")

    try:

        print(f"Generating a real {CAMPAIGN_COUNT}-scenario campaign (seed={CAMPAIGN_SEED})...")
        make_campaign(tmp_dir, count=CAMPAIGN_COUNT, master_seed=CAMPAIGN_SEED)

        legacy_dataset = at.load_campaign_dataset(tmp_dir)
        live_dataset = build_live_compatible_dataset(legacy_dataset)

        print(f"Loaded {len(legacy_dataset)} scenarios.\n")

        runner = ExperimentRunner()

        legacy_result = runner.run(
            legacy_dataset,
            ExperimentConfig(
                name="existing-evacuation-time-v1", model_name="evacuation_time",
                feature_set="scenario_features", random_state=0,
            ),
        )

        live_result = runner.run(
            live_dataset,
            ExperimentConfig(
                name="live-compatible-evacuation-time-v1", model_name="evacuation_time",
                feature_set="ai_features_canonical_v1", random_state=0,
            ),
        )

        legacy_columns = sorted(legacy_result.model.feature_schema.columns)
        live_columns = sorted(live_result.model.feature_schema.columns)

        print("=" * 70)
        print("EXISTING MODEL (simulation-only scenario features)")
        print("=" * 70)
        print(f"feature_set:   scenario_features ({len(legacy_columns)} columns)")
        print(f"train/test:    {legacy_result.train_size}/{legacy_result.test_size_actual}")
        print(f"metrics:       {legacy_result.metrics}")

        print()
        print("=" * 70)
        print("LIVE-COMPATIBLE MODEL (ai_features canonical schema only)")
        print("=" * 70)
        print(f"feature_set:   ai_features_canonical_v1 ({len(live_columns)} columns)")
        print(f"columns:       {live_columns}")
        print(f"train/test:    {live_result.train_size}/{live_result.test_size_actual}")
        print(f"metrics:       {live_result.metrics}")

        print()
        print("=" * 70)
        print("COMPARISON")
        print("=" * 70)

        for metric in ("mae", "rmse", "r2"):

            legacy_value = legacy_result.metrics[metric]
            live_value = live_result.metrics[metric]
            delta = live_value - legacy_value

            print(f"{metric:>5}: existing={legacy_value:.3f}  live-compatible={live_value:.3f}  delta={delta:+.3f}")

        print()
        print("#" * 70)
        print("SECOND MODEL: bottleneck_occurrence (classification)")
        print("#" * 70)

        legacy_bottleneck = runner.run(
            legacy_dataset,
            ExperimentConfig(
                name="existing-bottleneck-occurrence-v1", model_name="bottleneck",
                feature_set="scenario_features", model_kwargs={"target": "occurrence"}, random_state=0,
            ),
        )
        live_bottleneck = runner.run(
            live_dataset,
            ExperimentConfig(
                name="live-compatible-bottleneck-occurrence-v1", model_name="bottleneck",
                feature_set="ai_features_canonical_v1", model_kwargs={"target": "occurrence"}, random_state=0,
            ),
        )

        print(f"existing        train/test: {legacy_bottleneck.train_size}/{legacy_bottleneck.test_size_actual}  "
              f"metrics: {legacy_bottleneck.metrics}")
        print(f"live-compatible train/test: {live_bottleneck.train_size}/{live_bottleneck.test_size_actual}  "
              f"metrics: {live_bottleneck.metrics}")

    finally:

        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
