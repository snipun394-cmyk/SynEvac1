"""Live-Compatible AI Model Training & Model Registry milestone.

Generates a large, diverse synthetic campaign, trains EvacuationTimeModel_
LiveCompatible and BottleneckOccurrenceModel_LiveCompatible on the
canonical live-compatible feature contract (ai_features/), evaluates
each against trivial baselines, registers both in a ModelRegistry, and
proves the registry + LiveAIInferenceService work end-to-end -- all
still NOT wired into LiveOrchestrator or advisory_system.

Not a pytest test (same convention as scripts/ai_feature_parity_
experiment.py / scripts/benchmark_live_camera_pipeline.py): a one-shot
report, run manually and read.

    python scripts/train_live_compatible_models.py [--count 5000] [--seed 2026]
"""

import argparse
import json
import shutil
import sys
import tempfile
import time

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtWidgets import QApplication
_app = QApplication.instance() or QApplication(sys.argv)

import ai_training as at

import ai_registry as reg

from ai_features.building_state_extractor import extract_canonical_features
from ai_features.simulation_extractor import build_building_state_at_alarm_activation


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    building = reg.make_training_building()
    definition = reg.make_training_definition()

    campaign_tmp = tempfile.mkdtemp(prefix="live_ai_campaign_")

    try:

        print(f"Generating {args.count} scenarios (seed={args.seed}) against the diverse training building...")

        start = time.perf_counter()
        campaign_dir, summary = reg.generate_training_campaign(
            campaign_tmp, building, definition, count=args.count, master_seed=args.seed,
        )
        generation_seconds = time.perf_counter() - start

        print(
            f"  requested={summary.total_requested} generated={summary.total_generated} "
            f"accepted={summary.accepted} rejected={summary.rejected} "
            f"({generation_seconds:.1f}s, {generation_seconds / max(summary.total_requested, 1) * 1000:.1f}ms/scenario)",
        )

        legacy_dataset = at.load_campaign_dataset(campaign_dir)
        live_dataset = reg.build_live_compatible_dataset(legacy_dataset, building)

        dataset_identifier = f"live-ai-campaign-seed{args.seed}-n{len(live_dataset)}"

        print(f"\nLoaded {len(live_dataset)} scenarios into the live-compatible dataset.\n")

        print("=" * 72)
        print("TRAINING: EvacuationTimeModel_LiveCompatible")
        print("=" * 72)

        evac_result = reg.train_evacuation_time_model(
            live_dataset, training_seed=args.seed, dataset_identifier=dataset_identifier,
        )

        _print_evacuation_report(evac_result)

        print()
        print("=" * 72)
        print("TRAINING: BottleneckOccurrenceModel_LiveCompatible")
        print("=" * 72)

        bottleneck_result = reg.train_bottleneck_occurrence_model(
            live_dataset, training_seed=args.seed, dataset_identifier=dataset_identifier,
        )

        _print_bottleneck_report(bottleneck_result)

        print()
        print("=" * 72)
        print("MODEL REGISTRY + SAFE INFERENCE SERVICE")
        print("=" * 72)

        registry = reg.ModelRegistry()
        registry.register_model(evac_result.model, evac_result.metadata)
        registry.register_model(bottleneck_result.model, bottleneck_result.metadata)

        service = reg.LiveAIInferenceService(registry)

        _benchmark_and_report(service, building)

        if args.save_dir:

            evac_dir = str(Path(args.save_dir) / evac_result.metadata.model_id)
            bottleneck_dir = str(Path(args.save_dir) / bottleneck_result.metadata.model_id)

            reg.save_live_model(evac_result.model, evac_result.metadata, evac_dir)
            reg.save_live_model(bottleneck_result.model, bottleneck_result.metadata, bottleneck_dir)

            print(f"\nSaved model artifacts to {args.save_dir}")

        summary_path = Path(tempfile.gettempdir()) / "live_ai_training_summary.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump({
                "campaign": {
                    "requested": summary.total_requested, "accepted": summary.accepted,
                    "generation_seconds": generation_seconds,
                },
                "evacuation_time": {
                    "metadata": evac_result.metadata.to_dict(),
                    "baseline_metrics": evac_result.baseline_metrics,
                    "scenario_split_counts": vars(evac_result.scenario_split_counts),
                    "extra_evaluation": _json_safe(evac_result.extra_evaluation),
                },
                "bottleneck_occurrence": {
                    "metadata": bottleneck_result.metadata.to_dict(),
                    "baseline_metrics": bottleneck_result.baseline_metrics,
                    "scenario_split_counts": vars(bottleneck_result.scenario_split_counts),
                    "extra_evaluation": _json_safe(bottleneck_result.extra_evaluation),
                },
            }, handle, indent=2, default=str)

        print(f"\nFull JSON summary written to {summary_path}")

    finally:

        shutil.rmtree(campaign_tmp, ignore_errors=True)


def _json_safe(obj):

    import numpy as np

    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _print_evacuation_report(result):

    counts = result.scenario_split_counts
    print(f"scenarios: train={counts.train} validation={counts.validation} test={counts.test}")
    print(f"test metrics:     {result.extra_evaluation['test_metrics']}")
    print(f"baseline (mean):  {result.baseline_metrics['mean']}")
    print(f"baseline (median):{result.baseline_metrics['median']}")
    print(f"model status:     {result.extra_evaluation['model_status']}")
    print(f"uncertainty:      available={result.extra_evaluation['uncertainty_available']}")
    print(f"error by occupancy range: {result.extra_evaluation['error_breakdown_by_occupancy']['by_occupancy_range']}")
    print(f"worst-case errors (top 5): {result.extra_evaluation['error_breakdown_by_occupancy']['worst_case_errors']}")


def _print_bottleneck_report(result):

    counts = result.scenario_split_counts
    print(f"scenarios: train={counts.train} validation={counts.validation} test={counts.test}")
    print(f"test metrics:      {result.extra_evaluation['test_metrics']}")
    print(f"baseline (majority):{result.baseline_metrics['most_frequent']}")
    print(f"baseline (freq):   {result.baseline_metrics['stratified']}")
    print(f"model status:      {result.extra_evaluation['model_status']}")
    print(f"train class balance: {result.extra_evaluation['train_class_balance']}")
    print(f"test class balance:  {result.extra_evaluation['test_class_balance']}")
    print(f"confusion matrix:    {result.extra_evaluation['confusion_matrix']}")
    print(f"probability calibration (Brier score): {result.extra_evaluation['probability_calibration']['brier_score']}")


def _benchmark_and_report(service, building):

    state = build_building_state_at_alarm_activation(building, total_occupants=8, ignition_zone_id="zone-lobby")

    start = time.perf_counter()
    for _ in range(20):
        extract_canonical_features(state)
    extraction_ms = (time.perf_counter() - start) / 20 * 1000

    start = time.perf_counter()
    for _ in range(20):
        service._registry.get_latest_compatible_model("evacuation_time")
    lookup_ms = (time.perf_counter() - start) / 20 * 1000

    start = time.perf_counter()
    for _ in range(20):
        service.predict_evacuation_time(state, timestamp=0.0)
    evac_inference_ms = (time.perf_counter() - start) / 20 * 1000

    start = time.perf_counter()
    for _ in range(20):
        service.predict_bottleneck_occurrence(state, timestamp=0.0)
    bottleneck_inference_ms = (time.perf_counter() - start) / 20 * 1000

    start = time.perf_counter()
    for _ in range(20):
        extract_canonical_features(state)
        service._registry.get_latest_compatible_model("evacuation_time")
        service.predict_evacuation_time(state, timestamp=0.0)
        service.predict_bottleneck_occurrence(state, timestamp=0.0)
    combined_ms = (time.perf_counter() - start) / 20 * 1000

    print(f"BuildingState feature extraction: {extraction_ms:.3f} ms")
    print(f"model registry lookup (cached):   {lookup_ms:.3f} ms")
    print(f"evacuation time inference:        {evac_inference_ms:.3f} ms")
    print(f"bottleneck occurrence inference:  {bottleneck_inference_ms:.3f} ms")
    print(f"combined (extract+lookup+both):   {combined_ms:.3f} ms")

    evac_pred = service.predict_evacuation_time(state, timestamp=0.0)
    bottleneck_pred = service.predict_bottleneck_occurrence(state, timestamp=0.0)
    print(f"\nsample evacuation prediction: {evac_pred}")
    print(f"sample bottleneck prediction: {bottleneck_pred}")


if __name__ == "__main__":
    main()
