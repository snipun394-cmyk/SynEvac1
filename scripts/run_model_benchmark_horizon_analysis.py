"""Predictive Model Development & Benchmark Campaign milestone, Phase 5's
horizon axis. Uses the ACTUAL prediction_evaluation framework
(PredictionRegistry / GroundTruthTimeline / evaluator.evaluate() /
horizon_analysis) end to end -- reused, not reimplemented.

Important, disclosed methodological note (see docs/architecture/
model_benchmark.md's own "Horizon robustness -- a disclosed limitation"
section): bottleneck_occurrence is a WHOLE-SCENARIO prediction computed
once from pre-simulation features (ai_features.CANONICAL_LIVE_SCHEMA),
never a function of elapsed time. It has no genuine "predict N seconds
from now" behavior the way predictive_dataset's own Target V2 does. This
script tags the SAME per-scenario prediction with each of the 5 nominal
horizons Phase 5 asks for (5/10/20/30/60s) purely to exercise the
horizon_analysis machinery honestly -- the EXPECTED, CORRECT result is
flat accuracy across every horizon bucket, because the model's inputs
never change with the nominal horizon tag. This is a structural property
of the model class being benchmarked, not evidence of horizon-robustness
in the sense Pipeline A's own per-candidate-per-tick models could claim.

Usage: python scripts/run_model_benchmark_horizon_analysis.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import joblib

from prediction_evaluation.registry import PredictionRegistry
from prediction_evaluation.models import GroundTruthSample
from prediction_evaluation.timeline import GroundTruthTimeline
from prediction_evaluation.evaluator import evaluate

SCRATCH = Path(r"C:\Users\riddh\AppData\Local\Temp\claude\C--Users-riddh-Desktop-SynEvac1\3cfe9653-d222-4b7b-ad2c-e86698320fce\scratchpad")
CACHE_PATH = SCRATCH / "benchmark_dataset_cache.joblib"
RESULTS_DIR = SCRATCH / "benchmark_results"
OUTPUT_PATH = RESULTS_DIR / "horizon_analysis.json"

HORIZONS = (5.0, 10.0, 20.0, 30.0, 60.0)
SAMPLE_SIZE = 60  # scenarios sampled per model -- matches the scale of the
                   # existing prediction_evaluation E2E test convention


class _Bottleneck:
    def __init__(self, probability, predicted_occurrence):
        self.probability = probability
        self.predicted_occurrence = predicted_occurrence


class _Snapshot:
    def __init__(self, timestamp, bottleneck):
        self.timestamp = timestamp
        self.bottleneck = bottleneck
        self.feature_schema_version = "1.0"


def _positive_proba_single(model, row):

    try:
        proba = model.predict_proba([row])
        classes = list(model.label_encoder.classes_)
        positive_index = classes.index(True) if True in classes else 1
        return float(proba[0, positive_index])
    except (AttributeError, TypeError):
        return None


def main():

    cache = joblib.load(CACHE_PATH)
    fitted = joblib.load(RESULTS_DIR / "fitted_models.joblib")

    live_rows = cache["live_feature_rows"]
    ground_truth_rows = cache["ground_truth_rows"]

    sample_indices = list(range(min(SAMPLE_SIZE, len(live_rows))))

    all_results = {}

    for algorithm, model in fitted["classification"].items():

        registry = PredictionRegistry()
        gt_samples = []

        for scenario_index in sample_indices:

            row = live_rows[scenario_index]
            gt = ground_truth_rows[scenario_index] or {}

            probability = _positive_proba_single(model, row)
            predicted = bool(model.predict([row])[0])

            payload = _Snapshot(timestamp=0.0, bottleneck=_Bottleneck(probability, predicted))
            congestion_detected = bool(gt.get("doors_that_became_bottlenecks"))

            for horizon in HORIZONS:

                registry.record(
                    timestamp=0.0, prediction_horizon_seconds=horizon, payload=payload,
                    source="benchmark", scenario_id=f"scenario-{scenario_index}",
                )
                gt_samples.append(GroundTruthSample(
                    timestamp=horizon, source="benchmark", congestion_detected=congestion_detected,
                ))

        timeline = GroundTruthTimeline(gt_samples)
        report = evaluate(registry.all(), timeline, tolerance_seconds=0.5, horizon_buckets_seconds=HORIZONS)

        all_results[algorithm] = {
            "matched": len(report.matched_evaluations),
            "unmatched": len(report.unmatched_predictions),
            "by_horizon": {
                str(h): {
                    "evaluation_count": bucket.evaluation_count,
                    "f1": bucket.classification.f1,
                    "precision": bucket.classification.precision,
                    "recall": bucket.classification.recall,
                    "roc_auc": bucket.classification.roc_auc,
                }
                for h, bucket in report.by_horizon.items()
            },
        }

        print(f"[{algorithm}] " + json.dumps(all_results[algorithm]["by_horizon"], indent=2), flush=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
