"""Predictive Model Development & Benchmark Campaign milestone, Phase 9 --
register the selected model(s) via the EXISTING ai_registry.registry.
ModelRegistry. Never touches any runtime startup/wiring path (live_
runtime/factory.py, live_system/main.py, etc.) -- Shadow Mode's existing
wiring (already proven end-to-end in prior milestones) is completely
untouched; this script only proves that IF that wiring's ModelRegistry
were populated by this benchmark's selected model, it would be served
correctly, exactly like ai_registry.registry.get_latest_compatible_model()
already guarantees for any correctly-registered LIVE_COMPATIBLE model.

Selection rule (documented, not hidden): for each target, rank every
successfully-trained candidate by its PRIMARY metric on the held-out TEST
split, after first requiring it beat its own Dummy control on that same
metric (a model that cannot beat a trivial baseline is never eligible,
regardless of rank).

For evacuation_time (regression): MAE, ascending -- the direct, literal
"average prediction error in seconds" question a deployment cares about.

For bottleneck_occurrence (classification): ROC-AUC, NOT F1 -- discovered
during this benchmark campaign (see docs/architecture/model_benchmark.md's
own "Severe class imbalance" section): the test split is 95.1% positive
(doors_that_became_bottlenecks is true for the vast majority of scenarios
in this small fixed building), which makes F1 a misleading ranking metric
-- MLP achieved F1=0.975 by degenerating to "always predict positive"
(confusion matrix tn=0, ROC-AUC=0.477, i.e. AT CHANCE, actually
marginally below Dummy's own 0.483), which a raw-F1 ranking would have
scored ABOVE several genuinely-discriminating models. ROC-AUC is
threshold-independent and directly measures whether a model's predicted
probabilities actually separate the (rare) negative class from the
(common) positive class -- this correctly disqualifies MLP (fails to
beat Dummy) while correctly ranking the tree ensembles and Logistic
Regression, which all score ROC-AUC ~0.83.

Usage: python scripts/run_model_benchmark_registration.py
"""
import datetime
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import joblib

from ai_features.feature_schema import CANONICAL_LIVE_FEATURE_NAMES, SCHEMA_VERSION
from ai_registry.metadata import Deployability, ModelMetadata, save_live_model
from ai_registry.registry import ModelRegistry
from ai_registry.inference_service import LiveAIInferenceService
from live_system.live_ai_gateway import RegistryLiveAIInferenceGateway
import ai_features as af
import ai_registry as reg

SCRATCH = Path(r"C:\Users\riddh\AppData\Local\Temp\claude\C--Users-riddh-Desktop-SynEvac1\3cfe9653-d222-4b7b-ad2c-e86698320fce\scratchpad")
RESULTS_DIR = SCRATCH / "benchmark_results"
MODEL_ARTIFACT_DIR = SCRATCH / "benchmark_registered_models"
DATASET_IDENTIFIER = "model-benchmark-campaign-v1-11997-scenarios"
TRAINING_SEED = 20260730


def _eligible_candidates(results, *, metric_key, higher_is_better):

    successful = [r for r in results if "error" not in r]
    dummy = next((r for r in successful if r["algorithm"] == "dummy"), None)
    dummy_score = dummy["test_metrics"].get(metric_key) if dummy else None

    eligible = []
    for r in successful:

        if r["algorithm"] == "dummy":
            continue

        score = r["test_metrics"].get(metric_key)
        if score is None:
            continue

        if dummy_score is not None:
            beats_dummy = (score > dummy_score) if higher_is_better else (score < dummy_score)
            if not beats_dummy:
                continue

        eligible.append(r)

    return eligible


def _select_winner(results, *, metric_key, higher_is_better):

    eligible = _eligible_candidates(results, metric_key=metric_key, higher_is_better=higher_is_better)

    if not eligible:
        return None, "no candidate beat its own Dummy control on the primary metric"

    eligible.sort(key=lambda r: r["test_metrics"][metric_key], reverse=higher_is_better)

    return eligible[0], None


def _select_classification_winner(results, *, roc_auc_tolerance=0.01):

    # ROC-AUC is the primary metric (see module docstring -- F1 is
    # misleading under this dataset's 95.1% class imbalance). Among
    # candidates within `roc_auc_tolerance` of the single best ROC-AUC
    # (a statistically-tied cluster, not a genuine ranking -- this
    # benchmark's own top two candidates differed by ~0.00001, far
    # inside any reasonable noise band), the tiebreaker is calibration_
    # error, ascending: a live deployment (Shadow-Mode's own
    # bottleneck_threshold=0.5 probability cutoff) depends on predicted
    # probabilities actually meaning what they claim, not merely on
    # rank-ordering being correct -- ROC-AUC alone cannot see that,
    # calibration_error directly measures it.

    eligible = _eligible_candidates(results, metric_key="roc_auc", higher_is_better=True)

    if not eligible:
        return None, "no candidate beat its own Dummy control on ROC-AUC", []

    best_auc = max(r["test_metrics"]["roc_auc"] for r in eligible)
    tied_cluster = [r for r in eligible if best_auc - r["test_metrics"]["roc_auc"] <= roc_auc_tolerance]

    tied_cluster.sort(key=lambda r: r["test_metrics"]["calibration_error"] if r["test_metrics"]["calibration_error"] is not None else float("inf"))

    return tied_cluster[0], None, tied_cluster


def main():

    with open(RESULTS_DIR / "classification_results.json", encoding="utf-8") as f:
        classification_results = json.load(f)
    with open(RESULTS_DIR / "regression_results.json", encoding="utf-8") as f:
        regression_results = json.load(f)

    fitted = joblib.load(RESULTS_DIR / "fitted_models.joblib")

    winner_classification, reject_reason_c, tied_cluster = _select_classification_winner(classification_results)
    winner_regression, reject_reason_r = _select_winner(regression_results, metric_key="mae", higher_is_better=False)

    selection_report = {
        "classification": {
            "winner": winner_classification["algorithm"] if winner_classification else None,
            "reject_reason": reject_reason_c,
            "selection_metric": "roc_auc (not f1 -- see module docstring: 95.1% class imbalance makes F1 misleading), "
                                 "with calibration_error as tiebreaker among near-tied ROC-AUC candidates",
            "roc_auc_tied_cluster": [
                {"algorithm": r["algorithm"], "roc_auc": r["test_metrics"]["roc_auc"], "calibration_error": r["test_metrics"]["calibration_error"]}
                for r in tied_cluster
            ],
            "test_roc_auc": winner_classification["test_metrics"]["roc_auc"] if winner_classification else None,
            "test_f1": winner_classification["test_metrics"]["f1"] if winner_classification else None,
        },
        "regression": {
            "winner": winner_regression["algorithm"] if winner_regression else None,
            "reject_reason": reject_reason_r,
            "test_mae": winner_regression["test_metrics"]["mae"] if winner_regression else None,
        },
    }
    print(json.dumps(selection_report, indent=2), flush=True)

    registry = ModelRegistry()
    version_stamp = datetime.datetime.now(datetime.timezone.utc).strftime("v%Y%m%d%H%M%S")

    if winner_classification:

        model = fitted["classification"][winner_classification["algorithm"]]

        metadata = ModelMetadata(
            model_id=f"BottleneckOccurrenceModel_Benchmark-{version_stamp}",
            model_type="bottleneck_occurrence",
            model_version=version_stamp,
            training_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            training_dataset_identifier=DATASET_IDENTIFIER,
            training_seed=TRAINING_SEED,
            feature_schema_version=SCHEMA_VERSION,
            ordered_feature_names=CANONICAL_LIVE_FEATURE_NAMES,
            prediction_target="bottleneck_occurrence (bool(GroundTruth.doors_that_became_bottlenecks)) -- "
                               f"selected via Predictive Model Development & Benchmark Campaign, algorithm={winner_classification['algorithm']!r}",
            model_deployability=Deployability.LIVE_COMPATIBLE,
            training_metrics={"train_time_seconds": winner_classification["train_time_seconds"]},
            validation_metrics={"val": winner_classification.get("val_metrics"), "test": winner_classification["test_metrics"]},
            missing_data_policy="None (never a fabricated zero) for any field whose *_available/*_observed "
                                 "companion flag is False -- unchanged from the existing ai_registry.training convention.",
        )

        save_live_model(model, metadata, str(MODEL_ARTIFACT_DIR / "bottleneck_occurrence"))
        registry.register_model(model, metadata)

        print(f"Registered classification winner: {winner_classification['algorithm']} as {metadata.model_id}", flush=True)

    if winner_regression:

        model = fitted["regression"][winner_regression["algorithm"]]

        metadata = ModelMetadata(
            model_id=f"EvacuationTimeModel_Benchmark-{version_stamp}",
            model_type="evacuation_time",
            model_version=version_stamp,
            training_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            training_dataset_identifier=DATASET_IDENTIFIER,
            training_seed=TRAINING_SEED,
            feature_schema_version=SCHEMA_VERSION,
            ordered_feature_names=CANONICAL_LIVE_FEATURE_NAMES,
            prediction_target="total_evacuation_time (seconds, from simulation/incident start) -- "
                               f"selected via Predictive Model Development & Benchmark Campaign, algorithm={winner_regression['algorithm']!r}",
            model_deployability=Deployability.LIVE_COMPATIBLE,
            training_metrics={"train_time_seconds": winner_regression["train_time_seconds"]},
            validation_metrics={"val": winner_regression.get("val_metrics"), "test": winner_regression["test_metrics"]},
            missing_data_policy="None (never a fabricated zero) for any field whose *_available/*_observed "
                                 "companion flag is False -- unchanged from the existing ai_registry.training convention.",
        )

        save_live_model(model, metadata, str(MODEL_ARTIFACT_DIR / "evacuation_time"))
        registry.register_model(model, metadata)

        print(f"Registered regression winner: {winner_regression['algorithm']} as {metadata.model_id}", flush=True)

    # -- prove Shadow-Mode's existing gateway serves exactly these models, end to end --
    service = LiveAIInferenceService(registry)
    gateway = RegistryLiveAIInferenceGateway(service, include_evacuation_time=winner_regression is not None)

    building = reg.make_training_building()
    state = af.build_building_state_at_alarm_activation(building, total_occupants=10, timestamp=0.0)

    snapshot = gateway.predict(state, 0.0)

    proof = {
        "system_status": snapshot.system_status.name if snapshot.system_status else None,
        "bottleneck_probability": snapshot.bottleneck.probability if snapshot.bottleneck else None,
        "bottleneck_model_id": snapshot.bottleneck.model_id if snapshot.bottleneck else None,
        "evacuation_time_predicted_seconds": snapshot.evacuation_time_experimental.predicted_seconds if snapshot.evacuation_time_experimental else None,
        "errors": list(snapshot.errors), "warnings": list(snapshot.warnings),
    }
    print("SHADOW-MODE GATEWAY PROOF:", json.dumps(proof, indent=2), flush=True)

    selection_report["gateway_proof"] = proof

    with open(RESULTS_DIR / "registration_report.json", "w", encoding="utf-8") as f:
        json.dump(selection_report, f, indent=2, default=str)

    print("REGISTRATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
