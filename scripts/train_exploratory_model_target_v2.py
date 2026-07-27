"""Predictive Congestion Target V2 milestone, Phase 12/18 -- SMALL
EXPLORATORY MODEL VALIDATION ONLY. This is NOT Localized Predictive
Model V3. Uses the existing V2.2 12-field feature schema, unchanged,
against the newly-relabeled Target V2 (data/predictive_congestion_
target_v2/candidate_dataset_relabeled.csv). Trains XGBoost and
HistGradientBoosting only (plus trivial baselines) -- no broad
algorithm tournament, no hyperparameter search, no calibration/
ablation/topology-holdout campaign (that is Localized Predictive Model
V3's job, a future, separate milestone).

The only question this script answers: IS THE NEW, PHYSICALLY
MEANINGFUL TARGET PREDICTABLE AT ALL, better than chance, using
information already available at prediction time?

Usage: python scripts/train_exploratory_model_target_v2.py
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predictive_model.baselines import build_baselines
from predictive_model.feature_prep_v2_1 import build_experimental_feature_matrix
from predictive_model.imbalance import compute_class_weight_map, sample_weights_from_class_weight, tune_threshold
from predictive_model.metrics import compute_metrics, metrics_by_group
from predictive_model.scenario_split import apply_split, assert_no_scenario_overlap, split_scenarios
from predictive_model.tree_models import build_tree_models

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "predictive_congestion_target_v2"
CSV_PATH = DATA_DIR / "candidate_dataset_relabeled.csv"
SCENARIO_METADATA_PATH = DATA_DIR / "scenario_metadata.json"
OUTPUT_DIR = DATA_DIR

SEED = 20260726
HORIZON = 20.0
N_JOBS = 2
RANDOM_FOREST_MAX_DEPTH = 20  # unused here (no RandomForest), kept for build_tree_models() signature parity

BASE_FEATURE_NAMES = (
    "total_active_occupant_count", "candidate_capacity", "candidate_walking_distance",
    "candidate_traversable", "candidate_adjacent_zone_occupancy", "candidate_queue_length",
    "candidate_approaching_count", "candidate_congestion_level",
)
EXPERIMENTAL_FEATURE_NAMES = ("candidate_recent_flow_rate", "candidate_congestion_trend", "candidate_alternative_route_count")

_COMPACT_DTYPES = {
    "observation_time": "float32",
    "total_active_occupant_count": "int32",
    "candidate_capacity": "float32",
    "candidate_walking_distance": "float32",
    "candidate_traversable": "bool",
    "candidate_adjacent_zone_occupancy": "float32",
    "candidate_queue_length": "float32",
    "candidate_approaching_count": "float32",
    "candidate_recent_flow_rate": "float32",
    "candidate_alternative_route_count": "int32",
}


def _mem_log(label: str) -> None:
    vm = psutil.virtual_memory()
    print(f"    [mem] {label}: available={vm.available/1e6:.0f}MB ({vm.percent:.0f}% used)", flush=True)


def _load() -> pd.DataFrame:

    keep_columns = (
        ["scenario_id", "observation_time", "candidate_id", "candidate_type"]
        + list(BASE_FEATURE_NAMES) + list(EXPERIMENTAL_FEATURE_NAMES)
        + ["target_v2"]
    )

    chunks = []
    for chunk in pd.read_csv(CSV_PATH, chunksize=250_000, usecols=keep_columns):
        chunk = chunk.astype({k: v for k, v in _COMPACT_DTYPES.items() if k in chunk.columns})
        chunks.append(chunk)

    frame = pd.concat(chunks, ignore_index=True)
    del chunks

    for column in ("scenario_id", "candidate_id", "candidate_type", "candidate_congestion_level", "candidate_congestion_trend"):
        frame[column] = frame[column].astype("category")

    # build_experimental_feature_matrix expects a "target" column
    frame = frame.rename(columns={"target_v2": "target"})

    return frame


def _is_trivial_baseline(name: str) -> bool:
    return name in ("majority_class", "always_negative", "random")


def main() -> None:

    overall_start = time.time()
    _mem_log("start")

    with open(SCENARIO_METADATA_PATH, "r", encoding="utf-8") as f:
        scenario_metadata = json.load(f)
    meta_by_scenario = {m["scenario_id"]: m for m in scenario_metadata}

    print("Loading relabeled dataset (Target V2 label only)...")
    t0 = time.time()
    frame = _load()
    load_seconds = time.time() - t0
    print(f"    Loaded {len(frame)} rows in {load_seconds:.1f}s")
    _mem_log("after load")

    all_scenario_ids = frame["scenario_id"].astype(str).unique().tolist()
    split = split_scenarios(all_scenario_ids, seed=SEED)
    print(f"Scenario split: train={len(split.train_scenario_ids)} val={len(split.val_scenario_ids)} "
          f"test={len(split.test_scenario_ids)}")

    train_df, val_df, test_df = apply_split(frame, split)
    assert_no_scenario_overlap(split, train_df, val_df, test_df)

    train_trainable = train_df[train_df["target"].notna()].copy()
    val_trainable = val_df[val_df["target"].notna()].copy()
    test_trainable = test_df[test_df["target"].notna()].copy()

    train_feat = build_experimental_feature_matrix(train_trainable)
    val_feat = build_experimental_feature_matrix(val_trainable)
    test_feat = build_experimental_feature_matrix(test_trainable)

    print(f"Trainable rows: train={len(train_feat.y)} val={len(val_feat.y)} test={len(test_feat.y)} "
          f"(train positive rate={train_feat.y.mean():.4f})")
    _mem_log("after feature matrices")

    report: Dict[str, Any] = {
        "target_version": "v2-persistent-demand-service-imbalance",
        "horizon_seconds": HORIZON,
        "scenario_split": split.to_dict(),
        "trainable_row_counts": {"train": int(len(train_feat.y)), "val": int(len(val_feat.y)), "test": int(len(test_feat.y))},
        "models": {},
    }

    models = {
        **build_baselines(seed=SEED),
        "gradient_boosting": build_tree_models(seed=SEED, n_jobs=N_JOBS)["gradient_boosting"],
        "xgboost": build_tree_models(seed=SEED, n_jobs=N_JOBS)["xgboost"],
    }

    best_name = None
    best_pr_auc = -1.0
    best_model = None
    best_test_prob = None
    best_threshold = 0.5

    for name, model in models.items():

        t_start = time.time()
        trivial = _is_trivial_baseline(name)

        if trivial:
            model.fit(train_feat.X, train_feat.y)
        else:
            cw = compute_class_weight_map(train_feat.y)
            sw = sample_weights_from_class_weight(train_feat.y, cw)
            model.fit(train_feat.X, train_feat.y, sample_weight=sw)

        val_prob = model.predict_proba(val_feat.X)
        test_prob = model.predict_proba(test_feat.X)

        threshold = 0.5 if trivial else tune_threshold(val_feat.y, val_prob, metric="f1")[0]

        test_metrics = compute_metrics(test_feat.y, test_prob, threshold=threshold)
        test_by_type = metrics_by_group(test_feat.y, test_prob, test_feat.candidate_types, threshold=threshold)

        report["models"][name] = {
            "threshold": threshold,
            "test_metrics": test_metrics.to_dict(),
            "test_metrics_by_candidate_type": test_by_type,
            "train_seconds": time.time() - t_start,
        }

        print(f"    {name}: test ROC-AUC={test_metrics.roc_auc} PR-AUC={test_metrics.pr_auc} ({time.time()-t_start:.1f}s)")
        _mem_log(f"after {name}")

        if not trivial and test_metrics.pr_auc is not None and test_metrics.pr_auc > best_pr_auc:
            best_pr_auc = test_metrics.pr_auc
            best_name = name
            best_model = model
            best_test_prob = test_prob
            best_threshold = threshold

    print(f"BEST MODEL: {best_name}")
    report["best_model"] = best_name

    # by-topology-family breakdown for the best model
    test_trainable_reset = test_trainable.reset_index(drop=True)
    test_trainable_reset["topology_family"] = test_trainable_reset["scenario_id"].astype(str).map(
        lambda sid: meta_by_scenario.get(sid, {}).get("topology_family")
    )
    by_topology: Dict[str, Any] = {}
    for fam in sorted(test_trainable_reset["topology_family"].dropna().unique()):
        mask = (test_trainable_reset["topology_family"] == fam).to_numpy()
        if mask.sum() == 0:
            continue
        fam_metrics = compute_metrics(test_feat.y[mask], best_test_prob[mask], threshold=best_threshold)
        by_topology[fam] = fam_metrics.to_dict()
        print(f"    topology[{fam}]: n={mask.sum()} PR-AUC={fam_metrics.pr_auc} ROC-AUC={fam_metrics.roc_auc}")

    report["best_model_by_topology_family"] = by_topology
    report["total_wall_seconds"] = time.time() - overall_start

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "exploratory_model_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nDone in {report['total_wall_seconds']:.1f}s. Report written to {report_path}")


if __name__ == "__main__":
    main()
