"""Localized Predictive Model V3.1 milestone, Phase 10 -- controlled
topology-diversity scaling experiment.

Generates a MODEST campaign (200 scenarios each, well within the
milestone's suggested 100-250 range) for the two new structural
variants defined in predictive_dataset/topologies_v3_1_variants.py
(multi_exit_wide_6exit, twin_stair_highrise_3stair) -- deterministic
simulation via the SAME master_seed/dt/feature-extraction/Target V2
pipeline as every other predictive_dataset campaign, Target V2 itself
untouched.

For each of the two topology-holdout-FAILING families, builds a
2-point diversity-vs-PR-AUC learning curve:

  Level 0 (baseline): train on the SAME 3-of-4-family scenario
    population Phase 9/V3 already used, test on the ORIGINAL held-out
    family's ORIGINAL test scenarios (unchanged, for a fair, apples-
    to-apples comparison with V3's own topology-holdout numbers).
  Level 1 (+structural diversity): Level 0's train set PLUS the new
    structural variant's 200 scenarios (same design pattern as the
    held-out family, genuinely different structure) -- test set is
    STILL the original held-out family's original test scenarios,
    unchanged, so any PR-AUC change is attributable only to the added
    training diversity.
"""
import csv
import gc
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from behaviour_profile_resolver import register_occupants
from scenario_generator.batch_generator import iter_batch
from scenario_generator.request import BatchGenerationRequest
from scenario_runner import run as run_scenario_context
from simulation_runtime import SimulationRuntime
from ai_decision.engine import AIDecisionEngine

from predictive_dataset.campaign_config_v2 import MASTER_SEED_V2, MINIMUM_END_TIME_SECONDS
from predictive_dataset.candidate import edges_by_candidate_id, enumerate_candidates
from predictive_dataset.simulation_extractor_v2_1 import (
    EXPERIMENTAL_FEATURE_NAMES, build_alternative_route_counts, extract_experimental_candidate_features,
)
from predictive_dataset.target_generator_v2 import compute_qualifying_onsets, generate_candidate_label_v2
from predictive_dataset.topologies_v3_1_variants import variant_specs

import train_localized_predictive_model_v3 as v3
from predictive_model.imbalance import compute_class_weight_map, sample_weights_from_class_weight
from predictive_model.metrics import compute_metrics
from predictive_model.topology_holdout import apply_topology_holdout, assert_no_holdout_overlap, build_topology_holdout_splits

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "localized_predictive_model_v3_1"
OUTPUT_PATH = OUTPUT_DIR / "topology_diversity_experiment_report.json"

VARIANT_TO_PARENT_FAMILY = {
    "multi_exit_wide_6exit": "multi_exit_wide",
    "twin_stair_highrise_3stair": "twin_stair_highrise",
}

BASE_FEATURE_NAMES = v3.BASE_FEATURE_NAMES
PRIMARY_HORIZON = v3.PRIMARY_HORIZON


def _simulate_variant(spec) -> pd.DataFrame:

    candidates = enumerate_candidates(spec.building)
    edges = edges_by_candidate_id(spec.building)
    alt_route_counts = build_alternative_route_counts(candidates)

    request = BatchGenerationRequest(
        definition=spec.definition, definition_id=f"model-v3-1-topology-diversity-{spec.name}",
        building=spec.building, master_seed=MASTER_SEED_V2, count=spec.scenario_count,
    )

    rows = []

    for scenario in iter_batch(request):
        try:
            context = run_scenario_context(scenario, spec.building)
            register_occupants(context)

            decision_engine = AIDecisionEngine(base_engine=context.engine)
            runtime = SimulationRuntime(context, decision_engine, dt=5.0)
            runtime.clock.end_time = max(runtime.clock.end_time, MINIMUM_END_TIME_SECONDS)

            tick_results = runtime.run()
            movement_result = runtime.movement_result
            building = context.building

            onsets_by_candidate = {
                c.candidate_id: compute_qualifying_onsets(movement_result, c.candidate_id) for c in candidates
            }

            for tick in tick_results:
                for candidate in candidates:
                    edge = edges[candidate.candidate_id]
                    onsets = onsets_by_candidate[candidate.candidate_id]

                    features = extract_experimental_candidate_features(
                        candidate, edge, tick.time, building=building, movement_result=movement_result,
                        occupancy_snapshot=tick.occupancy_snapshot, alternative_route_counts=alt_route_counts,
                    )
                    label = generate_candidate_label_v2(
                        candidate.candidate_id, movement_result, tick.time, PRIMARY_HORIZON, onsets=onsets,
                    )

                    row = {
                        "scenario_id": scenario.metadata.scenario_id, "observation_time": tick.time,
                        "candidate_id": candidate.candidate_id, "candidate_type": candidate.candidate_type,
                    }
                    for name in BASE_FEATURE_NAMES + EXPERIMENTAL_FEATURE_NAMES:
                        row[name] = features[name]
                    row["currently_congested"] = label.currently_congested
                    row["target"] = label.target
                    rows.append(row)

        except Exception as exc:  # noqa: BLE001
            print(f"  [{spec.name}] scenario failed: {exc}", flush=True)

    frame = pd.DataFrame(rows)
    frame["scenario_id"] = frame["scenario_id"].astype(str) + f"::{spec.name}"  # namespace to avoid ID collisions
    return frame


def _fit_xgboost(X, y, sample_weight):
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist",
        eval_metric="logloss", scale_pos_weight=1.0, random_state=v3.SEED, n_jobs=v3.N_JOBS,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def main() -> None:

    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    v3._watchdog_log_path = str(OUTPUT_DIR / "memory_watchdog_topology_diversity.log")
    with open(v3._watchdog_log_path, "w", encoding="utf-8") as f:
        f.write(f"watchdog started {time.time():.1f}\n")
    threading.Thread(target=v3._memory_watchdog, daemon=True).start()

    with open(v3.SCENARIO_METADATA_PATH, "r", encoding="utf-8") as f:
        scenario_metadata = json.load(f)

    print("Simulating structural variants...")
    variant_frames = {}
    for spec in variant_specs():
        t0 = time.time()
        print(f"  simulating {spec.name} ({spec.scenario_count} scenarios)...")
        variant_frames[spec.name] = _simulate_variant(spec)
        print(f"    {len(variant_frames[spec.name])} rows in {time.time()-t0:.1f}s")

    print("Loading canonical Target V2 dataset...")
    frame = v3._load_dataset_chunked(v3.CSV_PATH)
    full_trainable = v3._trainable_rows(frame)
    del frame
    gc.collect()

    holdout_splits = {s.held_out_family: s for s in build_topology_holdout_splits(scenario_metadata)}

    report = {}

    for variant_name, parent_family in VARIANT_TO_PARENT_FAMILY.items():

        print(f"=== {parent_family} (variant: {variant_name}) ===")
        holdout = holdout_splits[parent_family]

        holdout_train_df, holdout_test_df = apply_topology_holdout(full_trainable, holdout)
        assert_no_holdout_overlap(holdout, holdout_train_df, holdout_test_df)

        variant_df = variant_frames[variant_name]
        variant_trainable = variant_df[variant_df["target"].notna()].copy()

        family_report = {}

        for level, train_frame in (
            ("level_0_baseline", holdout_train_df),
            ("level_1_plus_structural_variant", pd.concat([holdout_train_df, variant_trainable], ignore_index=True)),
        ):

            t0 = time.time()
            train_feat = v3.build_experimental_feature_matrix(train_frame)
            test_feat = v3.build_experimental_feature_matrix(holdout_test_df)

            class_weight_map = compute_class_weight_map(train_feat.y)
            sample_weight = sample_weights_from_class_weight(train_feat.y, class_weight_map)
            model = _fit_xgboost(train_feat.X, train_feat.y, sample_weight)

            prob = model.predict_proba(test_feat.X)[:, 1]
            metrics = compute_metrics(test_feat.y, prob, threshold=0.5)

            family_report[level] = {
                "train_row_count": int(len(train_feat.y)),
                "train_scenario_count": int(train_frame["scenario_id"].nunique()),
                "test_metrics": metrics.to_dict(),
                "seconds": time.time() - t0,
            }
            print(f"  {level}: rows={family_report[level]['train_row_count']} "
                  f"PR-AUC={metrics.pr_auc:.4f} ROC-AUC={metrics.roc_auc:.4f} ({time.time()-t0:.1f}s)")

            del train_feat, test_feat, model, prob
            gc.collect()

        report[parent_family] = family_report

        del holdout_train_df, holdout_test_df, variant_trainable
        gc.collect()

    # persist the raw variant datasets too (small, gitignored) for
    # reproducibility / potential reuse -- NOT merged into the canonical
    # candidate_dataset_relabeled.csv.
    for name, vframe in variant_frames.items():
        vframe.to_csv(OUTPUT_DIR / f"variant_dataset_{name}.csv", index=False)

    report["total_wall_seconds"] = time.time() - t_start

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Done in {time.time() - t_start:.1f}s. Report written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
