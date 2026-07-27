"""Localized Predictive Model V3 milestone, Task #42 -- horizon
sensitivity check (10s/20s/30s), REDUCED SCALE.

The already-relabeled Target V2 CSV (candidate_dataset_relabeled.csv)
was generated at a single, fixed 20s horizon -- extending the label
window to 30s (or shrinking to 10s) requires re-deriving labels from
each scenario's raw movement_result, which was never persisted (see
[[predictive_congestion_target_v2_milestone]]). Re-simulating all 2,500
scenarios just for a secondary sensitivity check would cost the same
~5 minutes as the original full campaign for a check the milestone
charter explicitly scopes as "if cheap" / reduced-scale -- so this
script deterministically re-simulates a SMALL, topology-diverse subset
(100 scenarios per topology family = 400 total, same master_seed as
every other Target V2 artifact -- re-extraction, not new data, exactly
the pattern already established for full-scale campaigns) and computes
ALL THREE horizons' labels from the SAME onsets/movement_result per
scenario -- so the only added cost vs a single-horizon run is three
cheap label-generation calls per row, not three simulation passes.

target_generator_v2.generate_candidate_label_v2() is horizon-
parameterized already -- this script does not modify target semantics
in any way, it only calls it three times with different horizon
values on the same precomputed onsets.
"""
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from behaviour_profile_resolver import register_occupants
from scenario_generator.batch_generator import iter_batch
from scenario_generator.request import BatchGenerationRequest
from scenario_runner import run as run_scenario_context
from simulation_runtime import SimulationRuntime
from ai_decision.engine import AIDecisionEngine

from predictive_dataset.campaign_config_v2 import MASTER_SEED_V2, MINIMUM_END_TIME_SECONDS
from predictive_dataset.candidate import edges_by_candidate_id, enumerate_candidates
from predictive_dataset.simulation_extractor_v2_1 import (
    EXPERIMENTAL_FEATURE_NAMES,
    build_alternative_route_counts,
    extract_experimental_candidate_features,
)
from predictive_dataset.target_generator_v2 import compute_qualifying_onsets, generate_candidate_label_v2
from predictive_dataset.topologies_v2 import all_topology_specs

import train_localized_predictive_model_v3 as v3
from predictive_model.feature_prep_v2_1 import build_experimental_feature_matrix
from predictive_model.imbalance import compute_class_weight_map, sample_weights_from_class_weight
from predictive_model.metrics import compute_metrics
from predictive_model.scenario_split import apply_split, assert_no_scenario_overlap, split_scenarios

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = v3.OUTPUT_DIR / "horizon_sensitivity_experiment.json"

SCENARIOS_PER_FAMILY = 100
HORIZONS = (10.0, 20.0, 30.0)

BASE_FEATURE_NAMES = v3.BASE_FEATURE_NAMES
IDENTITY_COLUMNS = ("scenario_id", "observation_time", "candidate_id", "candidate_type")


def _simulate_and_label(spec, n_scenarios: int) -> list:

    candidates = enumerate_candidates(spec.building)
    edges = edges_by_candidate_id(spec.building)
    alt_route_counts = build_alternative_route_counts(candidates)

    request = BatchGenerationRequest(
        definition=spec.definition, definition_id=f"model-v3-horizon-sensitivity-{spec.name}",
        building=spec.building, master_seed=MASTER_SEED_V2, count=spec.scenario_count,
    )

    rows = []
    used = 0

    for scenario in iter_batch(request):

        if used >= n_scenarios:
            break

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
                candidate.candidate_id: compute_qualifying_onsets(movement_result, candidate.candidate_id)
                for candidate in candidates
            }

            for tick in tick_results:
                for candidate in candidates:

                    edge = edges[candidate.candidate_id]
                    onsets = onsets_by_candidate[candidate.candidate_id]

                    features = extract_experimental_candidate_features(
                        candidate, edge, tick.time,
                        building=building, movement_result=movement_result,
                        occupancy_snapshot=tick.occupancy_snapshot,
                        alternative_route_counts=alt_route_counts,
                    )

                    row = {
                        "scenario_id": scenario.metadata.scenario_id,
                        "observation_time": tick.time,
                        "candidate_id": candidate.candidate_id,
                        "candidate_type": candidate.candidate_type,
                        "topology_family": spec.name,
                    }
                    for name in BASE_FEATURE_NAMES + EXPERIMENTAL_FEATURE_NAMES:
                        row[name] = features[name]

                    for horizon in HORIZONS:
                        label = generate_candidate_label_v2(
                            candidate.candidate_id, movement_result, tick.time, horizon, onsets=onsets,
                        )
                        suffix = int(horizon)
                        row[f"currently_congested_{suffix}"] = label.currently_congested
                        row[f"target_{suffix}"] = label.target

                    rows.append(row)

            used += 1

        except Exception as exc:  # noqa: BLE001
            print(f"  [{spec.name}] scenario failed: {exc}", flush=True)

    return rows


def main() -> None:

    t_start = time.time()
    all_rows = []

    for spec in all_topology_specs():
        n = min(SCENARIOS_PER_FAMILY, spec.scenario_count)
        print(f"Simulating {n} scenarios for topology_family={spec.name} ...", flush=True)
        t0 = time.time()
        rows = _simulate_and_label(spec, n)
        all_rows.extend(rows)
        print(f"    {len(rows)} rows in {time.time() - t0:.1f}s", flush=True)

    frame = pd.DataFrame(all_rows)
    del all_rows
    gc.collect()

    print(f"Total rows: {len(frame)}, scenarios: {frame['scenario_id'].nunique()}, "
          f"elapsed so far {time.time() - t_start:.1f}s")

    split = split_scenarios(frame["scenario_id"].astype(str).unique().tolist(), seed=v3.SEED)
    train_df, val_df, test_df = apply_split(frame, split)
    assert_no_scenario_overlap(split, train_df, val_df, test_df)
    del frame
    gc.collect()

    results = {}

    for horizon in HORIZONS:

        suffix = int(horizon)
        print(f"Horizon {suffix}s ...")

        horizon_train = train_df.rename(columns={
            f"target_{suffix}": "target", f"currently_congested_{suffix}": "currently_congested",
        })
        horizon_test = test_df.rename(columns={
            f"target_{suffix}": "target", f"currently_congested_{suffix}": "currently_congested",
        })

        train_trainable = horizon_train[horizon_train["target"].notna()].copy()
        test_trainable = horizon_test[horizon_test["target"].notna()].copy()

        train_feat = build_experimental_feature_matrix(train_trainable)
        test_feat = build_experimental_feature_matrix(test_trainable)

        model = v3._fresh_model("xgboost")
        class_weight_map = compute_class_weight_map(train_feat.y)
        sample_weight = sample_weights_from_class_weight(train_feat.y, class_weight_map)
        model.fit(train_feat.X, train_feat.y, sample_weight=sample_weight)

        test_prob = model.predict_proba(test_feat.X)
        metrics = compute_metrics(test_feat.y, test_prob, threshold=0.5)

        results[str(suffix)] = {
            "horizon_seconds": horizon,
            "train_row_count": int(len(train_feat.y)),
            "train_positive_rate": float(train_feat.y.mean()),
            "test_row_count": int(len(test_feat.y)),
            "test_positive_rate": float(test_feat.y.mean()),
            "test_metrics": metrics.to_dict(),
        }
        print(f"    n_train={len(train_feat.y)} pos_rate={train_feat.y.mean():.4f} "
              f"test PR-AUC={metrics.pr_auc:.4f} ROC-AUC={metrics.roc_auc:.4f}")

        del horizon_train, horizon_test, train_trainable, test_trainable, train_feat, test_feat, model, test_prob
        gc.collect()

    report = {
        "experiment": "horizon_sensitivity_v3",
        "note": "REDUCED SCALE -- 100 scenarios/topology family (400 total), fresh scenario-level "
                "split at this reduced scale (same seed/convention, NOT the full-scale production split). "
                "Primary/production horizon remains 20s; this is a sensitivity check only.",
        "scenarios_per_family": SCENARIOS_PER_FAMILY,
        "master_seed": MASTER_SEED_V2,
        "training_split_seed": v3.SEED,
        "scenario_split": {
            "train_scenario_count": len(split.train_scenario_ids),
            "val_scenario_count": len(split.val_scenario_ids),
            "test_scenario_count": len(split.test_scenario_ids),
        },
        "by_horizon": results,
        "total_wall_seconds": time.time() - t_start,
    }

    v3.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Done in {time.time() - t_start:.1f}s. Report written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
