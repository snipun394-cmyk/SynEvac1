"""Localized Predictive Model V2.1 milestone, Phase 11-12 -- EXPERIMENTAL,
targeted hypothesis-test campaign. NOT a replacement for Predictive Dataset
V2 (data/predictive_dataset_campaign_v2/, frozen, untouched by this script).

Generates a SMALL, reduced-scale campaign (~20% of V2's scenario counts,
same 4 topology families, same master_seed=20270115, same 20s horizon
only) from the SAME movement-simulation runs, but extracts features
TWICE per row: once with the frozen V2 9-field schema, once with the
same 9 fields PLUS the 3 experimental fields from
predictive_dataset/simulation_extractor_v2_1.py. This isolates the
FEATURE effect from the "less training data" effect cleanly -- both
CSVs it writes describe the literal same simulated occupants/scenarios,
differing only in which columns exist.

Per the V2.1 milestone's own "training performance already saturates
around 25%" instruction, this deliberately does NOT rerun anything
close to the full 2500-scenario campaign.

Usage: python scripts/run_predictive_dataset_campaign_v2_1_experiment.py
"""

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from behaviour_profile_resolver import register_occupants
from scenario_generator.batch_generator import iter_batch
from scenario_generator.request import BatchGenerationRequest
from scenario_runner import run as run_scenario_context
from simulation_runtime import SimulationRuntime
from ai_decision.engine import AIDecisionEngine

from predictive_dataset.campaign_config_v2 import MASTER_SEED_V2, MINIMUM_END_TIME_SECONDS
from predictive_dataset.candidate import edges_by_candidate_id, enumerate_candidates
from predictive_dataset.simulation_extractor import extract_simulation_candidate_features
from predictive_dataset.simulation_extractor_v2_1 import (
    EXPERIMENTAL_FEATURE_NAMES,
    build_alternative_route_counts,
    extract_experimental_candidate_features,
)
from predictive_dataset.target_generator import generate_candidate_label
from predictive_dataset.topologies_v2 import all_topology_specs

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "predictive_dataset_campaign_v2_1_experiment"

HORIZON = 20.0
SCALE_FRACTION = 0.20  # ~20% of V2's own per-family scenario_count

BASE_FEATURE_NAMES = (
    "total_active_occupant_count", "candidate_capacity", "candidate_walking_distance",
    "candidate_traversable", "candidate_adjacent_zone_occupancy", "candidate_queue_length",
    "candidate_approaching_count", "candidate_congestion_level",
)

IDENTITY_COLUMNS = ("scenario_id", "observation_time", "candidate_id", "candidate_type")
LABEL_COLUMNS = ("currently_congested", "had_any_activity_in_window", "target")

BASELINE_CSV_COLUMNS = IDENTITY_COLUMNS + BASE_FEATURE_NAMES + LABEL_COLUMNS
EXPERIMENTAL_CSV_COLUMNS = IDENTITY_COLUMNS + BASE_FEATURE_NAMES + EXPERIMENTAL_FEATURE_NAMES + LABEL_COLUMNS


def main() -> None:

    specs = all_topology_specs()

    baseline_rows = []
    experimental_rows = []
    scenario_metadata = []

    accepted = 0
    failed = 0

    campaign_start = time.perf_counter()

    for spec in specs:

        scenario_count = max(1, round(spec.scenario_count * SCALE_FRACTION))

        candidates = enumerate_candidates(spec.building)
        edges = edges_by_candidate_id(spec.building)
        alt_route_counts = build_alternative_route_counts(candidates)

        floor_count = len(spec.building.floors)
        exit_count = sum(len(floor.exits) for floor in spec.building.floors)
        stair_count = sum(len(floor.stairs) for floor in spec.building.floors)
        door_count = sum(len(floor.doors) for floor in spec.building.floors)

        request = BatchGenerationRequest(
            definition=spec.definition, definition_id=f"predictive-dataset-v2-1-experiment-{spec.name}",
            building=spec.building, master_seed=MASTER_SEED_V2, count=scenario_count,
        )

        print(f"[{spec.name}] generating {scenario_count} scenarios (alt_route_counts={alt_route_counts})")

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

                total_occupants = len(movement_result.occupants)

                for tick in tick_results:

                    for candidate in candidates:

                        edge = edges[candidate.candidate_id]

                        experimental_features = extract_experimental_candidate_features(
                            candidate, edge, tick.time,
                            building=building, movement_result=movement_result,
                            occupancy_snapshot=tick.occupancy_snapshot,
                            alternative_route_counts=alt_route_counts,
                        )

                        label = generate_candidate_label(candidate.candidate_id, movement_result, tick.time, HORIZON)

                        identity = {
                            "scenario_id": scenario.metadata.scenario_id,
                            "observation_time": tick.time,
                            "candidate_id": candidate.candidate_id,
                            "candidate_type": candidate.candidate_type,
                        }
                        labels = {
                            "currently_congested": label.currently_congested,
                            "had_any_activity_in_window": label.had_any_activity_in_window,
                            "target": label.target,
                        }

                        baseline_row = dict(identity)
                        for name in BASE_FEATURE_NAMES:
                            baseline_row[name] = experimental_features[name]
                        baseline_row.update(labels)
                        baseline_rows.append(baseline_row)

                        experimental_row = dict(identity)
                        for name in BASE_FEATURE_NAMES + EXPERIMENTAL_FEATURE_NAMES:
                            experimental_row[name] = experimental_features[name]
                        experimental_row.update(labels)
                        experimental_rows.append(experimental_row)

                accepted += 1
                scenario_metadata.append({
                    "scenario_id": scenario.metadata.scenario_id,
                    "topology_family": spec.name,
                    "floor_count": floor_count, "exit_count": exit_count,
                    "stair_count": stair_count, "door_count": door_count,
                    "total_occupants": total_occupants,
                })

            except Exception as exc:  # noqa: BLE001 -- a failed scenario is skipped, not fatal to the campaign
                failed += 1
                print(f"  [{spec.name}] scenario failed: {exc}")

    elapsed = time.perf_counter() - campaign_start
    print(f"Accepted {accepted}, failed {failed}. Rows: {len(baseline_rows)}. Wall: {elapsed:.1f}s")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_csv = OUTPUT_DIR / "candidate_dataset_baseline.csv"
    with open(baseline_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BASELINE_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(baseline_rows)

    experimental_csv = OUTPUT_DIR / "candidate_dataset_experimental.csv"
    with open(experimental_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENTAL_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(experimental_rows)

    with open(OUTPUT_DIR / "scenario_metadata.json", "w", encoding="utf-8") as f:
        json.dump(scenario_metadata, f, indent=2)

    report = {
        "scale_fraction": SCALE_FRACTION,
        "horizon_seconds": HORIZON,
        "master_seed": MASTER_SEED_V2,
        "accepted_scenarios": accepted,
        "failed_scenarios": failed,
        "row_count": len(baseline_rows),
        "wall_seconds": elapsed,
        "baseline_csv_columns": list(BASELINE_CSV_COLUMNS),
        "experimental_csv_columns": list(EXPERIMENTAL_CSV_COLUMNS),
    }
    with open(OUTPUT_DIR / "campaign_v2_1_experiment_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote {baseline_csv}, {experimental_csv}")


if __name__ == "__main__":
    main()
