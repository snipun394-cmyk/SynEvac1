"""Localized Predictive Model V2.2 milestone, Phase 6 -- FULL-SCALE
feature extraction. Regenerates the SAME 4 topology families at V2's
ORIGINAL scenario counts (500/800/700/500 = 2,500, same
master_seed=20270115 as both Predictive Dataset V2 and the V2.1
targeted experiment) and extracts the 9-field V2 schema PLUS the 3
V2.1 experimental fields in one pass.

Deterministic regeneration, not "new data": same master_seed + same
topology definitions (predictive_dataset/topologies_v2.py, byte-for-byte
unchanged since commit a3a2c56) produce the literal same simulated
scenarios/occupants Predictive Dataset V2 already validated -- this
script only adds 3 more feature columns per row, it does not draw a new
population. Per this milestone's own "prefer reuse/re-extraction... do
not regenerate millions of rows unnecessarily" instruction, the
BASELINE 9-field comparison numbers are read from the ALREADY-EXISTING
data/predictive_dataset_campaign_v2/candidate_dataset_v2.csv (never
regenerated here) -- this script writes ONLY the 12-field experimental
CSV, not a second copy of the 9-field baseline.

Unlike scripts/run_predictive_dataset_campaign_v2_1_experiment.py (which
accumulates all rows in memory before writing -- fine at 500 scenarios,
~467K rows), this script STREAMS rows to CSV as they are produced.
Holding ~2.4M rows of Python dicts in memory on this ~7.3GB-RAM
development machine risked exactly the kind of near-OOM the Model V2
milestone's own first RandomForest attempt hit -- streaming bounds
memory to one scenario's rows at a time (order ~1,000-4,000).

Usage: python scripts/run_predictive_dataset_campaign_v2_2_fullscale.py
"""

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psutil

from behaviour_profile_resolver import register_occupants
from scenario_generator.batch_generator import iter_batch
from scenario_generator.request import BatchGenerationRequest
from scenario_runner import run as run_scenario_context
from simulation_runtime import SimulationRuntime
from ai_decision.engine import AIDecisionEngine

from scenario.engineering_state import DoorState, StairAvailability

from predictive_dataset.campaign_config_v2 import MASTER_SEED_V2, MINIMUM_END_TIME_SECONDS
from predictive_dataset.candidate import edges_by_candidate_id, enumerate_candidates
from predictive_dataset.simulation_extractor_v2_1 import (
    EXPERIMENTAL_FEATURE_NAMES,
    build_alternative_route_counts,
    extract_experimental_candidate_features,
)
from predictive_dataset.target_generator import generate_candidate_label
from predictive_dataset.topologies_v2 import all_topology_specs

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "predictive_dataset_campaign_v2_2"

HORIZON = 20.0

BASE_FEATURE_NAMES = (
    "total_active_occupant_count", "candidate_capacity", "candidate_walking_distance",
    "candidate_traversable", "candidate_adjacent_zone_occupancy", "candidate_queue_length",
    "candidate_approaching_count", "candidate_congestion_level",
)

IDENTITY_COLUMNS = ("scenario_id", "observation_time", "candidate_id", "candidate_type")
LABEL_COLUMNS = ("currently_congested", "had_any_activity_in_window", "target")

EXPERIMENTAL_CSV_COLUMNS = IDENTITY_COLUMNS + BASE_FEATURE_NAMES + EXPERIMENTAL_FEATURE_NAMES + LABEL_COLUMNS

MIN_AVAILABLE_MEMORY_BYTES = 300_000_000


def _check_memory(label: str) -> None:
    vm = psutil.virtual_memory()
    if vm.available < MIN_AVAILABLE_MEMORY_BYTES:
        raise MemoryError(f"Available memory critically low ({vm.available/1e6:.0f}MB) at {label!r} -- aborting.")


def main() -> None:

    specs = all_topology_specs()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    experimental_csv_path = OUTPUT_DIR / "candidate_dataset_experimental.csv"

    scenario_metadata = []
    accepted = 0
    failed = 0
    row_count = 0

    campaign_start = time.perf_counter()

    with open(experimental_csv_path, "w", newline="", encoding="utf-8") as csv_file:

        writer = csv.DictWriter(csv_file, fieldnames=EXPERIMENTAL_CSV_COLUMNS)
        writer.writeheader()

        for spec in specs:

            scenario_count = spec.scenario_count  # FULL V2 scale, no reduction

            candidates = enumerate_candidates(spec.building)
            edges = edges_by_candidate_id(spec.building)
            alt_route_counts = build_alternative_route_counts(candidates)

            floor_count = len(spec.building.floors)
            exit_count = sum(len(floor.exits) for floor in spec.building.floors)
            stair_count = sum(len(floor.stairs) for floor in spec.building.floors)
            door_count = sum(len(floor.doors) for floor in spec.building.floors)

            request = BatchGenerationRequest(
                definition=spec.definition, definition_id=f"predictive-dataset-v2-2-fullscale-{spec.name}",
                building=spec.building, master_seed=MASTER_SEED_V2, count=scenario_count,
            )

            print(f"[{spec.name}] generating {scenario_count} scenarios "
                  f"(alt_route_counts={alt_route_counts})", flush=True)

            family_index = 0

            for scenario in iter_batch(request):

                family_index += 1
                if family_index % 100 == 0:
                    print(f"  [{spec.name}] {family_index}/{scenario_count} "
                          f"(elapsed {time.perf_counter() - campaign_start:.1f}s, rows so far {row_count})", flush=True)
                    _check_memory(f"{spec.name} {family_index}/{scenario_count}")

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
                    scenario_row_count = 0

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

                            row = {
                                "scenario_id": scenario.metadata.scenario_id,
                                "observation_time": tick.time,
                                "candidate_id": candidate.candidate_id,
                                "candidate_type": candidate.candidate_type,
                            }
                            for name in BASE_FEATURE_NAMES + EXPERIMENTAL_FEATURE_NAMES:
                                row[name] = experimental_features[name]
                            row["currently_congested"] = label.currently_congested
                            row["had_any_activity_in_window"] = label.had_any_activity_in_window
                            row["target"] = label.target

                            writer.writerow(row)
                            scenario_row_count += 1

                    row_count += scenario_row_count
                    accepted += 1

                    fire = scenario.fire
                    blocked_door_count = sum(1 for state in scenario.door_states if state.state != DoorState.OPEN)
                    blocked_exit_count = sum(1 for state in scenario.exit_states if not state.is_open)
                    unavailable_stair_count = sum(
                        1 for state in scenario.stair_states if state.availability == StairAvailability.CLOSED
                    )

                    scenario_metadata.append({
                        "scenario_id": scenario.metadata.scenario_id,
                        "topology_family": spec.name,
                        "floor_count": floor_count, "exit_count": exit_count,
                        "stair_count": stair_count, "door_count": door_count,
                        "total_occupants": total_occupants,
                        "ignition_zone_id": fire.ignition_zone_id if fire is not None else None,
                        "fire_growth_time_seconds": fire.growth_parameters.get("growth_time") if fire is not None else None,
                        "fire_profile": fire.fire_profile if fire is not None else None,
                        "blocked_door_count": blocked_door_count,
                        "blocked_exit_count": blocked_exit_count,
                        "unavailable_stair_count": unavailable_stair_count,
                        "evacuation_duration": movement_result.total_evacuation_time,
                        "unreachable_occupant_count": len(movement_result.unreachable_occupant_ids),
                        "contributed_rows": scenario_row_count > 0,
                    })

                except Exception as exc:  # noqa: BLE001 -- a failed scenario is skipped, not fatal
                    failed += 1
                    print(f"  [{spec.name}] scenario failed: {exc}", flush=True)

    elapsed = time.perf_counter() - campaign_start
    print(f"Accepted {accepted}, failed {failed}. Rows: {row_count}. Wall: {elapsed:.1f}s")

    with open(OUTPUT_DIR / "scenario_metadata.json", "w", encoding="utf-8") as f:
        json.dump(scenario_metadata, f, indent=2)

    report = {
        "horizon_seconds": HORIZON,
        "master_seed": MASTER_SEED_V2,
        "accepted_scenarios": accepted,
        "failed_scenarios": failed,
        "row_count": row_count,
        "wall_seconds": elapsed,
        "experimental_csv_columns": list(EXPERIMENTAL_CSV_COLUMNS),
        "note": (
            "Baseline 9-field comparison numbers are read from the ALREADY-EXISTING "
            "data/predictive_dataset_campaign_v2/candidate_dataset_v2.csv, never regenerated here."
        ),
    }
    with open(OUTPUT_DIR / "campaign_v2_2_fullscale_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote {experimental_csv_path}")


if __name__ == "__main__":
    main()
