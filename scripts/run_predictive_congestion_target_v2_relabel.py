"""Predictive Congestion Target V2 milestone, Phase 8/14 -- FULL-SCALE
relabeling. Deterministically reproduces the SAME 2,500 scenarios
Predictive Dataset V2 / Model V2.2 already used (same master_seed
=20270115, same predictive_dataset/topologies_v2.py definitions,
byte-for-byte reproduction -- see scripts/
run_predictive_dataset_campaign_v2_2_fullscale.py's own docstring for
why this counts as re-extraction, not new data) and, for every
candidate-time row, computes BOTH the V2.2 12-field feature schema AND
BOTH congestion targets side by side:

  - target_v1: predictive_dataset.target_generator (COMPLETELY
    UNTOUCHED by this milestone -- Target V1 stays exactly reproducible).
  - target_v2: predictive_dataset.target_generator_v2 (the new, frozen,
    persistent-demand-service-imbalance definition).

Writing both targets from the SAME simulation runs into the SAME CSV is
what makes the Phase 16 V1-vs-V2 agreement/disagreement analysis exact
row-for-row, not an approximation across two separately-generated
datasets.

Streams rows directly to CSV (never accumulates in memory) -- the same
discipline scripts/run_predictive_dataset_campaign_v2_2_fullscale.py
already established, avoiding the 9.6M-row memory problem this
milestone's own Phase 24 explicitly warns against repeating.

Usage: python scripts/run_predictive_congestion_target_v2_relabel.py
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
from predictive_dataset.target_generator_v2 import compute_qualifying_onsets, generate_candidate_label_v2
from predictive_dataset.topologies_v2 import all_topology_specs

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "predictive_congestion_target_v2"

HORIZON = 20.0

BASE_FEATURE_NAMES = (
    "total_active_occupant_count", "candidate_capacity", "candidate_walking_distance",
    "candidate_traversable", "candidate_adjacent_zone_occupancy", "candidate_queue_length",
    "candidate_approaching_count", "candidate_congestion_level",
)

IDENTITY_COLUMNS = ("scenario_id", "observation_time", "candidate_id", "candidate_type")
V1_LABEL_COLUMNS = ("currently_congested_v1", "had_any_activity_in_window_v1", "target_v1")
V2_LABEL_COLUMNS = ("currently_congested_v2", "had_any_activity_in_window_v2", "target_v2", "lead_time_seconds_v2")

CSV_COLUMNS = IDENTITY_COLUMNS + BASE_FEATURE_NAMES + EXPERIMENTAL_FEATURE_NAMES + V1_LABEL_COLUMNS + V2_LABEL_COLUMNS

MIN_AVAILABLE_MEMORY_BYTES = 300_000_000


def _check_memory(label: str) -> None:
    vm = psutil.virtual_memory()
    if vm.available < MIN_AVAILABLE_MEMORY_BYTES:
        raise MemoryError(f"Available memory critically low ({vm.available/1e6:.0f}MB) at {label!r} -- aborting.")


def _min_lead_time(onsets, time: float, horizon: float):

    candidates = [onset - time for onset, _ in onsets if time < onset <= time + horizon]
    return min(candidates) if candidates else None


def main() -> None:

    specs = all_topology_specs()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "candidate_dataset_relabeled.csv"

    scenario_metadata = []
    accepted = 0
    failed = 0
    row_count = 0

    campaign_start = time.perf_counter()

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:

        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for spec in specs:

            candidates = enumerate_candidates(spec.building)
            edges = edges_by_candidate_id(spec.building)
            alt_route_counts = build_alternative_route_counts(candidates)

            floor_count = len(spec.building.floors)
            exit_count = sum(len(floor.exits) for floor in spec.building.floors)
            stair_count = sum(len(floor.stairs) for floor in spec.building.floors)
            door_count = sum(len(floor.doors) for floor in spec.building.floors)

            request = BatchGenerationRequest(
                definition=spec.definition, definition_id=f"predictive-congestion-target-v2-{spec.name}",
                building=spec.building, master_seed=MASTER_SEED_V2, count=spec.scenario_count,
            )

            print(f"[{spec.name}] {spec.scenario_count} scenarios", flush=True)
            family_index = 0

            for scenario in iter_batch(request):

                family_index += 1
                if family_index % 200 == 0:
                    print(f"  [{spec.name}] {family_index}/{spec.scenario_count} "
                          f"(elapsed {time.perf_counter() - campaign_start:.1f}s, rows so far {row_count})", flush=True)
                    _check_memory(f"{spec.name} {family_index}")

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

                    # onsets depend only on candidate_id, not tick time --
                    # precompute ONCE per candidate per scenario (Phase 24's
                    # own "avoid unnecessary duplication" instruction).
                    onsets_by_candidate = {
                        candidate.candidate_id: compute_qualifying_onsets(movement_result, candidate.candidate_id)
                        for candidate in candidates
                    }

                    scenario_row_count = 0

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

                            label_v1 = generate_candidate_label(candidate.candidate_id, movement_result, tick.time, HORIZON)
                            label_v2 = generate_candidate_label_v2(
                                candidate.candidate_id, movement_result, tick.time, HORIZON, onsets=onsets,
                            )

                            row = {
                                "scenario_id": scenario.metadata.scenario_id,
                                "observation_time": tick.time,
                                "candidate_id": candidate.candidate_id,
                                "candidate_type": candidate.candidate_type,
                            }
                            for name in BASE_FEATURE_NAMES + EXPERIMENTAL_FEATURE_NAMES:
                                row[name] = features[name]

                            row["currently_congested_v1"] = label_v1.currently_congested
                            row["had_any_activity_in_window_v1"] = label_v1.had_any_activity_in_window
                            row["target_v1"] = label_v1.target

                            row["currently_congested_v2"] = label_v2.currently_congested
                            row["had_any_activity_in_window_v2"] = label_v2.had_any_activity_in_window
                            row["target_v2"] = label_v2.target
                            row["lead_time_seconds_v2"] = (
                                _min_lead_time(onsets, tick.time, HORIZON) if label_v2.target else None
                            )

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

                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    print(f"  [{spec.name}] scenario failed: {exc}", flush=True)

    elapsed = time.perf_counter() - campaign_start
    print(f"Accepted {accepted}, failed {failed}. Rows: {row_count}. Wall: {elapsed:.1f}s "
          f"({row_count/elapsed:.0f} rows/sec)")

    with open(OUTPUT_DIR / "scenario_metadata.json", "w", encoding="utf-8") as f:
        json.dump(scenario_metadata, f, indent=2)

    report = {
        "horizon_seconds": HORIZON,
        "master_seed": MASTER_SEED_V2,
        "accepted_scenarios": accepted,
        "failed_scenarios": failed,
        "row_count": row_count,
        "wall_seconds": elapsed,
        "rows_per_second": row_count / elapsed if elapsed > 0 else None,
        "csv_columns": list(CSV_COLUMNS),
    }
    with open(OUTPUT_DIR / "relabel_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
