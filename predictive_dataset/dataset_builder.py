import csv

from typing import Any, Dict, List, Sequence, Tuple

from predictive_dataset.candidate import edges_by_candidate_id, enumerate_candidates
from predictive_dataset.schema import CANDIDATE_FEATURE_NAMES, ROW_IDENTITY_COLUMNS
from predictive_dataset.simulation_extractor import extract_simulation_candidate_features
from predictive_dataset.target_generator import generate_candidate_label


# =====================================================
# Per-Candidate Predictive AI Data Foundation milestone, Phase 3/14 --
# assembles SCENARIO x TIMESTEP x CANDIDATE rows from an already-
# completed simulation run. Takes the SAME TimelineRun shape
# dataset_builder.timeline already defines (scenario/building/
# movement_result/tick_results) -- not redefined here, just accepted
# structurally (duck-typed) so this module has no import-time
# dependency on dataset_builder/ and either module can evolve
# independently, matching that package's own "V1's API is frozen,
# never add a field to SimulationRun" discipline extended to not
# touching TimelineRun either.
#
# SAMPLING INTERVAL (Phase 3): one row per TICK ALREADY PRESENT in
# run.tick_results -- no additional subsampling layer. SimulationRuntime's
# own dt (simulation_runtime/clock.py) is already the caller-chosen,
# coarse observation granularity (never the underlying continuous-time
# event resolution the discrete-event MultiAgentSimulation solves at) --
# reusing it directly avoids a second, redundant resampling concept.
# docs/architecture/localized_predictive_ai_dataset.md documents the
# chosen default (dt=5.0s) and why.
# =====================================================


DEFAULT_HORIZONS: Tuple[float, ...] = (10.0, 20.0, 30.0, 60.0)


def build_candidate_dataset_rows(run, horizons: Sequence[float] = DEFAULT_HORIZONS) -> List[Dict[str, Any]]:

    scenario = run.scenario
    building = run.building
    movement_result = run.movement_result

    candidates = enumerate_candidates(building)
    edges = edges_by_candidate_id(building)

    rows: List[Dict[str, Any]] = []

    for tick in run.tick_results:

        occupancy_snapshot = tick.occupancy_snapshot

        for candidate in candidates:

            edge = edges[candidate.candidate_id]

            features = extract_simulation_candidate_features(
                candidate, edge, tick.time,
                building=building, movement_result=movement_result, occupancy_snapshot=occupancy_snapshot,
            )

            for horizon in horizons:

                label = generate_candidate_label(candidate.candidate_id, movement_result, tick.time, horizon)

                row: Dict[str, Any] = {
                    "scenario_id": scenario.metadata.scenario_id,
                    "observation_time": tick.time,
                    "candidate_id": candidate.candidate_id,
                    "candidate_type": candidate.candidate_type,
                    "prediction_horizon": horizon,
                }
                row.update(features)
                row["currently_congested"] = label.currently_congested
                row["had_any_activity_in_window"] = label.had_any_activity_in_window
                row["target"] = label.target

                rows.append(row)

    return rows


def build_candidate_dataset(runs: Sequence, horizons: Sequence[float] = DEFAULT_HORIZONS) -> List[Dict[str, Any]]:

    rows: List[Dict[str, Any]] = []

    for run in runs:
        rows.extend(build_candidate_dataset_rows(run, horizons=horizons))

    return rows


# =====================================================


CSV_COLUMNS: Tuple[str, ...] = (
    ROW_IDENTITY_COLUMNS
    + ("prediction_horizon",)
    + tuple(name for name in CANDIDATE_FEATURE_NAMES if name != "candidate_type")
    + ("currently_congested", "had_any_activity_in_window", "target")
)


def export_candidate_dataset_csv(rows: Sequence[Dict[str, Any]], path: str) -> str:

    with open(path, "w", newline="", encoding="utf-8") as csv_file:

        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    return path
