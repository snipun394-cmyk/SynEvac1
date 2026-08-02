"""
Hybrid Flow Regions (Option D), Milestone 5 -- Experimental Validation
Through the Calibration Benchmark.

Reuses the four already-completed NIST office-building recreations
(Published Scenario Validation Campaigns V1-V3) EXACTLY as built --
same Building construction functions, same ScenarioDefinition, same
DEFINITION_ID, same MASTER_SEED -- and runs each one twice per seeded
scenario through the already-existing, unmodified
calibration_benchmark.simulation_seam.run_with_overrides():

- BASELINE: StairCapacityModel + StairAwareCongestionModel,
  use_flow_regions=False -- today's production admission control,
  byte-for-byte the same as the original validation campaign.
- EXPERIMENTAL: FlowRegionCapacityModel + FlowRegionCongestionModel,
  use_flow_regions=True (calibration_benchmark.candidates.
  FlowRegionCapacityCandidate's own candidate arm) -- the Milestone
  1-4 opt-in Flow Region architecture.

No SynEvac source file is modified by this script. No default is
promoted. This is read-only evaluation of an already-built, already-
tested, already-opt-in capability.

Metrics beyond the Calibration Benchmark Framework's own five
(evacuation_time, congestion_duration, queue_length,
peak_occupancy_ratio, exit_utilization_balance -- see
calibration_benchmark/metrics.py) are computed directly from each run's
own MultiAgentSimulationResult, using the same "no resimulation, only
read back already-produced fields" discipline: average/maximum queue
wait time, queue-wait percentage of total travel time, per-edge total
queue-wait (used to name the dominant bottleneck), and per-edge
crossing counts (flow through exits/stairs).

KNOWN MEASUREMENT LIMITATION, disclosed rather than patched (see the
companion report): calibration_benchmark.metrics.compute_peak_occupancy_ratio()
reconstructs brand-new, unmapped Edge objects from the Building itself
(via ground_truth.analyzer.ordered_doors/exits/stairs), never the
actual graph.flow_regions used by the real run. Calling
FlowRegionCapacityModel.capacity() on one of those reconstructed edges
never satisfies isinstance(..., FlowRegion), so it always falls back to
its own edge-delegating base_model (StairCapacityModel) --
peak_occupancy_ratio ("Maximum Density" in the framework's own report)
is therefore NOT a valid Flow-Region-aware comparison in this campaign;
every other metric here is read from the real simulated
MultiAgentSimulationResult/GroundTruth and is unaffected.
"""

import json
import os
import statistics as pystats
import sys

sys.path.insert(0, os.path.dirname(__file__))

from run_nist_10story_validation import (  # noqa: E402
    build_nist_10story_building, build_nist_10story_definition,
    DEFINITION_ID as DEFINITION_ID_10, MASTER_SEED as MASTER_SEED_10,
)
from run_nist_18story_validation import (  # noqa: E402
    build_nist_18story_building, build_nist_18story_definition,
    DEFINITION_ID as DEFINITION_ID_18, MASTER_SEED as MASTER_SEED_18,
)
from run_nist_24story_validation import (  # noqa: E402
    build_nist_24story_building, build_nist_24story_definition,
    DEFINITION_ID as DEFINITION_ID_24, MASTER_SEED as MASTER_SEED_24,
)
from run_nist_31story_validation import (  # noqa: E402
    build_nist_31story_building, build_nist_31story_definition,
    DEFINITION_ID as DEFINITION_ID_31, MASTER_SEED as MASTER_SEED_31,
)

from navigation.edge import Edge  # noqa: E402
from navigation.graph_builder import NavigationGraphGenerator  # noqa: E402

from scenario_pipeline import run_batch_pipeline  # noqa: E402

from calibration_benchmark.metrics import compute_exit_utilization_balance  # noqa: E402
from calibration_benchmark.simulation_seam import run_with_overrides  # noqa: E402

from simulator.capacity import StairCapacityModel  # noqa: E402
from simulator.congestion import StairAwareCongestionModel  # noqa: E402
from simulator.flow_region_capacity import FlowRegionCapacityModel  # noqa: E402
from simulator.flow_region_congestion import FlowRegionCongestionModel  # noqa: E402

from research_framework.statistics import (  # noqa: E402
    confidence_interval, effect_size_cohens_d, paired_comparison,
)


PUBLISHED_EVACUATION_TIME_S = {
    "10-story": 1022.0,
    "18-story": 1192.0,
    "24-story": 1090.0,
    "31-story": 1002.0,
}

BUILDINGS = {
    "10-story": (build_nist_10story_building, build_nist_10story_definition, DEFINITION_ID_10, MASTER_SEED_10),
    "18-story": (build_nist_18story_building, build_nist_18story_definition, DEFINITION_ID_18, MASTER_SEED_18),
    "24-story": (build_nist_24story_building, build_nist_24story_definition, DEFINITION_ID_24, MASTER_SEED_24),
    "31-story": (build_nist_31story_building, build_nist_31story_definition, DEFINITION_ID_31, MASTER_SEED_31),
}


# =====================================================
# Per-run metric extraction -- reads already-produced fields only.
# =====================================================


def _all_steps(movement_result):

    for timeline in movement_result.occupants.values():
        for step in timeline.steps:
            yield step


def _queue_wait_stats(movement_result):

    waits = [getattr(step, "queue_wait_time", 0.0) or 0.0 for step in _all_steps(movement_result)]

    if not waits:
        return {"mean": 0.0, "max": 0.0}

    return {"mean": pystats.fmean(waits), "max": max(waits)}


def _queue_wait_percentage_population_mean(movement_result):

    percentages = []

    for timeline in movement_result.occupants.values():

        if timeline.arrival_time is None or not timeline.steps:
            continue

        total_travel_time = timeline.arrival_time - timeline.depart_time

        if total_travel_time <= 0:
            continue

        total_wait = sum(getattr(s, "queue_wait_time", 0.0) or 0.0 for s in timeline.steps)
        percentages.append(100.0 * total_wait / total_travel_time)

    return pystats.fmean(percentages) if percentages else None


def _per_edge_queue_wait_totals(movement_result):

    totals = {}

    for step in _all_steps(movement_result):

        wait = getattr(step, "queue_wait_time", 0.0) or 0.0
        totals[step.edge.id] = totals.get(step.edge.id, 0.0) + wait

    return totals


def _per_edge_crossing_counts(movement_result, edge_type):

    counts = {}

    for step in _all_steps(movement_result):

        if step.edge.edge_type == edge_type:
            counts[step.edge.id] = counts.get(step.edge.id, 0) + 1

    return counts


def _edge_label(edge_id, building_copy):

    for floor in building_copy.ordered_floors():

        for door in floor.doors:
            if door.id == edge_id:
                return f"Door '{door.name}' ({edge_id})"

        for exit_obj in floor.exits:
            if exit_obj.id == edge_id:
                return f"Exit '{exit_obj.name}' ({edge_id})"

        for stair in floor.stairs:
            if stair.id == edge_id:
                return f"Stair '{stair.name}' ({edge_id})"

    return edge_id


def _record(scenario, ground_truth, movement_result, building_copy):

    queue_wait_stats = _queue_wait_stats(movement_result)
    per_edge_wait = _per_edge_queue_wait_totals(movement_result)

    dominant_bottleneck_id = max(per_edge_wait, key=per_edge_wait.get) if per_edge_wait else None

    return {
        "scenario_id": scenario.metadata.scenario_id,
        "total_evacuation_time": ground_truth.total_evacuation_time,
        "people_evacuated": ground_truth.people_evacuated,
        "people_trapped": ground_truth.people_trapped,
        "peak_congestion_value": ground_truth.peak_congestion_value,
        "congestion_duration": ground_truth.congestion_duration,
        "exit_utilization_balance": compute_exit_utilization_balance(ground_truth, building_copy),
        "avg_queue_wait_time": queue_wait_stats["mean"],
        "max_queue_wait_time": queue_wait_stats["max"],
        "queue_wait_percentage": _queue_wait_percentage_population_mean(movement_result),
        "dominant_bottleneck_id": dominant_bottleneck_id,
        "dominant_bottleneck_label": (
            _edge_label(dominant_bottleneck_id, building_copy) if dominant_bottleneck_id else None
        ),
        "dominant_bottleneck_total_wait": per_edge_wait.get(dominant_bottleneck_id) if dominant_bottleneck_id else None,
        "exit_flow": _per_edge_crossing_counts(movement_result, Edge.EXIT),
        "stair_flow": _per_edge_crossing_counts(movement_result, Edge.STAIR),
    }


# =====================================================
# Per-building campaign -- one seeded scenario batch, two arms each.
# =====================================================


def run_building_comparison(name, n_seeds, dt=1.0):

    build_building, build_definition, definition_id, master_seed = BUILDINGS[name]

    building = build_building()
    definition = build_definition()

    batch = run_batch_pipeline(definition, definition_id, building, master_seed, n_seeds)

    baseline_records, candidate_records = [], []

    for scenario in batch.scenarios:

        baseline_movement, baseline_gt, baseline_building_copy = run_with_overrides(
            scenario, building,
            capacity_model=StairCapacityModel(), congestion_model=StairAwareCongestionModel(),
            dt=dt, use_flow_regions=False,
        )
        baseline_records.append(_record(scenario, baseline_gt, baseline_movement, baseline_building_copy))

        candidate_movement, candidate_gt, candidate_building_copy = run_with_overrides(
            scenario, building,
            capacity_model=FlowRegionCapacityModel(), congestion_model=FlowRegionCongestionModel(),
            dt=dt, use_flow_regions=True,
        )
        candidate_records.append(_record(scenario, candidate_gt, candidate_movement, candidate_building_copy))

    return baseline_records, candidate_records


# =====================================================
# Aggregation and paired statistics -- reuses
# research_framework.statistics exactly as calibration_benchmark/
# harness.py itself does, for direct methodological consistency.
# =====================================================


NUMERIC_FIELDS = (
    "total_evacuation_time", "avg_queue_wait_time", "max_queue_wait_time",
    "queue_wait_percentage", "peak_congestion_value", "congestion_duration",
    "exit_utilization_balance", "people_evacuated", "people_trapped",
)


def _paired_non_none(baseline_values, candidate_values):

    paired_a, paired_b = [], []

    for a, b in zip(baseline_values, candidate_values):

        if a is not None and b is not None:
            paired_a.append(float(a))
            paired_b.append(float(b))

    return paired_a, paired_b


def _compare_field(field_name, baseline_records, candidate_records):

    baseline_values = [r[field_name] for r in baseline_records]
    candidate_values = [r[field_name] for r in candidate_records]

    paired_a, paired_b = _paired_non_none(baseline_values, candidate_values)
    n = len(paired_a)

    return {
        "n_pairs": n,
        "baseline_mean": (sum(paired_a) / n) if n else None,
        "candidate_mean": (sum(paired_b) / n) if n else None,
        "baseline_ci": confidence_interval(paired_a).to_dict() if n else None,
        "candidate_ci": confidence_interval(paired_b).to_dict() if n else None,
        "paired": paired_comparison(paired_a, paired_b).to_dict() if n >= 2 else None,
        "effect_size_cohens_d": effect_size_cohens_d(paired_b, paired_a).to_dict() if n >= 2 else None,
    }


def _dominant_bottleneck_mode(records):

    labels = [r["dominant_bottleneck_label"] for r in records if r["dominant_bottleneck_label"]]

    if not labels:
        return None

    return pystats.mode(labels)


def _aggregate_flow(records, key):

    totals = {}

    for record in records:
        for edge_id, count in record[key].items():
            totals[edge_id] = totals.get(edge_id, 0) + count

    n = len(records) or 1

    return {edge_id: total / n for edge_id, total in totals.items()}


def summarize_building(name, baseline_records, candidate_records):

    published = PUBLISHED_EVACUATION_TIME_S[name]

    comparisons = {
        field: _compare_field(field, baseline_records, candidate_records)
        for field in NUMERIC_FIELDS
    }

    baseline_evac_mean = comparisons["total_evacuation_time"]["baseline_mean"]
    candidate_evac_mean = comparisons["total_evacuation_time"]["candidate_mean"]

    return {
        "building": name,
        "published_evacuation_time_s": published,
        "n_seeds": len(baseline_records),
        "baseline_overprediction_ratio": (
            baseline_evac_mean / published if baseline_evac_mean is not None else None
        ),
        "candidate_overprediction_ratio": (
            candidate_evac_mean / published if candidate_evac_mean is not None else None
        ),
        "comparisons": comparisons,
        "baseline_dominant_bottleneck": _dominant_bottleneck_mode(baseline_records),
        "candidate_dominant_bottleneck": _dominant_bottleneck_mode(candidate_records),
        "baseline_exit_flow_mean": _aggregate_flow(baseline_records, "exit_flow"),
        "candidate_exit_flow_mean": _aggregate_flow(candidate_records, "exit_flow"),
        "baseline_stair_flow_mean": _aggregate_flow(baseline_records, "stair_flow"),
        "candidate_stair_flow_mean": _aggregate_flow(candidate_records, "stair_flow"),
    }


def main():

    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    dt = 1.0

    output_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "architecture")
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for name in BUILDINGS:

        print(f"Running {name} ({n_seeds} seeds x 2 arms)...", flush=True)

        baseline_records, candidate_records = run_building_comparison(name, n_seeds, dt=dt)
        summary = summarize_building(name, baseline_records, candidate_records)

        all_results[name] = {
            "summary": summary,
            "baseline_records": baseline_records,
            "candidate_records": candidate_records,
        }

        print(json.dumps(summary, indent=2, default=str))

    with open(
        os.path.join(output_dir, "flow_region_nist_validation_v5_raw_results.json"), "w", encoding="utf-8",
    ) as handle:
        json.dump(all_results, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
